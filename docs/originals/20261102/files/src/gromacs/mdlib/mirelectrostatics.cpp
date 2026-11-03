/* GMX_MIREANN NEW FILE: replace local QM-MM Coulomb, preserve full periodic PME.
 * Local extension, LGPL-2.1-or-later.
 * U = U_GROMACS(q_new) + U_MIREANN(local) - U_bare_local(q_new).
 * Force = fixed-q GROMACS force + model force - local fixed-q Coulomb force
 *         - (dq/dR)^T [dU_GROMACS/dq - dU_bare_local/dq].
 * The model force already includes charge response for U_MIREANN(local).
 * The final VJP supplies ONLY the response of U_GROMACS - U_bare_local;
 * adding the model's own local response a second time would be incorrect.
 * GMX_MIREANN SCOPE: the reference is 1/r inside the actual PME real-space
 * cutoff and erf(alpha*r)/r outside, evaluated for the model input images.
 * The outer force includes the Gaussian derivative. This follows the supplied
 * Amber energy prescription, not a complete periodic QM-MM Ewald subtraction.
 */
#include "gmxpre.h"
#include "mireann.h"
#include "mirecoulomb.h"
#include <cmath>
#include <numeric>
#include <unordered_set>
#include "gromacs/ewald/pme.h"
#include "gromacs/math/units.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/keyvaluetree.h"
namespace gmx::dynamiccharge
{
namespace
{
using Vec=std::array<double,3>;
static_assert(sizeof(Vec)==3*sizeof(double),"MIREANN C ABI requires contiguous xyz triples");
double norm2(const Vec& value) { return value[0]*value[0]+value[1]*value[1]+value[2]*value[2]; }
Vec difference(const Vec& a,const Vec& b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
void add(Vec& a,const Vec& b,double scale=1) { for (int d=0;d<3;++d) { a[d]+=scale*b[d]; } }
}
void applyMireannCorrection(RuntimeState& state,const t_inputrec& inputrec,const gmx_localtop_t& topology,
                            t_forcerec& fr,ForceWithVirial& output,gmx_enerdata_t* energy,bool computeVirial)
{
    auto* model=dynamic_cast<MireannPredictor*>(state.model.get());
    if (!model) { return; }
    if (inputrec.coulombtype!=CoulombInteractionType::Pme || inputrec.pbcType!=PbcType::Xyz
        || inputrec.ewald_geometry!=EwaldGeometry::Default || inputrec.epsilon_surface!=0
        || !fr.pmedata)
    { throw std::runtime_error("MIREANN requires periodic 3D PME with conducting boundaries (epsilon-surface=0)"); }
    if (fr.ic->coulomb.modifier!=InteractionModifiers::None
        && fr.ic->coulomb.modifier!=InteractionModifiers::PotShift)
    { throw std::runtime_error("MIREANN supports PME Coulomb modifiers None or Potential-shift"); }
    const auto& frame=state.frame;
    const auto& env=model->environment();
    // GMX_MIREANN: if a gas NNP is enabled, its topology replacement must cover
    // exactly the same physical QM atoms; otherwise intramolecular MM terms
    // could silently survive on part of the charge-producing region.
    if (inputrec.internalParameters && inputrec.internalParameters->keyExists("nnpot-input-group"))
    {
        std::vector<int> gasRegion;
        for (const auto& value : (*inputrec.internalParameters)["nnpot-input-group"].asArray().values())
        { gasRegion.push_back(static_cast<int>(value.cast<std::int64_t>())); }
        auto chargeRegion=frame.region;
        std::sort(gasRegion.begin(),gasRegion.end());
        std::sort(chargeRegion.begin(),chargeRegion.end());
        if (gasRegion!=chargeRegion)
        { throw std::runtime_error("MIREANN and the gas NNP must select the same QM atoms"); }
    }
    // GMX_MIREANN_PSEUDOBOND: both topology and electrostatic modules must
    // agree; mixing the default link-atom topology with this mode is invalid.
    const auto mireOptions=readMireannOptions(inputrec);
    const bool pseudo=mireOptions.boundary=="pseudo-bond";
    if (pseudo && inputrec.internalParameters
        && inputrec.internalParameters->keyExists("nnpot-input-group")
        && (!inputrec.internalParameters->keyExists("nnpot-pseudo-bond")
            || !(*inputrec.internalParameters)["nnpot-pseudo-bond"].cast<bool>()))
    { throw std::runtime_error("MIREANN pseudo-bond requires nnpot-boundary=pseudo-bond"); }
    const size_t nqm=frame.region.size(),nmm=env.mmGlobal.size();
    const double constant=fr.ic->coulomb.epsfac,alpha=fr.ic->coulomb.ewaldCoeff;
    const double potentialShift=fr.ic->coulomb.modifier==InteractionModifiers::PotShift
                                ? fr.ic->coulomb.ewaldShift : 0.0;
    const double invSqrtPi=1/std::sqrt(M_PI);
    matrix box;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) { box[i][j]=static_cast<real>(frame.boxNm[i][j]); }
    t_pbc pbc; set_pbc(&pbc,PbcType::Xyz,box);
    std::vector<RVec> probe(nqm);
    std::vector<double> phi(nqm),nearPhi(nqm,0);
    for (size_t i=0;i<nqm;++i) for (int d=0;d<3;++d)
    { probe[i][d]=static_cast<real>(frame.positionsNm[frame.region[i]][d]); }
    // Reuse the potential grid from THIS force evaluation. GPU grids are read
    // before launchGpuEndOfStepTasks clears them. Never run a second Ewald sum.
    gmx_pme_mireann_potentials(fr.pmedata,probe,box,phi);
    // GMX_MIREANN_GPU_AUDIT: CUDA computes short-range charge potentials and
    // the local reference on the inference GPU. Preserve the original CPU
    // path below, including native GROMACS triclinic PBC. Ordinary listed 1-4
    // terms remain on CPU (small topology lists, unchanged force definition).
    double bareEnergy=0;
    std::vector<Vec> bareQm(nqm,Vec{}),bareMm(nmm,Vec{});
    std::vector<double> bareUnweightedPerMm(nmm,0),gpuShort(nqm,0);
    bool gpuClassical=false;
    const bool rectangular=box[0][1]==0 && box[0][2]==0 && box[1][0]==0
        && box[1][2]==0 && box[2][0]==0 && box[2][1]==0;
    const auto classicalDevice=mireOptions.electrostaticsDevice=="auto"
        ? mireOptions.device : mireOptions.electrostaticsDevice;
    if(rectangular && classicalDevice.rfind("cuda:",0)==0
       && box[0][0]>2*fr.ic->coulomb.cutoff && box[1][1]>2*fr.ic->coulomb.cutoff
       && box[2][2]>2*fr.ic->coulomb.cutoff)
    {
        if(state.mireannExclusions.empty())
        {
            const size_t n=frame.positionsNm.size();
            state.mireannExclusions.assign(nqm*n,0);
            std::vector<int> localToGlobal(state.localCount,-1);
            for(size_t g=0;g<n;++g){localToGlobal[state.globalToLocal[g]]=static_cast<int>(g);}
            for(size_t i=0;i<nqm;++i)
            {
                for(int local:topology.excls[state.regionLocal[i]])
                {
                    if(local<0 || static_cast<size_t>(local)>=localToGlobal.size() || localToGlobal[local]<0)
                    { throw std::runtime_error("Invalid global CUDA exclusion mapping"); }
                    const int global=localToGlobal[local];
                    if(!pseudo && state.outsideRegion[global])
                    { throw std::runtime_error("MIREANN QM/MM boundary crosses topology exclusions; select complete unlinked QM molecules"); }
                    state.mireannExclusions[i*n+global]=1;
                }
            }
        }
        GmxMireannClassicalInput args{};
        args.numAtoms=static_cast<int>(frame.positionsNm.size());args.numQm=static_cast<int>(nqm);
        args.numMm=static_cast<int>(nmm);args.pseudoBond=pseudo;args.topologyRevision=state.mappingRevision;
        args.cudaDevice=std::stoi(classicalDevice.substr(5));
        args.positionsNm=reinterpret_cast<const double*>(frame.positionsNm.data());args.chargesE=frame.chargesE.data();
        args.qmGlobal=frame.region.data();args.exclusions=state.mireannExclusions.data();
        args.qmImageNm=reinterpret_cast<const double*>(env.qmNm.data());
        args.mmImageNm=reinterpret_cast<const double*>(env.mmNm.data());args.mmGlobal=env.mmGlobal.data();
        args.mmChargesE=env.mmChargesE.data();args.weights=env.weights.data();
        for(int d=0;d<3;++d){args.boxDiagonalNm[d]=box[d][d];}
        args.coulombConstant=constant;args.cutoffNm=fr.ic->coulomb.cutoff;
        args.alphaPerNm=alpha;args.potentialShiftPerNm=potentialShift;
        GmxMireannClassicalOutput result{gpuShort.data(),nearPhi.data(),
            reinterpret_cast<double*>(bareQm.data()),reinterpret_cast<double*>(bareMm.data()),bareUnweightedPerMm.data()};
        gpuClassical=model->classical(args,result);
        if(!gpuClassical && mireOptions.electrostaticsDevice!="auto")
        { throw std::runtime_error("Explicit CUDA classical corrections require a build with -DGMX_GPU=CUDA"); }
        if(gpuClassical)
        {
            for(size_t i=0;i<nqm;++i){bareEnergy+=frame.chargesE[frame.region[i]]*nearPhi[i];}
        }
    }
    const double volume=box[0][0]*box[1][1]*box[2][2];
    const double qsum=std::accumulate(frame.chargesE.begin(),frame.chargesE.end(),0.0);
    for (size_t i=0;i<nqm;++i)
    {
        const int global=frame.region[i],local=state.globalToLocal[global];
        const auto excluded=topology.excls[local];
        std::unordered_set<int> excludedLocal(excluded.begin(),excluded.end());
        // PME self and uniform-background charge correction derivatives.
        phi[i]-=2*constant*alpha*invSqrtPi*frame.chargesE[global];
        phi[i]-=M_PI*constant*qsum/(alpha*alpha*volume);
        if(gpuClassical){phi[i]+=gpuShort[i];continue;}
        for (size_t other=0;other<frame.positionsNm.size();++other)
        {
            if (static_cast<int>(other)==global) { continue; }
            const bool isExcluded=excludedLocal.count(state.globalToLocal[other])!=0;
            // GMX_MIREANN_PSEUDOBOND: crossed exclusions are intentional;
            // their PME correction is included immediately below.
            // ORIGINAL | if (isExcluded && state.outsideRegion[other])
            if (isExcluded && state.outsideRegion[other] && !pseudo)
            { throw std::runtime_error("MIREANN QM/MM boundary crosses topology exclusions; select complete unlinked QM molecules"); }
            RVec otherPosition;
            for (int d=0;d<3;++d) { otherPosition[d]=static_cast<real>(frame.positionsNm[other][d]); }
            rvec dx; pbc_dx(&pbc,probe[i],otherPosition,dx);
            const double r2=static_cast<double>(dx[0])*dx[0]+static_cast<double>(dx[1])*dx[1]+static_cast<double>(dx[2])*dx[2];
            if (r2<=1e-20) { throw std::runtime_error("Overlapping MIREANN electrostatic atoms"); }
            const double radius=std::sqrt(r2),q=frame.chargesE[other];
            if (isExcluded) { phi[i]-=constant*q*std::erf(alpha*radius)/radius; }
            else if (radius<fr.ic->coulomb.cutoff)
            {
                double potential=std::erfc(alpha*radius)/radius;
                if (fr.ic->coulomb.modifier==InteractionModifiers::PotShift) { potential-=fr.ic->coulomb.ewaldShift; }
                phi[i]+=constant*q*potential;
            }
        }
    }
    // GMX_MIREANN_CROSS14 BEGIN: accumulate the reference before the 1-4
    // loop. Excluded 1-2/1-3 pairs have no direct reference; ordinary cross
    // QM/MM 1-4 pairs DO have a scaled direct Coulomb reference, separate
    // from the screened, non-excluded local environment reference below.
    // GMX_MIREANN_GPU_AUDIT: original accumulators moved before CUDA dispatch:
    // double bareEnergy=0;
    // std::vector<Vec> bareQm(nqm,Vec{}),bareMm(nmm,Vec{});
    // std::vector<double> bareUnweightedPerMm(nmm,0);
    // Ordinary 1-4 pairs are separate from the exclusion-corrected PME energy.
    // Explicit-charge pair functions do not use the dynamically committed
    // topology charges. Reject them on QM sites instead of silently applying
    // the ordinary LJ14/fudgeQQ formula to a different Hamiltonian.
    for (const auto type : {InteractionFunction::LennardJonesCoulomb14Q,
                            InteractionFunction::LennardJonesCoulombNonBondedPairs})
    {
        const auto& list=topology.idef.il[type].iatoms;
        for (size_t entry=0;entry<list.size();entry+=3)
        {
            if (std::find(state.regionLocal.begin(),state.regionLocal.end(),list[entry+1])!=state.regionLocal.end()
                || std::find(state.regionLocal.begin(),state.regionLocal.end(),list[entry+2])!=state.regionLocal.end())
            { throw std::runtime_error("MIREANN does not support explicit-charge pair functions on QM atoms; use ordinary pairs with topology charges"); }
        }
    }
    const auto& pairs=topology.idef.il[InteractionFunction::LennardJones14].iatoms;
    for (size_t entry=0;entry<pairs.size();entry+=3)
    {
        const int aLocal=pairs[entry+1],bLocal=pairs[entry+2];
        // Skip pure-MM pairs before performing any global-index lookup.
        const auto qaLocal=std::find(state.regionLocal.begin(),state.regionLocal.end(),aLocal);
        const auto qbLocal=std::find(state.regionLocal.begin(),state.regionLocal.end(),bLocal);
        if (qaLocal==state.regionLocal.end() && qbLocal==state.regionLocal.end()) { continue; }
        auto a=std::find(state.globalToLocal.begin(),state.globalToLocal.end(),aLocal);
        auto b=std::find(state.globalToLocal.begin(),state.globalToLocal.end(),bLocal);
        if (a==state.globalToLocal.end() || b==state.globalToLocal.end()) { throw std::runtime_error("Invalid MIREANN 1-4 mapping"); }
        const int ag=static_cast<int>(a-state.globalToLocal.begin()),bg=static_cast<int>(b-state.globalToLocal.begin());
        const auto qa=std::find(frame.region.begin(),frame.region.end(),ag),qb=std::find(frame.region.begin(),frame.region.end(),bg);
        if (qa==frame.region.end() && qb==frame.region.end()) { continue; }
        RVec ax,bx;
        for (int d=0;d<3;++d) { ax[d]=static_cast<real>(frame.positionsNm[ag][d]); bx[d]=static_cast<real>(frame.positionsNm[bg][d]); }
        rvec dx; pbc_dx(&pbc,ax,bx,dx);
        const double radius=std::sqrt(static_cast<double>(dx[0])*dx[0]+static_cast<double>(dx[1])*dx[1]+static_cast<double>(dx[2])*dx[2]);
        if (qa!=frame.region.end()) { phi[qa-frame.region.begin()]+=constant*fr.fudgeQQ*frame.chargesE[bg]/radius; }
        if (qb!=frame.region.end()) { phi[qb-frame.region.begin()]+=constant*fr.fudgeQQ*frame.chargesE[ag]/radius; }
        // GMX_MIREANN_CROSS14: only one endpoint may be QM. Pure-MM and
        // pure-QM pairs never enter this subtraction. LJ-14 stays untouched.
        if (pseudo && ((qa!=frame.region.end()) != (qb!=frame.region.end())))
        {
            const bool aIsQm=qa!=frame.region.end();
            const size_t qi=aIsQm ? qa-frame.region.begin() : qb-frame.region.begin();
            const int mmGlobal=aIsQm ? bg : ag;
            const auto mm=std::find(env.mmGlobal.begin(),env.mmGlobal.end(),mmGlobal);
            if (mm==env.mmGlobal.end())
            { throw std::runtime_error("MIREANN cross-boundary 1-4 MM atom is outside the model environment; increase mm-cutoff"); }
            const size_t mj=mm-env.mmGlobal.begin();
            Vec displacement;
            for (int d=0;d<3;++d) { displacement[d]=aIsQm ? dx[d] : -dx[d]; }
            // No erf screening or distance switch: this is the ordinary
            // topology 1-4 Coulomb term C*fudgeQQ*qi*qj/r, at physical q(e).
            // Zero-background atoms keep physical FF charges here as well.
            const auto pair=directCoulombPair(displacement,
                frame.chargesE[frame.region[qi]],frame.chargesE[mmGlobal],constant*fr.fudgeQQ);
            bareEnergy+=pair.energyKjMol;
            nearPhi[qi]+=pair.potentialAtQmKjPerE;
            add(bareQm[qi],pair.forceOnQmKjNm);
            add(bareMm[mj],pair.forceOnQmKjNm,-1);
        }
    }
    // ORIGINAL accumulators moved before the 1-4 loop:
    // double bareEnergy=0;
    // std::vector<Vec> bareQm(nqm,Vec{}),bareMm(nmm,Vec{});
    // std::vector<double> bareUnweightedPerMm(nmm,0);
    // GMX_MIREANN_CROSS14 END
    for (size_t i=0;i<nqm;++i)
    {
        const double qi=frame.chargesE[frame.region[i]];
        // GMX_MIREANN_GPU_AUDIT: preserve this complete original CPU loop.
        // CUDA fills the same accumulators; do not subtract the reference twice.
        for (size_t j=0;!gpuClassical && j<nmm;++j)
        {
            // GMX_MIREANN_PSEUDOBOND: Amber make_qm_mm_list skips topology
            // exclusions (including normal 1-4 pairs). Do not subtract a
            // full 1/r here: scaled 1-4 Coulomb was subtracted above.
            // ORIGINAL comment: Keep their classical terms.
            if (pseudo)
            {
                const auto excluded=topology.excls[state.regionLocal[i]];
                const int mmLocal=state.globalToLocal[env.mmGlobal[j]];
                if (std::find(excluded.begin(),excluded.end(),mmLocal)!=excluded.end()) { continue; }
            }
            const Vec dx=difference(env.qmNm[i],env.mmNm[j]);
            // GMX_MIREANN: preserve the exact model input image and committed q(e).
            // ORIGINAL | const auto pair=directCoulombPair(dx,qi,env.mmChargesE[j],constant);
            const auto pair=embeddingCoulombPair(dx,qi,env.mmChargesE[j],constant,
                                                fr.ic->coulomb.cutoff,alpha,potentialShift);
            // ORIGINAL last argument: alpha (no matching PME potential shift).
            const double weight=env.weights[j];
            bareEnergy+=weight*pair.energyKjMol;
            bareUnweightedPerMm[j]+=pair.energyKjMol;
            nearPhi[i]+=weight*pair.potentialAtQmKjPerE;
            add(bareQm[i],pair.forceOnQmKjNm,weight);
            add(bareMm[j],pair.forceOnQmKjNm,-weight);
        }
        phi[i]-=nearPhi[i];
        if (!std::isfinite(phi[i])) { throw std::runtime_error("Nonfinite MIREANN electrostatic potential"); }
    }
    // GMX_MIREANN_CROSS14: phi already includes the subtraction of BOTH
    // non-excluded local and scaled 1-4 reference potentials. Thus the
    // VJP does not reintroduce the charge response of the removed 1-4 term.
    // Turning off the residual VJP preserves the model's own autograd force.
    if (!mireOptions.chargeResponse) { std::fill(phi.begin(),phi.end(),0.0); }
    // ORIGINAL | auto correction=model->forces(phi);
    auto correction=model->forces(phi);
    for (size_t i=0;i<nqm;++i) { add(correction.qm[i],bareQm[i],-1); }
    for (size_t j=0;j<nmm;++j)
    {
        add(correction.mm[j],bareMm[j],-1);
        // Optional smooth shell: effective input q_j=w(r_j)*q_j. Include its
        // derivative, with the opposite reaction on the selected center atom.
        const double derivative=env.weightDerivativePerNm[j];
        if (derivative!=0)
        {
            const double radius=std::sqrt(norm2(env.mmNm[j]));
            const double dEdw=correction.backgroundDerivative[j]*env.mmChargesE[j]-bareUnweightedPerMm[j];
            const double scale=-dEdw*derivative/radius;
            add(correction.mm[j],env.mmNm[j],scale);
            add(correction.qm[env.centerRegionIndex],env.mmNm[j],-scale);
        }
    }
    double virial[3][3]{};
    auto accumulate=[&](int global,const Vec& position,const Vec& force)
    {
        const int local=state.globalToLocal[global];
        for (int d=0;d<3;++d)
        {
            const real value=static_cast<real>(force[d]);
            if (!std::isfinite(value)) { throw std::runtime_error("MIREANN force exceeds GROMACS precision"); }
            output.force_[local][d]+=value;
        }
        if (computeVirial) for (int a=0;a<3;++a) for (int b=0;b<3;++b)
        { virial[a][b]-=0.5*position[a]*force[b]; }
    };
    for (size_t i=0;i<nqm;++i) { accumulate(frame.region[i],env.qmNm[i],correction.qm[i]); }
    for (size_t j=0;j<nmm;++j) { accumulate(env.mmGlobal[j],env.mmNm[j],correction.mm[j]); }
    matrix converted;
    for (int a=0;a<3;++a) for (int b=0;b<3;++b)
    {
        converted[a][b]=static_cast<real>(virial[a][b]);
        if (!std::isfinite(converted[a][b])) { throw std::runtime_error("MIREANN virial exceeds GROMACS precision"); }
    }
    output.addVirialContribution(converted);
    // Keep TPR/interaction enums unchanged. NN Potential contains gas PES (if
    // enabled) plus this net electrostatic replacement; raw terms are audited.
    energy->term[InteractionFunction::NeuralNetworkPotentialEnergy]+=static_cast<real>(env.modelEnergyKj-bareEnergy);
    model->record(frame,bareEnergy);
}
} // namespace gmx::dynamiccharge
