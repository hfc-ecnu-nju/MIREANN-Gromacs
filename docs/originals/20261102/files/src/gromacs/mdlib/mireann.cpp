/* GMX_MIREANN NEW FILE: native GROMACS neighbor/PBC selection and C-ABI inference.
 * Local extension, LGPL-2.1-or-later.
 */
#include "gmxpre.h"
#include "mireann.h"
#include "mireann_abi.h"
#include <chrono>
#include <cstdio>
#include <numeric>
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/selection/nbsearch.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/keyvaluetree.h"
#if !defined(_WIN32)
#include <dlfcn.h>
#endif
namespace gmx::dynamiccharge
{
namespace
{
#if !defined(_WIN32)
struct MireLibrary
{
    explicit MireLibrary(const std::string& path)
    {
        int flags=RTLD_NOW|RTLD_LOCAL;
#ifdef RTLD_NODELETE
        flags|=RTLD_NODELETE;
#endif
#ifdef RTLD_DEEPBIND
        flags|=RTLD_DEEPBIND;
#endif
        pointer=dlopen(path.c_str(),flags);
        if (!pointer) { throw std::runtime_error(std::string("Cannot load MIREANN backend: ")+dlerror()); }
    }
    ~MireLibrary() { if (pointer) { dlclose(pointer); } }
    template<class T> T symbol(const char* name)
    {
        dlerror(); auto* result=dlsym(pointer,name);
        if (const char* error=dlerror()) { throw std::runtime_error(error); }
        return reinterpret_cast<T>(result);
    }
    void* pointer=nullptr;
};
class Predictor final : public MireannPredictor
{
public:
    Predictor(const Frame& frame,const QeqOptions& options,const std::string& path) :
        library_(path), predict_(library_.symbol<decltype(&gmx_mireann_predict)>("gmx_mireann_predict")),
        finish_(library_.symbol<decltype(&gmx_mireann_forces)>("gmx_mireann_forces")),
        classical_(library_.symbol<decltype(&gmx_mireann_classical)>("gmx_mireann_classical")),
        model_(nullptr,library_.symbol<decltype(&gmx_mireann_destroy)>("gmx_mireann_destroy")),
        options_(options), region_(frame.region)
    {
        const auto found=std::find(region_.begin(),region_.end(),options_.centerAtom-1);
        if (found==region_.end()) { throw std::runtime_error("MIREANN center atom is outside QM region"); }
        env_.centerRegionIndex=static_cast<int>(found-region_.begin());
        // GMX_MIREANN RESIDUE: cache immutable, global topology membership once.
        // Avoid parsing a full-system metadata array at every MD force call.
        if (options_.mmSelection=="residue")
        {
            if (!options_.topologyParameters || !options_.topologyParameters->keyExists("mireann-residue-ids"))
            { throw std::runtime_error("MIREANN residue selection requires topology metadata; re-run grompp with this version"); }
            const auto values=(*options_.topologyParameters)["mireann-residue-ids"].asArray().values();
            if (values.size()!=frame.positionsNm.size())
            { throw std::runtime_error("MIREANN residue metadata atom count mismatch"); }
            int maximum=-1;
            for (const auto& value : values)
            {
                const auto id=value.cast<std::int64_t>();
                if (id<0 || id>=static_cast<std::int64_t>(values.size()))
                { throw std::runtime_error("Invalid MIREANN topology residue ID"); }
                residueOfAtom_.push_back(static_cast<int>(id));
                maximum=std::max(maximum,static_cast<int>(id));
            }
            residueAtoms_.resize(maximum+1);
            for (size_t atom=0; atom<residueOfAtom_.size(); ++atom)
            { residueAtoms_[residueOfAtom_[atom]].push_back(static_cast<int>(atom)); }
        }
        // GMX_MIREANN_PSEUDOBOND: zero background sites keep their true FF
        // charges. This is not a global charge deletion or a mass/Z mutation.
        if (!options_.zeroAtoms.empty())
        {
            zeroAtoms_=parseQeqAtoms(options_.zeroAtoms);
            for (int atom : zeroAtoms_)
            {
                if (atom<0 || atom>=static_cast<int>(frame.positionsNm.size())
                    || std::find(region_.begin(),region_.end(),atom)!=region_.end())
                { throw std::runtime_error("Invalid MIREANN zero-background MM atom"); }
            }
        }
        search_.setCutoff(static_cast<real>(options_.mmCutoffAngstrom*0.1));
        // Independent environment list, not the shorter nbnxm force pair list.
        search_.setMode(AnalysisNeighborhood::eSearchMode_Automatic);
        std::vector<double> masses;
        for (int id:region_) { masses.push_back(frame.massesU.at(id)); }
        GmxQeqOptions create{};
        create.abiVersion=1; create.numAtoms=static_cast<int>(region_.size());
        create.modelPath=options_.model.c_str(); create.device=options_.device.c_str();
        create.elements=options_.elements.c_str(); create.massesU=masses.data();
        create.torchThreads=options_.threads;
        char error[4096]{}; double total=0;
        model_.reset(library_.symbol<decltype(&gmx_mireann_create)>("gmx_mireann_create")(&create,&total,error,sizeof(error)));
        if (!model_) { throw std::runtime_error(error); }
        double initial=0; for (int id:region_) { initial+=frame.chargesE.at(id); }
        if (!std::isfinite(total) || std::abs(total-initial)>1e-5)
        { throw std::runtime_error("Initial QM topology charge must equal the MIREANN model total charge; no implicit projection is applied"); }
        // GMX_MIREANN_PHYSICS_AUDIT: distinguish full subtraction environment
        // from the compact background actually supplied to the model.
        open(energyLog_,options_.energyLog,"step,time_ps,mm_atoms,model_energy_kJ_mol,subtracted_coulomb_kJ_mol,net_correction_kJ_mol,model_mm_atoms\n");
        // ORIGINAL columns ended with q_e,embedding_weight. Keep them, append
        // explicit model membership and the actual weighted background input.
        open(environmentLog_,options_.environmentLog,"step,time_ps,atom_1based,x_A,y_A,z_A,q_e,embedding_weight,in_model,model_q_e\n");
        open(profile_,options_.profile,"");
        // GMX_MIREANN_CROSS14: report the actual TPR force convention.
        // ORIGINAL message unconditionally said: charge-response VJP enabled.
        std::fprintf(stderr,"GMX_MIREANN: seven-input model, %zu QM atoms; center=%d, radius=%.9g A, switch=%.9g A; device=%s. Native GROMACS neighbor/PBC search; global PME minus Amber-style local Coulomb/erf reference; residual charge-response=%s.\n",
                     region_.size(),options_.centerAtom,options_.mmCutoffAngstrom,options_.switchWidthAngstrom,options_.device.c_str(),options_.chargeResponse ? "yes" : "no");
        std::fprintf(stderr,"GMX_MIREANN RESIDUE: mm-selection=%s; residue includes every MM atom if any MM atom enters the center sphere.\n",options_.mmSelection.c_str());
    }
    ~Predictor() override
    {
        if (profile_.is_open())
        {
            profile_<<"calls,warmup_calls,measured_calls,mean_model_ms,mean_selection_ms,mean_response_ms,gpu_classical_measured_calls,mean_gpu_classical_ms\n"
                    <<calls_<<','<<std::min(calls_,20LL)<<','<<measured_<<','
                    <<(measured_?modelMs_/measured_:0)<<','<<(measured_?selectionMs_/measured_:0)
                    <<','<<(measured_?responseMs_/measured_:0)<<','<<classicalMeasured_
                    <<','<<(classicalMeasured_?classicalMs_/classicalMeasured_:0)<<'\n';
        }
    }
    std::vector<double> predict(const Frame& frame) override
    {
        const auto start=std::chrono::steady_clock::now();
        matrix box;
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) { box[i][j]=static_cast<real>(frame.boxNm[i][j]); }
        t_pbc pbc; set_pbc(&pbc,PbcType::Xyz,box);
        coordinates_.resize(frame.positionsNm.size());
        for (size_t i=0;i<coordinates_.size();++i) for (int d=0;d<3;++d)
        { coordinates_[i][d]=static_cast<real>(frame.positionsNm[i][d]); }
        const auto& center=coordinates_.at(options_.centerAtom-1);
        env_.qmNm.resize(region_.size());
        qmInput_.resize(region_.size()*3);
        for (size_t i=0;i<region_.size();++i)
        {
            rvec dx; pbc_dx(&pbc,coordinates_[region_[i]],center,dx);
            for (int d=0;d<3;++d) { env_.qmNm[i][d]=dx[d]; qmInput_[3*i+d]=10.0*dx[d]; }
        }
        // GMX_MIREANN_PHYSICS_AUDIT: the supplied model images QM sites around
        // its FIRST input before shifting the COM. If that operation changes
        // an image chosen around our requested center, the model and reference
        // would use different QM-MM separations. Reject ambiguous large QM
        // regions instead of silently combining two Hamiltonians.
        for(size_t i=1;i<region_.size();++i)
        {
            const double z=(env_.qmNm[i][2]-env_.qmNm[0][2])/box[2][2];
            const double y=(env_.qmNm[i][1]-env_.qmNm[0][1]-z*box[2][1])/box[1][1];
            const double x=(env_.qmNm[i][0]-env_.qmNm[0][0]-y*box[1][0]-z*box[2][0])/box[0][0];
            if(std::nearbyint(x)!=0 || std::nearbyint(y)!=0 || std::nearbyint(z)!=0)
            { throw std::runtime_error("MIREANN QM region does not fit in one consistent model image; enlarge the box or change the QM region/center"); }
        }
        // A fresh search every force call is deliberate: the environment must follow
        // current positions even between GROMACS nstlist steps. No stale-radius list.
        const auto initialized=search_.initSearch(&pbc,coordinates_);
        auto pairs=initialized.startPairSearch(center.as_vec());
        AnalysisNeighborhoodPair pair;
        std::vector<std::pair<int,std::array<double,3>>> selected;
        while (pairs.findNextPair(&pair))
        {
            const int id=pair.refIndex();
            if (std::find(region_.begin(),region_.end(),id)!=region_.end()) { continue; }
            std::array<double,3> image;
            for (int d=0;d<3;++d) { image[d]=pair.dx()[d]; }
            selected.emplace_back(id,image);
        }
        // GMX_MIREANN RESIDUE: the native radius search above finds seed atoms.
        // Expand seeds to complete residues, choosing the closest seed (global
        // atom ID breaks ties) as that residue's image anchor. Other atoms are
        // imaged relative to the anchor, so a water crossing a box face stays
        // whole. Outer residue atoms are deliberately allowed beyond 25 A.
        // Existing atom-sphere implementation remains available with mode=atom.
        if (options_.mmSelection=="residue")
        {
            std::vector<int> anchor(residueAtoms_.size(),-1);
            std::vector<std::array<double,3>> anchorImage(residueAtoms_.size());
            auto squared=[](const std::array<double,3>& v) { return v[0]*v[0]+v[1]*v[1]+v[2]*v[2]; };
            for (const auto& seed : selected)
            {
                const int residue=residueOfAtom_.at(seed.first);
                if (anchor[residue]<0 || squared(seed.second)<squared(anchorImage[residue])
                    || (squared(seed.second)==squared(anchorImage[residue]) && seed.first<anchor[residue]))
                { anchor[residue]=seed.first; anchorImage[residue]=seed.second; }
            }
            selected.clear();
            for (size_t residue=0; residue<anchor.size(); ++residue)
            {
                if (anchor[residue]<0) { continue; }
                for (int atom : residueAtoms_[residue])
                {
                    if (std::find(region_.begin(),region_.end(),atom)!=region_.end()) { continue; }
                    rvec relative; pbc_dx(&pbc,coordinates_[atom],coordinates_[anchor[residue]],relative);
                    std::array<double,3> image;
                    for (int d=0; d<3; ++d) { image[d]=anchorImage[residue][d]+relative[d]; }
                    selected.emplace_back(atom,image);
                }
            }
        }
        // Stable input/force order independent of neighbor-search grid traversal.
        std::sort(selected.begin(),selected.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        env_.mmGlobal.clear(); env_.mmNm.clear(); env_.mmChargesE.clear();
        env_.weights.clear(); env_.weightDerivativePerNm.clear();
        mmInput_.clear(); mmChargeInput_.clear();
        modelToEnvironment_.clear(); // GMX_MIREANN_PHYSICS_AUDIT: compact zero-mask background
        // Explicit zero sites must be in the model environment: otherwise a
        // near-boundary replacement could silently depend on the cutoff.
        for (int atom : zeroAtoms_)
        {
            if (std::none_of(selected.begin(),selected.end(),[atom](const auto& v){return v.first==atom;}))
            { throw std::runtime_error("MIREANN zero-background atom is outside the selected MM environment"); }
        }
        for (const auto& entry:selected)
        {
            const int id=entry.first;
            const auto& image=entry.second;
            if (!env_.mmGlobal.empty() && env_.mmGlobal.back()==id)
            { throw std::runtime_error("MIREANN neighbor search returned duplicate atom images"); }
            const double radius=std::sqrt(image[0]*image[0]+image[1]*image[1]+image[2]*image[2]);
            double weight=1,derivative=0;
            const double cutoff=options_.mmCutoffAngstrom*0.1, width=options_.switchWidthAngstrom*0.1;
            if (width>0 && radius>cutoff-width)
            {
                const double t=std::clamp((radius-cutoff+width)/width,0.0,1.0);
                weight=1-10*t*t*t+15*t*t*t*t-6*t*t*t*t*t;
                derivative=(-30*t*t+60*t*t*t-30*t*t*t*t)/width;
            }
            env_.mmGlobal.push_back(id); env_.mmNm.push_back(image);
            env_.mmChargesE.push_back(frame.chargesE.at(id));
            env_.weights.push_back(weight); env_.weightDerivativePerNm.push_back(derivative);
            // ORIGINAL | for (double value:image) { mmInput_.push_back(10.0*value); }
            // GMX_MIREANN_PSEUDOBOND: model background only. env_.mmChargesE
            // retains physical q for the FF reference subtraction and VJP.
            const bool zero=std::find(zeroAtoms_.begin(),zeroAtoms_.end(),id)!=zeroAtoms_.end();
            // ORIGINAL | mmChargeInput_.push_back(frame.chargesE.at(id)*weight);
            // ORIGINAL | mmChargeInput_.push_back(zero ? 0.0 : frame.chargesE.at(id)*weight);
            // GMX_MIREANN_PHYSICS_AUDIT: zero-mask sites do not enter the model
            // background list at all, matching the Amber list convention. Keep
            // them in env_ for physical FF subtraction, with zero model force.
            if(!zero)
            {
                modelToEnvironment_.push_back(static_cast<int>(env_.mmGlobal.size()-1));
                for(double value:image){mmInput_.push_back(10.0*value);}
                mmChargeInput_.push_back(frame.chargesE.at(id)*weight);
            }
        }
        double cell[9];
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) { cell[3*i+j]=10.0*frame.boxNm[i][j]; }
        const double selection=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::vector<double> result(region_.size());
        char error[4096]{}; double modelEnergy=0,milliseconds=0;
        // ORIGINAL | predict_(model_.get(),static_cast<int>(env_.mmGlobal.size()),...)
        if (predict_(model_.get(),static_cast<int>(modelToEnvironment_.size()),qmInput_.data(),mmInput_.data(),
                     mmChargeInput_.data(),cell,result.data(),&modelEnergy,&milliseconds,error,sizeof(error)))
        { throw std::runtime_error(error); }
        env_.modelEnergyKj=modelEnergy*options_.energyScale;
        if (!std::isfinite(env_.modelEnergyKj)) { throw std::runtime_error("Nonfinite MIREANN energy after unit conversion"); }
        ++calls_;
        if (calls_>20) { ++measured_; modelMs_+=milliseconds; selectionMs_+=selection; }
        return result;
    }
    const MireannEnvironment& environment() const override { return env_; }
    MireannForces forces(const std::vector<double>& potential) override
    {
        const auto start=std::chrono::steady_clock::now();
        MireannForces result;
        result.qm.resize(region_.size()); result.mm.resize(env_.mmGlobal.size());
        result.backgroundDerivative.resize(env_.mmGlobal.size());
        // GMX_MIREANN_PHYSICS_AUDIT: returned model rows follow the compact list.
        std::vector<std::array<double,3>> modelMm(modelToEnvironment_.size());
        std::vector<double> modelBackground(modelToEnvironment_.size());
        char error[4096]{};
        if (finish_(model_.get(),potential.data(),options_.energyScale,
                    reinterpret_cast<double*>(result.qm.data()),reinterpret_cast<double*>(modelMm.data()),
                    modelBackground.data(),error,sizeof(error)))
        { throw std::runtime_error(error); }
        // ORIGINAL output destinations: result.mm.data(), result.backgroundDerivative.data().
        for(size_t j=0;j<modelToEnvironment_.size();++j)
        {
            result.mm[modelToEnvironment_[j]]=modelMm[j];
            result.backgroundDerivative[modelToEnvironment_[j]]=modelBackground[j];
        }
        if (calls_>20) { responseMs_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); }
        return result;
    }
    bool classical(const GmxMireannClassicalInput& input,GmxMireannClassicalOutput& output) override
    {
        const auto start=std::chrono::steady_clock::now();
        int usedGpu=0;char error[4096]{};
        if(classical_(model_.get(),&input,&output,&usedGpu,error,sizeof(error)))
        { throw std::runtime_error(error); }
        if(usedGpu && calls_>20)
        {
            ++classicalMeasured_;
            classicalMs_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        if(!classicalReported_)
        {
            std::fprintf(stderr,"GMX_MIREANN_GPU_AUDIT: classical charge-potential and reference pairs=%s; PME grid sampled separately; host integration unchanged.\n",usedGpu ? "CUDA" : "CPU fallback (backend has no CUDA classical kernels)");
            classicalReported_=true;
        }
        return usedGpu!=0;
    }
    void record(const Frame& frame,double bareEnergy) override
    {
        if (energyLog_.is_open())
        {
            energyLog_<<frame.step<<','<<frame.timePs<<','<<env_.mmGlobal.size()<<','
                      <<env_.modelEnergyKj<<','<<bareEnergy<<','<<env_.modelEnergyKj-bareEnergy
                      <<','<<modelToEnvironment_.size()<<'\n';
            energyLog_.flush();
        }
        if (environmentLog_.is_open())
        {
            for (size_t i=0;i<env_.mmGlobal.size();++i)
            {
                environmentLog_<<frame.step<<','<<frame.timePs<<','<<env_.mmGlobal[i]+1;
                for (double coordinate:env_.mmNm[i]) { environmentLog_<<','<<coordinate*10; }
                const bool inModel=std::find(zeroAtoms_.begin(),zeroAtoms_.end(),env_.mmGlobal[i])==zeroAtoms_.end();
                // ORIGINAL | environmentLog_<<','<<env_.mmChargesE[i]<<','<<env_.weights[i]<<'\n';
                environmentLog_<<','<<env_.mmChargesE[i]<<','<<env_.weights[i]<<','<<inModel
                    <<','<<(inModel ? env_.mmChargesE[i]*env_.weights[i] : 0.0)<<'\n';
            }
            environmentLog_.flush();
        }
        if ((energyLog_.is_open() && !energyLog_) || (environmentLog_.is_open() && !environmentLog_))
        { throw std::runtime_error("Failed writing MIREANN audit output"); }
    }
private:
    static void open(std::ofstream& stream,const std::string& path,const char* header)
    {
        if (path.empty()) { return; }
        stream.open(path);
        if (!stream) { throw std::runtime_error("Cannot open MIREANN output: "+path); }
        stream.precision(17); stream<<header;
    }
    MireLibrary library_;
    decltype(&gmx_mireann_predict) predict_;
    decltype(&gmx_mireann_forces) finish_;
    decltype(&gmx_mireann_classical) classical_; // GMX_MIREANN_GPU_AUDIT
    std::unique_ptr<void,decltype(&gmx_mireann_destroy)> model_;
    std::vector<int> zeroAtoms_; // GMX_MIREANN_PSEUDOBOND
    std::vector<int> modelToEnvironment_; // GMX_MIREANN_PHYSICS_AUDIT
    bool classicalReported_=false;
    QeqOptions options_;
    std::vector<int> region_;
    AnalysisNeighborhood search_;
    std::vector<RVec> coordinates_;
    std::vector<int> residueOfAtom_; // GMX_MIREANN: global topology, never GRO residue numbers.
    std::vector<std::vector<int>> residueAtoms_;
    MireannEnvironment env_;
    std::vector<double> qmInput_,mmInput_,mmChargeInput_;
    std::ofstream energyLog_,environmentLog_,profile_;
    long long calls_=0,measured_=0;
    double modelMs_=0,selectionMs_=0,responseMs_=0;
    long long classicalMeasured_=0;
    double classicalMs_=0;
};
#endif
}
std::unique_ptr<ChargePredictor> makeMireannPredictor(const Frame& frame,const QeqOptions& options,const std::string& backend)
{
#if !defined(_WIN32)
    return std::make_unique<Predictor>(frame,options,backend);
#else
    throw std::runtime_error("MIREANN currently requires Linux/WSL");
#endif
}
} // namespace gmx::dynamiccharge
