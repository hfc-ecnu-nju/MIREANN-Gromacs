// SPDX-License-Identifier: LGPL-2.1-or-later
/* GMX_DUAL_HAMILTONIAN. Both endpoints use native GROMACS force calculation.
 * MM driver: original topology and original charges, integrates mixed forces.
 * ML/MM worker: unchanged existing TPR/models/charge initialization/corrections,
 * evaluates requested coordinates in native rerun, never integrates its own MD.
 * Initial supported scope: Linux, one rank, CPU update, fixed-s NVE/NVT, no FEP,
 * no MTS, no virtual sites. GPU model/nonbonded/PME remain usable in the worker.
 */
#include "gmxpre.h"
#include "dualhamiltonian.h"
#include "dualhamiltonian_socket.h"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "dynamicchargeoptions.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/imdmodule.h"
#include "gromacs/mdtypes/imdpoptionprovider.h"
#include "gromacs/mdtypes/imdpoptionprovider_helpers.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/optionsection.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/keyvaluetree.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/keyvaluetreetransform.h"
#include "gromacs/utility/smalloc.h"
namespace gmx::dualhamiltonian
{
namespace
{
constexpr const char* section="dual-hamiltonian";
class Module final:public IMDModule,public IMdpOptionProvider
{
public:
    IMdpOptionProvider* mdpOptionProvider() override{return this;}
    IMDOutputProvider* outputProvider() override{return nullptr;}
    void initForceProviders(ForceProviders*) override{}
    void subscribeToPreProcessingNotifications(MDModulesNotifiers*) override{}
    void subscribeToSimulationSetupNotifications(MDModulesNotifiers*) override{}
    void subscribeToSimulationRunNotifications(MDModulesNotifiers*) override{}
    void initMdpTransform(IKeyValueTreeTransformRules* rules) override
    {
        addMdpTransformFromString<bool>(rules,&fromStdString<bool>,section,"active");
        addMdpTransformFromString<double>(rules,&fromStdString<double>,section,"s");
        for(const char* key:{"socket","energy-log"})
        {addMdpTransformFromString<std::string>(rules,[](std::string v){return v;},section,key);}
    }
    void initMdpOptions(IOptionsContainerWithSections* options) override
    {
        auto s=options->addSection(OptionSection(section));
        s.addOption(BooleanOption("active").store(&options_.active));
        s.addOption(DoubleOption("s").store(&options_.s));
        s.addOption(StringOption("socket").store(&options_.socket));
        s.addOption(StringOption("energy-log").store(&options_.energyLog));
    }
    void buildMdpOutput(KeyValueTreeObjectBuilder* b) const override
    {
        addMdpOutputValue(b,section,"active",options_.active);
        if(!options_.active){return;}
        if(!std::isfinite(options_.s)||options_.s<0||options_.s>1||options_.socket.empty())
        {GMX_THROW(InvalidInputError("dual-hamiltonian requires fixed s in [0,1] and a local socket path"));}
        addMdpOutputValue(b,section,"s",options_.s);
        addMdpOutputValue(b,section,"socket",options_.socket);
        addMdpOutputValue(b,section,"energy-log",options_.energyLog);
    }
private:Options options_;
};
std::vector<int> globalMapping(const t_mdatoms& atoms,const t_commrec& cr)
{
    std::vector<int> mapping(atoms.homenr,-1);
    for(int local=0;local<atoms.homenr;++local)
    {
        const int global=haveDDAtomOrdering(cr)?cr.dd->globalAtomIndices[local]:local;
        require(global>=0&&global<atoms.homenr&&mapping[global]<0,"invalid single-rank global atom mapping");
        mapping[global]=local;
    }
    return mapping;
}
Endpoint capture(const t_mdatoms& atoms,const std::vector<int>& globalToLocal,ArrayRef<const RVec> forces,const tensor virial,
                 const gmx_enerdata_t& energy,bool withVirial)
{
    Endpoint e;e.forces.reserve(3*atoms.homenr);e.charges.reserve(atoms.homenr);
    for(int i=0;i<atoms.homenr;++i)
    {
        const int local=globalToLocal[i];
        e.charges.push_back(atoms.chargeA[local]);
        for(int d=0;d<3;++d){e.forces.push_back(forces[local][d]);}
    }
    for(auto type:EnumerationWrapper<InteractionFunction>{}){e.terms.push_back(energy.term[type]);}
    for(auto type:EnumerationWrapper<NonBondedEnergyTerms>{})
    {for(real value:energy.grpp.energyGroupPairTerms[type]){e.pairs.push_back(value);}}
    if(withVirial){for(int i=0;i<3;++i)for(int j=0;j<3;++j){e.virial[3*i+j]=virial[i][j];}}
    e.validate(atoms.homenr);return e;
}
void restore(const Endpoint& e,const std::vector<int>& globalToLocal,ArrayRef<RVec> forces,tensor virial,gmx_enerdata_t* energy,bool withVirial)
{
    for(std::size_t i=0;i<e.charges.size();++i)for(int d=0;d<3;++d)
    {forces[globalToLocal[i]][d]=static_cast<real>(e.forces[3*i+d]);}
    std::size_t k=0;for(auto type:EnumerationWrapper<InteractionFunction>{}){energy->term[type]=static_cast<real>(e.terms.at(k++));}
    k=0;for(auto type:EnumerationWrapper<NonBondedEnergyTerms>{})
    {for(real& value:energy->grpp.energyGroupPairTerms[type]){value=static_cast<real>(e.pairs.at(k++));}}
    if(withVirial){for(int i=0;i<3;++i)for(int j=0;j<3;++j){virial[i][j]=static_cast<real>(e.virial[3*i+j]);}}
}
}
Options readOptions(const t_inputrec& ir)
{
    Options o;
    if(!ir.params||!ir.params->keyExists("applied-forces")){return o;}
    const auto& applied=(*ir.params)["applied-forces"].asObject();if(!applied.keyExists(section)){return o;}
    const auto& tree=applied[section].asObject();
    if(tree.keyExists("active")){o.active=tree["active"].cast<bool>();}
    if(!o.active){return o;}
    o.s=tree["s"].cast<double>();o.socket=tree["socket"].cast<std::string>();
    if(tree.keyExists("energy-log")){o.energyLog=tree["energy-log"].cast<std::string>();}
    require(std::isfinite(o.s)&&o.s>=0&&o.s<=1&&!o.socket.empty(),"invalid TPR options");return o;
}
std::unique_ptr<IMDModule> makeModule(){return std::make_unique<Module>();}
bool workerRequested(){const char* p=std::getenv("GMX_DUAL_HAMILTONIAN_WORKER");return p&&*p;}
class Worker
{
public:
    explicit Worker(const t_mdatoms& atoms,const t_commrec& cr):channel(std::getenv("GMX_DUAL_HAMILTONIAN_WORKER"),true)
    {
        const auto map=globalMapping(atoms,cr);
        for(int i=0;i<atoms.homenr;++i){rawCharges.push_back(atoms.chargeA[map[i]]);masses.push_back(atoms.massT[map[i]]);}
    }
    Channel channel;Request request;std::vector<double> rawCharges,masses;
};
std::shared_ptr<Worker> makeWorker(const t_mdatoms& atoms,const t_commrec& cr){return std::make_shared<Worker>(atoms,cr);}
double frameTime(const Worker& worker){return worker.request.time;}
bool nextFrame(Worker& w,t_trxframe* frame,int natoms)
{
    if(!w.channel.next(&w.request,natoms)){return false;}
    if(w.request.rawCharges!=w.rawCharges)
    {
        std::ostringstream message;message<<std::setprecision(17)<<"MM and ML/MM raw topology charge mismatch at step "<<w.request.step;
        for(std::size_t i=0;i<w.rawCharges.size();++i)
        {if(w.request.rawCharges[i]!=w.rawCharges[i]){message<<", atom "<<i+1<<": driver="<<w.request.rawCharges[i]<<", worker initial="<<w.rawCharges[i];break;}}
        require(false,message.str());
    }
    require(w.request.masses==w.masses,"endpoint atom masses/order differ");
    if(!frame->x){snew(frame->x,natoms);}
    frame->natoms=natoms;frame->bX=true;frame->bBox=true;frame->bStep=true;frame->bTime=true;
    frame->step=w.request.step;frame->time=w.request.time;
    for(int i=0;i<natoms;++i)for(int d=0;d<3;++d){frame->x[i][d]=static_cast<real>(w.request.coordinates[3*i+d]);}
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){frame->box[i][j]=static_cast<real>(w.request.box[3*i+j]);}
    return true;
}
class Client
{
public:
    explicit Client(const Options& o):options(o),channel(o.socket,false)
    {
        if(!o.energyLog.empty())
        {
            std::ifstream old(o.energyLog);const bool header=old.peek()==std::ifstream::traits_type::eof();
            log.open(o.energyLog,std::ios::app);require(log.good(),"cannot open endpoint energy log");
            log<<std::setprecision(17);
            if(header){log<<"step,time_ps,s,U_MM_kJ_mol,U_MLMM_kJ_mol,U_s_kJ_mol,dU_ds_kJ_mol,Q_MM_e,Q_MLMM_e,Q2_MM_e2,Q2_MLMM_e2\n";}
        }
    }
    Options options;Channel channel;ChargeBuffers charges;std::ofstream log;
};
void completeForce(t_forcerec& fr,const t_inputrec& ir,const t_mdatoms& atoms,const t_commrec& cr,
                   ArrayRef<const RVec> coordinates,ArrayRef<RVec> forces,
                   const matrix box,tensor virial,gmx_enerdata_t* energy,
                   std::int64_t step,double time,bool computeVirial)
{
    if(fr.dualHamiltonianWorker)
    {
        require(!readOptions(ir).active,"worker TPR must not recursively enable dual interpolation");
        fr.dualHamiltonianWorker->channel.respond(capture(atoms,globalMapping(atoms,cr),forces,virial,*energy,computeVirial),step,time);
        return;
    }
    const auto options=readOptions(ir);if(!options.active){return;}
    require(!workerRequested(),"force worker must be launched with -rerun");
    require(!dynamiccharge::readQeqOptions(ir).active,"MM driver must disable both qeq-reann and mireann");
    require(!ir.internalParameters||!ir.internalParameters->keyExists("nnpot-input-group"),"MM driver must disable NNPot and retain the original topology");
    if(!fr.dualHamiltonianClient){fr.dualHamiltonianClient=std::make_shared<Client>(options);}
    auto& client=*fr.dualHamiltonianClient;
    const auto map=globalMapping(atoms,cr);
    Request request;request.step=step;request.time=time;
    for(int i=0;i<atoms.homenr;++i)
    {
        const int local=map[i];
        request.rawCharges.push_back(atoms.chargeA[local]);request.masses.push_back(atoms.massT[local]);
        for(int d=0;d<3;++d){request.coordinates.push_back(coordinates[local][d]);}
    }
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){request.box[3*i+j]=box[i][j];}
    Endpoint mm=capture(atoms,map,forces,virial,*energy,computeVirial);
    Endpoint ml=client.channel.evaluate(request);
    client.charges.commit(mm.charges,ml.charges);
    const auto potential=static_cast<std::size_t>(InteractionFunction::PotentialEnergy);
    const double uMM=mm.terms.at(potential),uML=ml.terms.at(potential);
    mixInPlace(options.s,mm,ml);restore(mm,map,forces,virial,energy,computeVirial);
    if(client.log.is_open())
    {
        client.log<<step<<','<<time<<','<<options.s<<','<<uMM<<','<<uML<<','<<mix(options.s,uMM,uML)<<','<<(uML-uMM)
         <<','<<client.charges.sumMM()<<','<<client.charges.sumMLMM()<<','<<client.charges.squareMM()<<','<<client.charges.squareMLMM()<<'\n';
        client.log.flush();require(client.log.good(),"cannot write endpoint energy log");
    }
}
}
