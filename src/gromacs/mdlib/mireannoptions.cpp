/* GMX_MIREANN NEW FILE: seven-input embedding MDP/TPR options.
 * Local extension, LGPL-2.1-or-later. Length option explicitly uses Angstrom.
 */
#include "gmxpre.h"
#include "dynamicchargeoptions.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include "gromacs/mdtypes/imdmodule.h"
#include "gromacs/mdtypes/imdpoptionprovider.h"
#include "gromacs/mdtypes/imdpoptionprovider_helpers.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/topology/topology.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/optionsection.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/keyvaluetree.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/keyvaluetreetransform.h"
#include "gromacs/utility/stringutil.h"
namespace gmx::dynamiccharge
{
namespace
{
constexpr const char* mireName="mireann";
bool validDevice(const std::string& text)
{
    if (text=="cpu") { return true; }
    return text.size()>5 && text.compare(0,5,"cuda:")==0
        && std::all_of(text.begin()+5,text.end(),[](unsigned char c){return c>='0' && c<='9';});
}
class MireannModule final : public IMDModule, public IMdpOptionProvider
{
public:
    IMdpOptionProvider* mdpOptionProvider() override { return this; }
    IMDOutputProvider* outputProvider() override { return nullptr; }
    void initForceProviders(ForceProviders*) override {}
    // GMX_MIREANN RESIDUE: obtain true topology residue IDs, including distinct
    // instances of repeated water molecules; GRO residue numbers can wrap.
    // ORIGINAL | void subscribeToPreProcessingNotifications(MDModulesNotifiers*) override {}
    void subscribeToPreProcessingNotifications(MDModulesNotifiers* notifiers) override
    {
        if (!options_.active) { return; }
        notifiers->preProcessingNotifier_.subscribe([this](gmx_mtop_t* top)
        {
            residueIds_.clear();
            int nextResidue=0;
            for (const auto& block : top->molblock)
            {
                const auto& atoms=top->moltype[block.type].atoms;
                for (int molecule=0; molecule<block.nmol; ++molecule)
                {
                    for (int atom=0; atom<atoms.nr; ++atom)
                    { residueIds_.push_back(nextResidue+atoms.atom[atom].resind); }
                    nextResidue+=atoms.nres;
                }
            }
            if (residueIds_.size()!=static_cast<size_t>(top->natoms))
            { GMX_THROW(InvalidInputError("MIREANN residue topology atom count mismatch")); }
        });
        notifiers->preProcessingNotifier_.subscribe([this](KeyValueTreeObjectBuilder builder)
        {
            if (residueIds_.empty())
            { GMX_THROW(InvalidInputError("MIREANN residue topology was not supplied before TPR writing")); }
            auto array=builder.addUniformArray<std::int64_t>("mireann-residue-ids");
            for (int id : residueIds_) { array.addValue(id); }
        });
    }
    void subscribeToSimulationSetupNotifications(MDModulesNotifiers*) override {}
    void subscribeToSimulationRunNotifications(MDModulesNotifiers*) override {}
    void initMdpTransform(IKeyValueTreeTransformRules* rules) override
    {
        addMdpTransformFromString<bool>(rules,&fromStdString<bool>,mireName,"active");
        // GMX_MIREANN_CROSS14: explicit, persisted force convention.
        addMdpTransformFromString<bool>(rules,&fromStdString<bool>,mireName,"charge-response");
        // ORIGINAL | for(const char* key:{"threads","center-atom"})
        for (const char* key : {"threads","center-atom","mm-list-nst"})
        { addMdpTransformFromString<int>(rules,&fromStdString<int>,mireName,key); }
        // ORIGINAL | for(const char* key:{"mm-cutoff","switch-width","energy-scale"})
        for (const char* key : {"mm-cutoff","mm-list-cutoff","switch-width","energy-scale"})
        { addMdpTransformFromString<double>(rules,&fromStdString<double>,mireName,key); }
        for (const char* key : {"model","atoms","elements","device","charge-log","profile","energy-log","environment-log","boundary","zero-atoms"})
        { addMdpTransformFromString<std::string>(rules,[](std::string value){return value;},mireName,key); }
        addMdpTransformFromString<std::string>(rules,[](std::string value){return value;},mireName,"electrostatics-device"); // GMX_MIREANN_GPU_AUDIT
        addMdpTransformFromString<std::string>(rules,[](std::string value){return value;},mireName,"mm-selection");
    }
    void initMdpOptions(IOptionsContainerWithSections* options) override
    {
        auto section=options->addSection(OptionSection(mireName));
        section.addOption(BooleanOption("active").store(&options_.active));
        section.addOption(BooleanOption("charge-response").store(&options_.chargeResponse));
        section.addOption(IntegerOption("threads").store(&options_.threads));
        section.addOption(IntegerOption("center-atom").store(&options_.centerAtom));
        section.addOption(DoubleOption("mm-cutoff").store(&options_.mmCutoffAngstrom));
        // GMX_MIREANN_BUFFERED_LIST: Angstrom and MD-step units.
        section.addOption(DoubleOption("mm-list-cutoff").store(&options_.mmListCutoffAngstrom));
        section.addOption(IntegerOption("mm-list-nst").store(&options_.mmListNst));
        section.addOption(DoubleOption("switch-width").store(&options_.switchWidthAngstrom));
        section.addOption(DoubleOption("energy-scale").store(&options_.energyScale));
        section.addOption(StringOption("mm-selection").store(&options_.mmSelection));
        section.addOption(StringOption("electrostatics-device").store(&options_.electrostaticsDevice));
        for (auto item : {std::pair{"model",&options_.model}, {"atoms",&options_.atoms},
               {"elements",&options_.elements},{"device",&options_.device},
               {"boundary",&options_.boundary},{"zero-atoms",&options_.zeroAtoms},
               {"charge-log",&options_.chargeLog},{"profile",&options_.profile},
               {"energy-log",&options_.energyLog},{"environment-log",&options_.environmentLog}})
        { section.addOption(StringOption(item.first).store(item.second)); }
    }
    void buildMdpOutput(KeyValueTreeObjectBuilder* builder) const override
    {
        addMdpOutputComment(builder,mireName,"module","; GMX_MIREANN: center-atom is global 1-based; mm-cutoff/switch-width in Angstrom; charges in e");
        addMdpOutputValue(builder,mireName,"active",options_.active);
        if (!options_.active) { return; }
        if(options_.electrostaticsDevice!="auto" && !validDevice(options_.electrostaticsDevice))
        { GMX_THROW(InvalidInputError("mireann-electrostatics-device must be auto, cpu or cuda:N")); }
        if (options_.mmSelection!="atom" && options_.mmSelection!="residue")
        { GMX_THROW(InvalidInputError("mireann-mm-selection must be atom or residue")); }
        // A per-atom switching weight would break a selected residue back into
        // partially weighted atoms. No unvalidated residue-switch force is used.
// ORIGINAL |         if (options_.mmSelection=="residue" && options_.switchWidthAngstrom!=0)
// ORIGINAL |         { GMX_THROW(InvalidInputError("MIREANN residue selection currently requires switch-width=0; use atom for the tested per-atom smooth shell")); }
        // GMX_MIREANN_RESIDUE_SWITCH: shared C2 residue weights and complete chain forces are now implemented.
#ifndef GMX_QEQ_REANN_BUILT
        GMX_THROW(InvalidInputError("mireann-active=yes requires -DGMX_QEQ_REANN=ON"));
#endif
        // GMX_MIREANN_PSEUDOBOND: explicit opt-in; old TPRs retain the old guard.
        if (options_.boundary!="whole-molecule" && options_.boundary!="pseudo-bond")
        { GMX_THROW(InvalidInputError("mireann-boundary must be whole-molecule or pseudo-bond")); }
// ORIGINAL |         if (options_.boundary=="pseudo-bond" && options_.switchWidthAngstrom!=0)
// ORIGINAL |         { GMX_THROW(InvalidInputError("pseudo-bond currently requires switch-width=0")); }
        // GMX_MIREANN_RESIDUE_SWITCH: keep unsupported per-atom pseudo-bond
        // switching rejected; crossed 1-4 sites must remain in the unit-weight core.
        if (options_.boundary=="pseudo-bond" && options_.switchWidthAngstrom!=0
            && options_.mmSelection!="residue")
        { GMX_THROW(InvalidInputError("pseudo-bond smooth switching requires mm-selection=residue")); }
        if (options_.boundary!="pseudo-bond" && !options_.zeroAtoms.empty())
        { GMX_THROW(InvalidInputError("mireann-zero-atoms requires boundary=pseudo-bond")); }
        const auto atoms=parseQeqAtoms(options_.atoms);
        if (!options_.zeroAtoms.empty())
        {
            for (int id : parseQeqAtoms(options_.zeroAtoms))
            {
                if (std::find(atoms.begin(),atoms.end(),id)!=atoms.end())
                { GMX_THROW(InvalidInputError("mireann-zero-atoms must contain MM atoms only")); }
            }
        }
        std::istringstream words(options_.elements);
        std::string label; size_t count=0;
        while (words>>label) { ++count; }
        if (options_.model.empty() || count!=atoms.size() || options_.threads<1
            || !validDevice(options_.device))
        { GMX_THROW(InvalidInputError("MIREANN requires a seven-input model, equal-length atoms/elements, threads>=1, device=cpu or cuda:N")); }
        if (options_.centerAtom<1 || std::find(atoms.begin(),atoms.end(),options_.centerAtom-1)==atoms.end())
        { GMX_THROW(InvalidInputError("mireann-center-atom must be a global 1-based atom in mireann-atoms")); }
        if (!std::isfinite(options_.mmCutoffAngstrom) || options_.mmCutoffAngstrom<=0
            || !std::isfinite(options_.switchWidthAngstrom) || options_.switchWidthAngstrom<0
            || options_.switchWidthAngstrom>=options_.mmCutoffAngstrom)
        { GMX_THROW(InvalidInputError("MIREANN requires mm-cutoff>0 and 0<=switch-width<mm-cutoff, in Angstrom")); }
        if (!std::isfinite(options_.energyScale) || options_.energyScale<=0)
        { GMX_THROW(InvalidInputError("Specify mireann-energy-scale in kJ/mol per model energy unit; kcal/mol uses 4.184")); }
        if (!std::isfinite(options_.mmListCutoffAngstrom) || options_.mmListCutoffAngstrom<0
            || (options_.mmListCutoffAngstrom>0 && options_.mmListCutoffAngstrom<=options_.mmCutoffAngstrom)
            || options_.mmListNst<0)
        { GMX_THROW(InvalidInputError("mireann-mm-list-cutoff must be 0 (disabled) or > mm-cutoff in Angstrom; mm-list-nst must be >=0 (0 follows native MD neighbor search)")); }
        addMdpOutputValue(builder,mireName,"charge-response",options_.chargeResponse);
        addMdpOutputValue(builder,mireName,"electrostatics-device",options_.electrostaticsDevice);
        addMdpOutputValue(builder,mireName,"threads",options_.threads);
        addMdpOutputValue(builder,mireName,"center-atom",options_.centerAtom);
        addMdpOutputValue(builder,mireName,"mm-cutoff",options_.mmCutoffAngstrom);
        addMdpOutputValue(builder,mireName,"mm-list-cutoff",options_.mmListCutoffAngstrom);
        addMdpOutputValue(builder,mireName,"mm-list-nst",options_.mmListNst);
        addMdpOutputValue(builder,mireName,"mm-selection",options_.mmSelection);
        addMdpOutputValue(builder,mireName,"switch-width",options_.switchWidthAngstrom);
        addMdpOutputValue(builder,mireName,"energy-scale",options_.energyScale);
        for (auto item : {std::pair{"model",&options_.model}, {"atoms",&options_.atoms},
               {"elements",&options_.elements},{"device",&options_.device},
               {"boundary",&options_.boundary},{"zero-atoms",&options_.zeroAtoms},
               {"charge-log",&options_.chargeLog},{"profile",&options_.profile},
               {"energy-log",&options_.energyLog},{"environment-log",&options_.environmentLog}})
        { addMdpOutputValue(builder,mireName,item.first,*item.second); }
    }
private:
    QeqOptions options_;
    std::vector<int> residueIds_; // GMX_MIREANN: preprocessing only, persisted in TPR.
};
}
QeqOptions readMireannOptions(const t_inputrec& inputrec)
{
    QeqOptions options; options.modelType="mireann";
    if (!inputrec.params || !inputrec.params->keyExists("applied-forces")) { return options; }
    const auto& applied=(*inputrec.params)["applied-forces"].asObject();
    if (!applied.keyExists(mireName)) { return options; }
    const auto& tree=applied[mireName].asObject();
    if (tree.keyExists("active")) { options.active=tree["active"].cast<bool>(); }
    if (!options.active) { return options; }
    options.topologyParameters=inputrec.internalParameters.get();
    if (tree.keyExists("charge-response")) { options.chargeResponse=tree["charge-response"].cast<bool>(); }
    if (tree.keyExists("electrostatics-device")) { options.electrostaticsDevice=tree["electrostatics-device"].cast<std::string>(); }
    if (tree.keyExists("mm-selection")) { options.mmSelection=tree["mm-selection"].cast<std::string>(); }
    if (tree.keyExists("threads")) { options.threads=tree["threads"].cast<int>(); }
    if (tree.keyExists("center-atom")) { options.centerAtom=tree["center-atom"].cast<int>(); }
    if (tree.keyExists("mm-cutoff")) { options.mmCutoffAngstrom=tree["mm-cutoff"].cast<double>(); }
    // GMX_MIREANN_BUFFERED_LIST: absent keys leave the old TPR behavior intact.
    if (tree.keyExists("mm-list-cutoff")) { options.mmListCutoffAngstrom=tree["mm-list-cutoff"].cast<double>(); }
    if (tree.keyExists("mm-list-nst")) { options.mmListNst=tree["mm-list-nst"].cast<int>(); }
    if (tree.keyExists("switch-width")) { options.switchWidthAngstrom=tree["switch-width"].cast<double>(); }
    if (tree.keyExists("energy-scale")) { options.energyScale=tree["energy-scale"].cast<double>(); }
    for (auto item : {std::pair{"model",&options.model}, {"atoms",&options.atoms},
           {"elements",&options.elements},{"device",&options.device},
           {"boundary",&options.boundary},{"zero-atoms",&options.zeroAtoms},
           {"charge-log",&options.chargeLog},{"profile",&options.profile},
           {"energy-log",&options.energyLog},{"environment-log",&options.environmentLog}})
    { if (tree.keyExists(item.first)) { *item.second=tree[item.first].cast<std::string>(); } }
    return options;
}
std::unique_ptr<IMDModule> makeMireannModule() { return std::make_unique<MireannModule>(); }
} // namespace gmx::dynamiccharge
