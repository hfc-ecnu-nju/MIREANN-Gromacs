/* GMX_QEQ_REANN BEGIN: local extension, LGPL-2.1-or-later, 2026-10-21.
 * Register MDP options through IMdpOptionProvider, as other GROMACS modules do.
 * No extra TPR fields/version changes, environment mutation, or force provider:
 * charges must be committed BEFORE all short-range/listed/PME force work.
 */
#include "gmxpre.h"
#include "dynamicchargeoptions.h"
#include <cstdlib>
#include <limits>
#include <set>
#include <sstream>
#include <utility>
#include "gromacs/mdtypes/imdmodule.h"
#include "gromacs/mdtypes/imdpoptionprovider.h"
#include "gromacs/mdtypes/imdpoptionprovider_helpers.h"
#include "gromacs/mdtypes/inputrec.h"
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
constexpr const char* moduleName = "qeq-reann";
class QeqModule final : public IMDModule, public IMdpOptionProvider
{
public:
    IMdpOptionProvider* mdpOptionProvider() override { return this; }
    IMDOutputProvider* outputProvider() override { return nullptr; }
    void initForceProviders(ForceProviders*) override {}
    void subscribeToPreProcessingNotifications(MDModulesNotifiers*) override {}
    void subscribeToSimulationSetupNotifications(MDModulesNotifiers*) override {}
    void subscribeToSimulationRunNotifications(MDModulesNotifiers*) override {}
    void initMdpTransform(IKeyValueTreeTransformRules* rules) override
    {
        addMdpTransformFromString<bool>(rules, &fromStdString<bool>, moduleName, "active");
        addMdpTransformFromString<int>(rules, &fromStdString<int>, moduleName, "threads");
        for (const char* tag : {"model", "atoms", "elements", "device", "pbc", "charge-log", "profile"})
        {
            addMdpTransformFromString<std::string>(
                    rules, [](std::string value) { return value; }, moduleName, tag);
        }
    }
    void initMdpOptions(IOptionsContainerWithSections* options) override
    {
        auto section = options->addSection(OptionSection(moduleName));
        section.addOption(BooleanOption("active").store(&options_.active));
        section.addOption(IntegerOption("threads").store(&options_.threads));
        section.addOption(StringOption("model").store(&options_.model));
        section.addOption(StringOption("atoms").store(&options_.atoms));
        section.addOption(StringOption("elements").store(&options_.elements));
        section.addOption(StringOption("device").store(&options_.device));
        section.addOption(StringOption("pbc").store(&options_.pbc));
        section.addOption(StringOption("charge-log").store(&options_.chargeLog));
        section.addOption(StringOption("profile").store(&options_.profile));
    }
    void buildMdpOutput(KeyValueTreeObjectBuilder* builder) const override
    {
        addMdpOutputComment(builder, moduleName, "module", "; GMX_QEQ_REANN: per-force atomic charges, nm -> Angstrom, q(e)");
        addMdpOutputValue(builder, moduleName, "active", options_.active);
        if (!options_.active) { return; }
#ifndef GMX_QEQ_REANN_BUILT
        GMX_THROW(InvalidInputError("qeq-reann-active=yes requires a build with -DGMX_QEQ_REANN=ON"));
#endif
        const auto atoms = parseQeqAtoms(options_.atoms);
        std::istringstream elements(options_.elements);
        std::string symbol;
        size_t count = 0;
        while (elements >> symbol) { ++count; }
        if (options_.model.empty() || count != atoms.size() || options_.threads < 1
            || (options_.device != "cpu" && options_.device != "cuda:0"))
        {
            GMX_THROW(InvalidInputError("QEq requires model, equal-length atoms/elements lists, threads>=1, device=cpu or cuda:0"));
        }
        std::istringstream pbc(options_.pbc);
        int flag;
        for (int d = 0; d < 3; ++d)
        {
            if (!(pbc >> flag) || (flag != 0 && flag != 1))
            { GMX_THROW(InvalidInputError("qeq-reann-pbc requires three 0/1 flags")); }
        }
        if (pbc >> symbol) { GMX_THROW(InvalidInputError("Too many qeq-reann-pbc flags")); }
        addMdpOutputValue(builder, moduleName, "threads", options_.threads);
        addMdpOutputValue(builder, moduleName, "model", options_.model);
        addMdpOutputValue(builder, moduleName, "atoms", options_.atoms);
        addMdpOutputValue(builder, moduleName, "elements", options_.elements);
        addMdpOutputValue(builder, moduleName, "device", options_.device);
        addMdpOutputValue(builder, moduleName, "pbc", options_.pbc);
        addMdpOutputValue(builder, moduleName, "charge-log", options_.chargeLog);
        addMdpOutputValue(builder, moduleName, "profile", options_.profile);
    }
private:
    QeqOptions options_;
};
} // namespace

std::vector<int> parseQeqAtoms(const std::string& text)
{
    std::istringstream stream(text);
    std::string token;
    std::set<int> seen;
    std::vector<int> atoms;
    while (stream >> token)
    {
        size_t used = 0;
        long long id = 0;
        try { id = std::stoll(token, &used); }
        catch (const std::exception&) { GMX_THROW(InvalidInputError("Invalid qeq-reann-atoms entry: " + token)); }
        if (used != token.size() || id < 1 || id > std::numeric_limits<int>::max()
            || !seen.insert(static_cast<int>(id-1)).second)
        { GMX_THROW(InvalidInputError("qeq-reann-atoms requires unique positive 1-based atom numbers")); }
        atoms.push_back(static_cast<int>(id-1));
    }
    if (atoms.empty()) { GMX_THROW(InvalidInputError("qeq-reann-atoms is empty")); }
    return atoms;
}

QeqOptions readQeqOptions(const t_inputrec& inputrec)
{
    // GMX_MIREANN BEGIN: route one charge-producing model through existing commit.
    const auto mireann=readMireannOptions(inputrec);
    if (mireann.active)
    {
        if (inputrec.params && inputrec.params->keyExists("applied-forces"))
        {
            const auto& applied=(*inputrec.params)["applied-forces"].asObject();
            if (applied.keyExists(moduleName))
            {
                const auto& qeq=applied[moduleName].asObject();
                if (qeq.keyExists("active") && qeq["active"].cast<bool>())
                { GMX_THROW(InvalidInputError("Do not enable qeq-reann and mireann together: MIREANN itself returns the new QM charges")); }
            }
        }
        return mireann;
    }
    // GMX_MIREANN END
    QeqOptions options;
    if (!inputrec.params || !inputrec.params->keyExists("applied-forces")) { return options; }
    const auto& applied = (*inputrec.params)["applied-forces"].asObject();
    if (!applied.keyExists(moduleName)) { return options; }
    const auto& tree = applied[moduleName].asObject();
    if (tree.keyExists("active")) { options.active = tree["active"].cast<bool>(); }
    if (!options.active) { return options; }
    if (tree.keyExists("threads")) { options.threads = tree["threads"].cast<int>(); }
    for (auto item : {std::pair{"model", &options.model}, {"atoms", &options.atoms},
                      {"elements", &options.elements}, {"device", &options.device},
                      {"pbc", &options.pbc}, {"charge-log", &options.chargeLog}, {"profile", &options.profile}})
    { if (tree.keyExists(item.first)) { *item.second = tree[item.first].cast<std::string>(); } }
    return options;
}
bool dynamicChargesEnabled(const t_inputrec& inputrec)
{
    return readQeqOptions(inputrec).active || std::getenv("GMX_DYNAMIC_CHARGE_REGION") != nullptr;
}
std::unique_ptr<IMDModule> makeQeqModule() { return std::make_unique<QeqModule>(); }
} // namespace gmx::dynamiccharge
/* GMX_QEQ_REANN END */
