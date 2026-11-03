/* GMX_QEQ_REANN: local extension, LGPL-2.1-or-later, 2026-10-21.
 * MDP/TPR configuration only; no LibTorch headers cross into GROMACS.
 */
#ifndef GMX_MDLIB_DYNAMICCHARGEOPTIONS_H
#define GMX_MDLIB_DYNAMICCHARGEOPTIONS_H
#include <memory>
#include <string>
#include <vector>
struct t_inputrec;
namespace gmx
{
class IMDModule;
class KeyValueTreeObject; // GMX_MIREANN: immutable topology metadata stored in TPR.
namespace dynamiccharge
{
//! Values are serialized by GROMACS's existing key-value-tree TPR mechanism.
struct QeqOptions
{
    bool active = false;
    std::string model;
    std::string atoms;
    std::string elements;
    std::string device = "cpu";
    std::string pbc = "0 0 0";
    std::string chargeLog;
    std::string profile;
    int threads = 1;
    // GMX_MIREANN: independent seven-input embedding module, sharing charge commit.
    std::string modelType = "qeq";
    int centerAtom = 0; // required global 1-based QM atom, not a COM/group number
    double mmCutoffAngstrom = 25.0;
    // GMX_MIREANN_BUFFERED_LIST: optional Verlet-style environment candidates.
    // Zero list radius preserves old per-call searches/old TPR compatibility.
    // nst=0 follows native MD neighbor rebuilds; nst>0 is a maximum step age.
    double mmListCutoffAngstrom = 0.0;
    int mmListNst = 0;
    // GMX_MIREANN RESIDUE: default matches Amber qmcutmode=1. Atom selection
    // remains available explicitly for reproducing earlier interface tests.
    std::string mmSelection = "residue";
    // GMX_MIREANN_PSEUDOBOND: real topology sites, no artificial link atoms.
    std::string boundary = "whole-molecule";
    // GMX_MIREANN_CROSS14: true gives the gradient of the full stated energy;
    // false reproduces Amber's omission of the residual periodic response.
    bool chargeResponse = true;
    // GMX_MIREANN_GPU_AUDIT: auto follows inference; an explicit cuda:N also
    // supports a CPU model with CUDA classical corrections. No Python required.
    std::string electrostaticsDevice="auto";
    // GMX_RELEASE_REVIEW_20261102: clarify the already implemented compact list.
    // ORIGINAL | std::string zeroAtoms; // 1-based MM sites: zero model background q only, retain FF q.
    std::string zeroAtoms; // 1-based MM sites: EXCLUDE from model background; retain physical FF q.

    const KeyValueTreeObject* topologyParameters = nullptr; // borrowed from inputrec
    double switchWidthAngstrom = 0.0; // zero preserves a hard spherical selection
    double energyScale = 0.0; // explicit kJ/mol per model energy unit
    std::string energyLog;
    std::string environmentLog;
};
QeqOptions readQeqOptions(const t_inputrec& inputrec);
bool dynamicChargesEnabled(const t_inputrec& inputrec);
std::vector<int> parseQeqAtoms(const std::string& atoms);
std::unique_ptr<IMDModule> makeQeqModule();
// GMX_MIREANN: registered through the same existing IMDModule/TPR mechanism.
QeqOptions readMireannOptions(const t_inputrec& inputrec);
std::unique_ptr<IMDModule> makeMireannModule();
} // namespace dynamiccharge
} // namespace gmx
#endif
