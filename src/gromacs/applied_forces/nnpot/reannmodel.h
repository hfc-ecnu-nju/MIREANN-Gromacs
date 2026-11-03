// GMX_REANN_PES NEW FILE BEGIN: native NNPot adapter, LGPL-2.1-or-later.
#ifndef GMX_APPLIED_FORCES_REANNMODEL_H
#define GMX_APPLIED_FORCES_REANNMODEL_H

#include <memory>
#include <string>
#include "gromacs/applied_forces/nnpot/nnpotmodel.h"

namespace gmx
{
struct NNPotParameters;
class MDLogger;

// Molecular REANN PES: forward(pbc, cart[A], cell[A], species, mass[u]) -> (E,F).
// Mechanical embedding and whole molecules only; no external-MM model inputs.
class ReannModel final : public INNPotModel
{
public:
    ReannModel(const NNPotParameters& params, const MDLogger& logger,
               const MpiComm& mpiComm, bool validationOnly = false);
    ~ReannModel() override;

    void evaluateModel(gmx_enerdata_t* enerd, ArrayRef<RVec> forces,
                       ArrayRef<const int> indexLookup, ArrayRef<const int> mmIndices,
                       ArrayRef<const std::string> inputs, ArrayRef<RVec> positions,
                       ArrayRef<int> atomNumbers, ArrayRef<int> atomPairs,
                       ArrayRef<RVec> pairShifts, ArrayRef<RVec> positionsMM,
                       ArrayRef<real> chargesMM, real nnpCharge,
                       ArrayRef<const LinkFrontierAtom> linkFrontier,
                       matrix* box, PbcType* pbcType) override;

    bool outputsForces() const override { return true; }
    const matrix& virial() const { return virial_; }
    void setComputeVirial(bool requested) { computeVirial_ = requested; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    matrix virial_{};
    bool computeVirial_ = false;
};
} // namespace gmx
#endif
// GMX_REANN_PES NEW FILE END
