/* GMX_MIREANN NEW FILE: electrostatic subtraction for model input images.
 * Local extension, LGPL-2.1-or-later.
 *
 * Selection has already been made around the user-specified QM center. Use
 * exactly those imaged coordinates; do not image each QM-MM pair again and
 * do not reject pairs beyond rcoulomb. The Amber-compatible kernel is 1/r
 * inside rcoulomb and erf(alpha*r)/r outside. The latter force MUST include
 * the Gaussian derivative. This is not a complete periodic Ewald sum.
 */
#ifndef GMX_MDLIB_MIRECOULOMB_H
#define GMX_MDLIB_MIRECOULOMB_H

#include <array>
#include <cmath>
#include <stdexcept>

namespace gmx::dynamiccharge
{
struct DirectCoulombPair
{
    double energyKjMol;
    double potentialAtQmKjPerE; // dE/dq_QM at fixed MM charge and coordinates
    std::array<double, 3> forceOnQmKjNm; // opposite force acts on MM
};

inline DirectCoulombPair directCoulombPair(const std::array<double, 3>& qmMinusMmNm,
                                          double qmChargeE, double mmChargeE,
                                          double coulombConstant)
{
    double squaredDistance = 0;
    for (double value : qmMinusMmNm) { squaredDistance += value * value; }
    if (!std::isfinite(squaredDistance) || squaredDistance <= 1e-20
        || !std::isfinite(qmChargeE) || !std::isfinite(mmChargeE)
        || !std::isfinite(coulombConstant))
    {
        throw std::runtime_error("Invalid or overlapping MIREANN direct-Coulomb pair");
    }
    const double inverseDistance = 1.0 / std::sqrt(squaredDistance);
    DirectCoulombPair result{};
    result.potentialAtQmKjPerE = coulombConstant * mmChargeE * inverseDistance;
    result.energyKjMol = qmChargeE * result.potentialAtQmKjPerE;
    const double forceScale = result.energyKjMol / squaredDistance;
    for (int d = 0; d < 3; ++d)
    {
        result.forceOnQmKjNm[d] = forceScale * qmMinusMmNm[d];
    }
    return result;
}

// GMX_MIREANN: return the positive reference term; the caller subtracts its
// energy, force and charge derivative together. alpha is in nm^-1, rc in nm.
// At rc the real-space Ewald term is truncated just as in the supplied Amber
// implementation; do not take a finite difference across that branch boundary.
inline DirectCoulombPair embeddingCoulombPair(const std::array<double, 3>& dx,
                                             double qi, double qj, double constant,
                                             double rc, double alpha,
                                             double potentialShiftPerNm=0)
{
    auto result = directCoulombPair(dx, qi, qj, constant);
    if (!std::isfinite(rc) || rc <= 0 || !std::isfinite(alpha) || alpha <= 0
        || !std::isfinite(potentialShiftPerNm))
    {
        throw std::runtime_error("Invalid MIREANN Ewald coefficient or Coulomb cutoff");
    }
    double r2 = 0;
    for (double value : dx) { r2 += value * value; }
    if (r2 > rc * rc)
    {
        const double r = std::sqrt(r2);
        const double ar = alpha * r;
        const double screening = std::erf(ar);
        // -d[erf(alpha*r)/r]/dr / r, including the Gaussian term missing
        // from the reference Amber code. Never use screening/r^3 alone.
        const double forceFactor = screening - 2.0 * ar / std::sqrt(std::acos(-1.0))
                                                        * std::exp(-ar * ar);
        result.energyKjMol *= screening;
        result.potentialAtQmKjPerE *= screening;
        for (double& force : result.forceOnQmKjNm) { force *= forceFactor; }
    }
    else
    {
        // GMX_MIREANN_PHYSICS_AUDIT: the real-space PME potential shift must
        // be subtracted in the reference too. Otherwise a selected pair leaves
        // a charge-dependent energy jump at rc, although its fixed-q force has
        // no constant-shift term. None (shift=0) is exactly the old convention.
        result.potentialAtQmKjPerE-=constant*qj*potentialShiftPerNm;
        result.energyKjMol=qi*result.potentialAtQmKjPerE;
    }
    return result;
}
} // namespace gmx::dynamiccharge
#endif
