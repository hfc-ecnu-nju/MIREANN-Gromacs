/* GMX_MIREANN_GPU_AUDIT (2026-10-25), LGPL-2.1-or-later.
 * Host-only C ABI. Distances nm, q elementary charge, energy kJ/mol.
 * Orthorhombic boxes only: the caller retains the original CPU triclinic path.
 */
#ifndef GMX_MIREANN_CLASSICAL_ABI_H
#define GMX_MIREANN_CLASSICAL_ABI_H
#include <stdint.h>
typedef struct GmxMireannClassicalInput
{
    int numAtoms, numQm, numMm, pseudoBond, cudaDevice;
    uint64_t topologyRevision;
    const double* positionsNm;
    const double* chargesE;
    const int* qmGlobal;
    const unsigned char* exclusions; /* [numQm,numAtoms], global physical order */
    const double* qmImageNm;
    const double* mmImageNm;
    const int* mmGlobal;
    const double* mmChargesE;
    const double* weights;
    double boxDiagonalNm[3];
    double coulombConstant, cutoffNm, alphaPerNm, potentialShiftPerNm;
} GmxMireannClassicalInput;
typedef struct GmxMireannClassicalOutput
{
    double* shortRangePotential; /* excludes self/background and listed 1-4 */
    double* referencePotential;
    double* referenceQmForce;
    double* referenceMmForce;
    double* referenceUnweightedEnergyPerMm;
} GmxMireannClassicalOutput;
#ifdef __cplusplus
extern "C" {
#endif
void* gmx_mireann_cuda_classical_create(int device);
void gmx_mireann_cuda_classical_destroy(void* handle);
void gmx_mireann_cuda_classical_compute(void* handle,
    const GmxMireannClassicalInput* input, GmxMireannClassicalOutput* output);
#ifdef __cplusplus
}
#endif
#endif
