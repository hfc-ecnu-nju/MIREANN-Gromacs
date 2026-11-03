/* GMX_MIREANN NEW FILE: C boundary for seven-input electrostatic embedding.
 * Local extension, LGPL-2.1-or-later. No Torch C++ types enter GROMACS headers.
 */
#ifndef GMX_MDLIB_MIREANN_ABI_H
#define GMX_MDLIB_MIREANN_ABI_H
#include "qeq_reann_abi.h"
#include "mireann_classical_abi.h" // GMX_MIREANN_GPU_AUDIT
#ifdef __cplusplus
extern "C" {
#endif
void* gmx_mireann_create(const GmxQeqOptions* options, double* totalChargeE,
                        char* error, size_t errorSize);
/* GMX_REANN_ASYNC: background-charge derivatives are needed only for a
 * nonzero switching-width chain rule. Coordinate/QM-charge responses stay on.
 * Existing clients retain the original true default. Must be set between steps.
 */
int gmx_mireann_set_background_response(void* handle, int enabled,
                                       char* error, size_t errorSize);
int gmx_mireann_predict(void* handle, int numMm, const double* qmAngstrom,
                        const double* mmAngstrom, const double* mmChargesE,
                        const double* boxAngstrom, double* qmChargesE,
                        double* energyModelUnits, double* elapsedMs,
                        char* error, size_t errorSize);
/* phi is d(U_GROMACS-U_near)/dq_QM, in kJ/(mol e). The response is a VJP,
 * not an additional energy: global electrostatics already contains that energy.
 * Forces returned here include the model energy gradient AND this charge response.
 * bgDerivative is d[scale*E_model + phi.q]/d(q_MM_input), needed for a smooth shell.
 */
int gmx_mireann_forces(void* handle, const double* potentialKjPerE, double energyScale,
                       double* qmForceKjNm, double* mmForceKjNm, double* bgDerivative,
                       char* error, size_t errorSize);
void gmx_mireann_destroy(void* handle);
int gmx_mireann_classical(void* handle,const GmxMireannClassicalInput* input,
    GmxMireannClassicalOutput* output,int* usedGpu,char* error,size_t errorSize);
#ifdef __cplusplus
}
#endif
#endif
