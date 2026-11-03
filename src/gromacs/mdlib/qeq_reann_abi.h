// GMX_QEQ_REANN: new local extension; LGPL-2.1-or-later.
// Local QEq-REANN bridge. C ABI keeps LibTorch's C++ ABI out of GROMACS.
#ifndef GMX_MDLIB_QEQ_REANN_ABI_H
#define GMX_MDLIB_QEQ_REANN_ABI_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct GmxQeqOptions
{
    int abiVersion;
    int numAtoms;
    const char* modelPath;
    const char* device; // cpu or cuda:0; no silent device fallback
    const char* elements; // space-separated symbols in selected-atom order
    const double* massesU;
    int pbc[3];
    int torchThreads;
} GmxQeqOptions;
int gmx_qeq_abi_version(void);
void* gmx_qeq_create(const GmxQeqOptions* options, double* totalChargeE,
                     char* error, size_t errorSize);
int gmx_qeq_predict(void* handle, const double* coordinatesAngstrom,
                    const double* boxAngstrom, double* chargesE,
                    double* lagrangeMultiplier, double* elapsedMs,
                    char* error, size_t errorSize);
void gmx_qeq_destroy(void* handle);
#ifdef __cplusplus
}
#endif
#endif
