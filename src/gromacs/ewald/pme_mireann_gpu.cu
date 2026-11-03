/* GMX_MIREANN NEW FILE: gather charge derivatives on the solved CUDA PME mesh.
 * Local extension, LGPL-2.1-or-later. Uses the native PME stream so the solve,
 * gather, result transfer and subsequent mesh clearing have a defined order.
 * Only N_QM scalars return to the host; the full mesh stays on the device.
 */
#include "gmxpre.h"
#include "config.h"
#include <cuda_runtime.h>
#include <vector>
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/device_stream.h"
#include "pme_gpu_internal.h"
#include "pme_gpu_types.h"
#include "pme_gpu_types_host.h"
#include "pme_gpu_types_host_impl.h"

namespace
{
__global__ void gatherPotential(const float* grid, int support, const int* offsets,
                               const float* weights, float* potential)
{
    __shared__ float sum[64];
    const int atom = blockIdx.x, lane = threadIdx.x;
    float value = 0;
    for (int k = lane; k < support; k += 64)
    {
        const int i = atom*support+k;
        value += weights[i]*grid[offsets[i]];
    }
    sum[lane] = value;
    __syncthreads();
    for (int stride = 32; stride > 0; stride /= 2)
    {
        if (lane < stride) { sum[lane] += sum[lane+stride]; }
        __syncthreads();
    }
    if (lane == 0) { potential[atom] = sum[0]; }
}

// GMX_MIREANN_GPU_AUDIT: original per-step cudaMallocAsync/cudaFreeAsync
// allocation is replaced by one workspace owned by the PME instance.
// ORIGINAL:
// struct Workspace { void* pointer=nullptr; cudaStream_t stream;
// Workspace(size_t bytes,cudaStream_t streamIn):stream(streamIn)
// { checkDeviceError(cudaMallocAsync(&pointer,bytes,stream),"MIREANN PME allocation"); }
// ~Workspace(){if(pointer){cudaFreeAsync(pointer,stream);}} };
struct Workspace
{
    void* pointer = nullptr;
    size_t capacity=0;
    int device=0;
    std::vector<float> hostWeights,hostPotential;
    Workspace(){gmx::checkDeviceError(cudaGetDevice(&device),"MIREANN PME device");}
    void reserve(size_t bytes)
    {
        if(bytes<=capacity){return;}
        void* next=nullptr;
        gmx::checkDeviceError(cudaMalloc(&next,bytes),"MIREANN PME workspace allocation");
        if(pointer){cudaFree(pointer);}pointer=next;capacity=bytes;
    }
    ~Workspace()
    {
        // The pointer always belongs to its original PME GPU, also when model
        // inference used another CUDA device. Destructors must not throw.
        int previous=0;cudaGetDevice(&previous);cudaSetDevice(device);
        if(pointer){cudaFree(pointer);}cudaSetDevice(previous);
    }
};
}

void pme_gpu_mireann_gather(const PmeGpu* pmeGpu, int count, int support,
                             const int* offsets, const double* weights, double* potential)
{
    const auto stream = pmeGpu->archSpecific->pmeStream_.stream();
    const size_t entries = static_cast<size_t>(count)*support;
    const size_t indexBytes = ((entries*sizeof(int)+7)/8)*8;
    // ORIGINAL | std::vector<float> hostWeights(weights,weights+entries),hostPotential(count);
    // ORIGINAL | Workspace storage(indexBytes+(entries+count)*sizeof(float),stream);
    auto& owner=pmeGpu->archSpecific->mireannGatherWorkspace;
    if(!owner){owner=std::shared_ptr<void>(new Workspace,[](void* p){delete static_cast<Workspace*>(p);});}
    auto& storage=*static_cast<Workspace*>(owner.get());
    storage.reserve(indexBytes+(entries+count)*sizeof(float));
    storage.hostWeights.assign(weights,weights+entries);storage.hostPotential.resize(count);
    auto& hostWeights=storage.hostWeights;auto& hostPotential=storage.hostPotential;
    auto* deviceOffsets = static_cast<int*>(storage.pointer);
    auto* deviceWeights = reinterpret_cast<float*>(static_cast<char*>(storage.pointer)+indexBytes);
    auto* devicePotential = deviceWeights+entries;
    gmx::checkDeviceError(cudaMemcpyAsync(deviceOffsets, offsets, entries*sizeof(int),
                                          cudaMemcpyHostToDevice, stream), "MIREANN PME indices");
    gmx::checkDeviceError(cudaMemcpyAsync(deviceWeights, hostWeights.data(), entries*sizeof(float),
                                          cudaMemcpyHostToDevice, stream), "MIREANN PME spline weights");
    const auto* parameters = reinterpret_cast<const PmeGpuKernelParams*>(pmeGpu->kernelParams.get());
    gatherPotential<<<count,64,0,stream>>>(parameters->grid.d_realGrid[0], support,
                                          deviceOffsets, deviceWeights, devicePotential);
    gmx::checkDeviceError(cudaGetLastError(), "MIREANN PME gather kernel");
    gmx::checkDeviceError(cudaMemcpyAsync(hostPotential.data(), devicePotential, count*sizeof(float),
                                          cudaMemcpyDeviceToHost, stream), "MIREANN PME potentials");
    gmx::checkDeviceError(cudaStreamSynchronize(stream), "MIREANN PME gather completion");
    for (int i=0; i<count; ++i) { potential[i] = hostPotential[i]; }
}
