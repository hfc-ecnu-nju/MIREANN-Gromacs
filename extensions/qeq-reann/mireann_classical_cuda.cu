/* GMX_MIREANN_GPU_AUDIT (2026-10-25), LGPL-2.1-or-later.
 * Same reference Hamiltonian as mirecoulomb.h, including Gaussian derivative
 * and the selected GROMACS potential shift. No changes to model/weights/PME.
 * All pair work and reductions run on CUDA. Persistent, bounded workspaces
 * belong to one model instance; no per-step cudaMalloc/Free, no global cache.
 */
#include "mireann_classical_abi.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr int blockSize=128;
constexpr double invSqrtPi=0.56418958354775628695;
void check(cudaError_t status,const char* operation)
{
    if (status!=cudaSuccess)
    { throw std::runtime_error(std::string("MIREANN CUDA ")+operation+": "+cudaGetErrorString(status)); }
}
struct DeviceGuard
{
    int previous;
    explicit DeviceGuard(int device)
    { check(cudaGetDevice(&previous),"get device"); if (previous!=device) { check(cudaSetDevice(device),"set device"); } }
    ~DeviceGuard() { cudaSetDevice(previous); }
};
template<class T> struct Buffer
{
    T* data=nullptr;
    size_t capacity=0;
    ~Buffer() { if (data) { cudaFree(data); } }
    void reserve(size_t count)
    {
        if (count<=capacity) { return; }
        // Allocate first: a failed growth leaves the old allocation owned.
        size_t next=std::max(count,capacity+capacity/4+128);
        if (next>std::numeric_limits<size_t>::max()/sizeof(T)) { throw std::runtime_error("MIREANN CUDA size overflow"); }
        T* replacement=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&replacement),next*sizeof(T)),"workspace allocation");
        if (data) { check(cudaFree(data),"workspace release"); }
        data=replacement;capacity=next;
    }
};
struct Engine
{
    int device;
    cudaStream_t stream=nullptr;
    Buffer<double> x,q,qm,mm,mmq,w,pair,partial,result;
    Buffer<int> qmId,mmId;
    Buffer<unsigned char> excluded;
    std::vector<double> host;
    uint64_t revision=std::numeric_limits<uint64_t>::max();
    int cachedAtoms=-1,cachedQm=-1;
    explicit Engine(int id):device(id)
    { DeviceGuard guard(device);check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream creation"); }
    ~Engine()
    {
        // No throwing CUDA calls during destruction/error unwinding.
        int previous=0;cudaGetDevice(&previous);cudaSetDevice(device);
        if (stream) { cudaStreamSynchronize(stream);cudaStreamDestroy(stream); }
        // Members must be freed while this instance's CUDA device is current.
        // Explicitly release here; Buffer destructors then see null pointers.
        auto release=[](auto& b){ if(b.data){cudaFree(b.data);b.data=nullptr;} };
        release(x);release(q);release(qm);release(mm);release(mmq);release(w);
        release(pair);release(partial);release(result);release(qmId);release(mmId);release(excluded);
        cudaSetDevice(previous);
    }
    template<class T> void upload(Buffer<T>& dst,const T* src,size_t count)
    {
        dst.reserve(count);
        if (count) { check(cudaMemcpyAsync(dst.data,src,count*sizeof(T),cudaMemcpyHostToDevice,stream),"input copy"); }
    }
};
__device__ double sumBlock(double value,double* shared)
{
    const int lane=threadIdx.x;shared[lane]=value;__syncthreads();
    for (int stride=blockSize/2;stride;stride/=2)
    { if(lane<stride){shared[lane]+=shared[lane+stride];}__syncthreads(); }
    const double result=shared[0];__syncthreads();return result;
}
__global__ void shortRange(int n,int chunks,const double* x,const double* q,const int* qmId,
    const unsigned char* excluded,double lx,double ly,double lz,double c,double rc,double alpha,
    double shift,double* partial)
{
    __shared__ double shared[blockSize];
    const int i=blockIdx.y,j=blockIdx.x*blockSize+threadIdx.x,atom=qmId[i];
    double value=0;
    if(j<n && j!=atom)
    {
        const double box[3]={lx,ly,lz};double r2=0;
        for(int d=0;d<3;++d)
        { double dx=x[3*atom+d]-x[3*j+d];dx-=nearbyint(dx/box[d])*box[d];r2+=dx*dx; }
        if(r2<=1e-20) { value=nan(""); }
        else
        {
            const double r=sqrt(r2),ar=alpha*r;
            if(excluded[static_cast<size_t>(i)*n+j]) { value=-c*q[j]*erf(ar)/r; }
            else if(r<rc) { value=c*q[j]*(erfc(ar)/r-shift); }
        }
    }
    const double total=sumBlock(value,shared);
    if(threadIdx.x==0){partial[static_cast<size_t>(i)*chunks+blockIdx.x]=total;}
}
__global__ void referencePairs(int n,int nm,const double* qm,const double* mm,const int* qmId,
    const int* mmId,const double* charges,const double* mmq,const double* weights,
    const unsigned char* excluded,int pseudo,double c,double rc,double alpha,double shift,double* pair)
{
    const int i=blockIdx.y,j=blockIdx.x*blockSize+threadIdx.x;
    if(j>=nm){return;}
    const size_t offset=(static_cast<size_t>(i)*nm+j)*5;
    if(pseudo && excluded[static_cast<size_t>(i)*n+mmId[j]])
    { for(int d=0;d<5;++d){pair[offset+d]=0;}return; }
    double dx[3],r2=0;
    for(int d=0;d<3;++d){dx[d]=qm[3*i+d]-mm[3*j+d];r2+=dx[d]*dx[d];}
    if(r2<=1e-20){for(int d=0;d<5;++d){pair[offset+d]=nan("");}return;}
    const double r=sqrt(r2),ar=alpha*r,qi=charges[qmId[i]],base=c*mmq[j]/r;
    double potential=base,forceFactor=1;
    if(r2>rc*rc)
    { potential*=erf(ar);forceFactor=erf(ar)-2*ar*invSqrtPi*exp(-ar*ar); }
    else { potential-=c*mmq[j]*shift; }
    const double scale=weights[j]*qi*base/r2*forceFactor;
    for(int d=0;d<3;++d){pair[offset+d]=scale*dx[d];}
    pair[offset+3]=weights[j]*potential;
    pair[offset+4]=qi*potential;
}
__global__ void reduceQm(int nm,int chunks,const double* pair,const double* partial,double* result)
{
    __shared__ double shared[blockSize];const int i=blockIdx.x,lane=threadIdx.x;
    for(int d=0;d<4;++d)
    {
        double value=0;
        for(int j=lane;j<nm;j+=blockSize){value+=pair[(static_cast<size_t>(i)*nm+j)*5+d];}
        const double total=sumBlock(value,shared);if(lane==0){result[5*i+d]=total;}
    }
    double value=0;for(int k=lane;k<chunks;k+=blockSize){value+=partial[static_cast<size_t>(i)*chunks+k];}
    const double total=sumBlock(value,shared);if(lane==0){result[5*i+4]=total;}
}
__global__ void reduceMm(int nq,int nm,const double* pair,double* result)
{
    const int j=blockIdx.x*blockSize+threadIdx.x;if(j>=nm){return;}
    double value[4]={0,0,0,0};
    for(int i=0;i<nq;++i)
    {
        const size_t offset=(static_cast<size_t>(i)*nm+j)*5;
        for(int d=0;d<3;++d){value[d]-=pair[offset+d];}
        value[3]+=pair[offset+4];
    }
    for(int d=0;d<4;++d){result[5*nq+4*j+d]=value[d];}
}
}
extern "C" void* gmx_mireann_cuda_classical_create(int device) { return new Engine(device); }
extern "C" void gmx_mireann_cuda_classical_destroy(void* handle) { delete static_cast<Engine*>(handle); }
extern "C" void gmx_mireann_cuda_classical_compute(void* handle,
    const GmxMireannClassicalInput* in,GmxMireannClassicalOutput* out)
{
    auto& e=*static_cast<Engine*>(handle);DeviceGuard guard(e.device);
    const int n=in->numAtoms,nq=in->numQm,nm=in->numMm;
    if(n<=0 || nq<=0 || nq>n || nm<0 || nm>n || !(in->cutoffNm>0) || !(in->alphaPerNm>0))
    { throw std::runtime_error("Invalid MIREANN CUDA classical dimensions/parameters"); }
    for(int d=0;d<3;++d)
    {
        if(!std::isfinite(in->boxDiagonalNm[d]) || in->boxDiagonalNm[d]<=2*in->cutoffNm)
        { throw std::runtime_error("MIREANN CUDA classical path requires an orthorhombic box wider than twice rcoulomb"); }
    }
    for(int i=0;i<nq;++i){if(in->qmGlobal[i]<0 || in->qmGlobal[i]>=n){throw std::runtime_error("Invalid CUDA QM mapping");}}
    for(int j=0;j<nm;++j){if(in->mmGlobal[j]<0 || in->mmGlobal[j]>=n){throw std::runtime_error("Invalid CUDA MM mapping");}}
    const int chunks=(n+blockSize-1)/blockSize;
    e.upload(e.x,in->positionsNm,static_cast<size_t>(n)*3);e.upload(e.q,in->chargesE,n);
    e.upload(e.qm,in->qmImageNm,static_cast<size_t>(nq)*3);e.upload(e.mm,in->mmImageNm,static_cast<size_t>(nm)*3);
    e.upload(e.mmId,in->mmGlobal,nm);e.upload(e.mmq,in->mmChargesE,nm);e.upload(e.w,in->weights,nm);
    if(e.revision!=in->topologyRevision || e.cachedAtoms!=n || e.cachedQm!=nq)
    {
        e.upload(e.qmId,in->qmGlobal,nq);e.upload(e.excluded,in->exclusions,static_cast<size_t>(nq)*n);
        e.revision=in->topologyRevision;e.cachedAtoms=n;e.cachedQm=nq;
    }
    e.pair.reserve(static_cast<size_t>(nq)*nm*5);e.partial.reserve(static_cast<size_t>(nq)*chunks);
    e.result.reserve(static_cast<size_t>(nq)*5+static_cast<size_t>(nm)*4);
    shortRange<<<dim3(chunks,nq),blockSize,0,e.stream>>>(n,chunks,e.x.data,e.q.data,e.qmId.data,e.excluded.data,
        in->boxDiagonalNm[0],in->boxDiagonalNm[1],in->boxDiagonalNm[2],in->coulombConstant,
        in->cutoffNm,in->alphaPerNm,in->potentialShiftPerNm,e.partial.data);
    if(nm)
    {
        referencePairs<<<dim3((nm+blockSize-1)/blockSize,nq),blockSize,0,e.stream>>>(n,nm,e.qm.data,e.mm.data,
            e.qmId.data,e.mmId.data,e.q.data,e.mmq.data,e.w.data,e.excluded.data,in->pseudoBond,
            in->coulombConstant,in->cutoffNm,in->alphaPerNm,in->potentialShiftPerNm,e.pair.data);
    }
    reduceQm<<<nq,blockSize,0,e.stream>>>(nm,chunks,e.pair.data,e.partial.data,e.result.data);
    if(nm){reduceMm<<<(nm+blockSize-1)/blockSize,blockSize,0,e.stream>>>(nq,nm,e.pair.data,e.result.data);}
    check(cudaGetLastError(),"classical kernels");
    e.host.resize(static_cast<size_t>(nq)*5+static_cast<size_t>(nm)*4);
    check(cudaMemcpyAsync(e.host.data(),e.result.data,e.host.size()*sizeof(double),cudaMemcpyDeviceToHost,e.stream),"classical results");
    check(cudaStreamSynchronize(e.stream),"classical completion");
    if(!std::all_of(e.host.begin(),e.host.end(),[](double v){return std::isfinite(v);}))
    { throw std::runtime_error("Nonfinite MIREANN CUDA classical result or overlapping atoms"); }
    for(int i=0;i<nq;++i)
    {
        for(int d=0;d<3;++d){out->referenceQmForce[3*i+d]=e.host[5*i+d];}
        out->referencePotential[i]=e.host[5*i+3];out->shortRangePotential[i]=e.host[5*i+4];
    }
    for(int j=0;j<nm;++j)
    {
        for(int d=0;d<3;++d){out->referenceMmForce[3*j+d]=e.host[5*nq+4*j+d];}
        out->referenceUnweightedEnergyPerMm[j]=e.host[5*nq+4*j+3];
    }
}
