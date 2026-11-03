/* GMX_REANN_ASYNC NEW FILE (2026-10-26), LGPL-2.1-or-later.
 * Persistent Torch staging for the REANN/MIREANN bridges. Inspired by native
 * StatePropagatorDataGpu's capacity reuse and stream/event dependency model.
 *
 * This is NOT a replacement for native GROMACS coordinate/force buffers.
 * The present C ABI has host consumers, so each result batch still needs one
 * host fence. Never remove that fence without converting those consumers too.
 */
#ifndef GMX_MDLIB_REANN_ASYNC_BUFFERS_H
#define GMX_MDLIB_REANN_ASYNC_BUFFERS_H

// Native TorchModel undefines DIM around Torch headers for the same reason.
// Unlike that public adapter, restore the caller's macro after this helper.
#pragma push_macro("DIM")
#undef DIM

#include <torch/torch.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// GMX_RELEASE_REVIEW_20261102: distinguish the two independent CUDA builds.
// ORIGINAL | #if defined(GMX_QEQ_TORCH_CUDA) || (defined(GMX_GPU_CUDA) && GMX_GPU_CUDA)
#if defined(GMX_QEQ_TORCH_CUDA) || defined(GMX_REANN_TORCH_CUDA)
#    define GMX_REANN_ASYNC_HAS_CUDA 1
#    include <ATen/cuda/CUDAEvent.h>
#    include <c10/cuda/CUDAGuard.h>
#    include <c10/cuda/CUDAStream.h>
#else
#    define GMX_REANN_ASYNC_HAS_CUDA 0
#endif

namespace gmx::reann_async
{
// The stream is owned per model; no global default-stream changes. Native
// GROMACS streams are never synchronized by this helper. Current host staging
// already orders native producers/consumers at the bridge boundaries.
class Execution
{
public:
    explicit Execution(const torch::Device& device) : device_(device)
    {
        // Native TorchModel disables graph optimization for model compatibility.
        // Offer an explicit per-call guard for paired tests, without changing
        // other models or conflating TorchScript execution with optimization.
        if(const char* setting=std::getenv("GMX_REANN_GRAPH_OPTIMIZE"))
        {
            if(std::string(setting)=="0"){graphOptimize_=false;}
            else if(std::string(setting)=="1"){graphOptimize_=true;}
            else{throw std::runtime_error("GMX_REANN_GRAPH_OPTIMIZE must be 0 or 1");}
        }
        if (device_.is_cuda())
        {
#if GMX_REANN_ASYNC_HAS_CUDA
            const auto index = device_.has_index() ? device_.index() : c10::cuda::current_device();
            device_ = torch::Device(torch::kCUDA, index);
            stream_.emplace(c10::cuda::getStreamFromPool(false, index));
            ready_ = std::make_unique<at::cuda::CUDAEvent>();
#else
            throw std::runtime_error("REANN CUDA staging needs a CUDA-enabled build");
#endif
        }
    }

    class Guard
    {
    public:
        explicit Guard(const Execution& owner)
        {
            if(owner.graphOptimize_){graphGuard_.emplace(*owner.graphOptimize_);}
#if GMX_REANN_ASYNC_HAS_CUDA
            if (owner.stream_) { guard_.emplace(*owner.stream_); }
#else
            (void)owner;
#endif
        }
    private:
        std::optional<torch::jit::GraphOptimizerEnabledGuard> graphGuard_;
#if GMX_REANN_ASYNC_HAS_CUDA
        std::optional<c10::cuda::CUDAStreamGuard> guard_;
#endif
    };

    // Called ONCE after module.to(device), species, masses and PBC creation
    // on the constructor's current stream. Establish a device-side dependency
    // before this model's stream can read those immutable tensors.
    void adoptInitialization()
    {
#if GMX_REANN_ASYNC_HAS_CUDA
        if (stream_)
        {
            ready_->record(c10::cuda::getCurrentCUDAStream(device_.index()));
            ready_->block(*stream_);
        }
#endif
    }

    void waitForHost()
    {
#if GMX_REANN_ASYNC_HAS_CUDA
        if (stream_)
        {
            ready_->record(*stream_);
            ready_->synchronize(); // only this batch, never cudaDeviceSynchronize
            ++hostFences_;
        }
#endif
    }

    // Failure can leave queued copies referencing pinned buffers. Drain only
    // this model's stream before destroying or reusing any such storage.
    ~Execution()
    {
#if GMX_REANN_ASYNC_HAS_CUDA
        if (stream_)
        {
            try { stream_->synchronize(); } catch (...) { /* destructors cannot throw */ }
        }
#endif
    }
    const torch::Device& device() const { return device_; }
    std::uint64_t hostFences() const { return hostFences_; }

private:
    torch::Device device_;
    std::uint64_t hostFences_ = 0;
    std::optional<bool> graphOptimize_; // unset: preserve existing executor setting
#if GMX_REANN_ASYNC_HAS_CUDA
    std::optional<c10::cuda::CUDAStream> stream_;
    std::unique_ptr<at::cuda::CUDAEvent> ready_;
#endif
};

// Declare Execution AFTER buffers in owning classes, so its destructor drains
// in-flight work BEFORE the pinned buffers are destroyed (reverse member order).
class PackedBuffer
{
public:
    explicit PackedBuffer(const torch::Device& device) : device_(device) {}

    void reserve(std::int64_t elements)
    {
        if (elements < 0) { throw std::runtime_error("Negative REANN buffer length"); }
        if (host_.defined() && elements <= capacity_) { return; }
        capacity_ = std::max<std::int64_t>(elements, std::max<std::int64_t>(64, capacity_*2));
        const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
        host_ = torch::empty({capacity_}, options.pinned_memory(device_.is_cuda()));
        storage_ = device_.is_cuda()
                ? torch::empty({capacity_}, options.device(device_).pinned_memory(false)) : host_;
        ++allocations_;
    }

    double* hostData() { return host_.data_ptr<double>(); }
    const double* hostData() const { return host_.data_ptr<double>(); }

    void put(std::int64_t offset, const double* data, std::int64_t count)
    {
        checkRange(offset, count);
        if (count) { std::memcpy(hostData()+offset, data, count*sizeof(double)); }
    }

    void upload(std::int64_t count)
    {
        checkRange(0, count);
        torch::NoGradGuard noGrad;
        if (device_.is_cuda() && count)
        {
            storage_.narrow(0, 0, count).copy_(host_.narrow(0, 0, count), true);
            ++uploads_;
        }
    }

    torch::Tensor view(std::int64_t offset, std::int64_t count,
                       std::initializer_list<std::int64_t> shape) const
    {
        checkRange(offset, count);
        return storage_.narrow(0, offset, count).view(shape);
    }

    void queue(std::int64_t offset, const torch::Tensor& value,
               std::int64_t count, const char* name, double scale = 1.)
    {
        checkRange(offset, count);
        if (!value.defined() || value.numel() != count || value.scalar_type() != torch::kFloat64
            || value.device() != storage_.device())
        { throw std::runtime_error(std::string("Invalid REANN output layout/device: ")+name); }
        torch::NoGradGuard noGrad;
        if (count)
        {
            auto destination = storage_.narrow(0, offset, count);
            destination.copy_(value.detach().reshape({count}), true);
            if (scale != 1.) { destination.mul_(scale); }
        }
    }

    void download(std::int64_t count, Execution& execution)
    {
        checkRange(0, count);
        torch::NoGradGuard noGrad;
        if (device_.is_cuda() && count)
        {
            host_.narrow(0, 0, count).copy_(storage_.narrow(0, 0, count), true);
            ++downloads_;
        }
        execution.waitForHost();
        // One CPU pass after ONE completed result transfer replaces a separate
        // isfinite(...).all().item<bool>() GPU synchronization for every tensor.
        if (!std::all_of(hostData(), hostData()+count, [](double v) { return std::isfinite(v); }))
        { throw std::runtime_error("Nonfinite REANN output batch"); }
    }

    void get(std::int64_t offset, double* destination, std::int64_t count) const
    {
        checkRange(offset, count);
        if (count) { std::memcpy(destination, hostData()+offset, count*sizeof(double)); }
    }

    std::uint64_t allocations() const { return allocations_; }
    std::uint64_t uploads() const { return uploads_; }
    std::uint64_t downloads() const { return downloads_; }

private:
    void checkRange(std::int64_t offset, std::int64_t count) const
    {
        if (offset < 0 || count < 0 || offset > capacity_ || count > capacity_-offset)
        { throw std::runtime_error("REANN packed buffer range overflow"); }
    }
    torch::Device device_;
    torch::Tensor host_, storage_;
    std::int64_t capacity_ = 0;
    std::uint64_t allocations_ = 0, uploads_ = 0, downloads_ = 0;
};
} // namespace gmx::reann_async
#pragma pop_macro("DIM")
#endif
