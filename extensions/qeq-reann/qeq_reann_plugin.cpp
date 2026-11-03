// LibTorch adapter for the public shaxhch/QEq-REANN C++ forward interface.
// Uses the supplied TorchScript weights unchanged; no Python in the MD loop.
#include "qeq_reann_abi.h"
#include <torch/script.h>
#include <torch/cuda.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
// GMX_QEQ_REANN OPTIMIZATION: static-model graph replay uses a private Torch
// stream; it does not capture or alter any GROMACS force/integration kernels.
#include <cstring>
#include <optional>
#if defined(GMX_QEQ_TORCH_CUDA)
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace
{
void putError(char* buffer, size_t size, const char* message)
{
    if (buffer && size) { std::snprintf(buffer, size, "%s", message); }
}
struct Model
{
    torch::jit::Module module;
    torch::Device device;
    torch::Tensor pbc, species, masses;
    int count;
    bool gradients;
    double totalCharge;
    bool staticModel = false;
    long long predictions = 0;
    torch::Tensor cartBuffer, boxBuffer, hostCart, hostBox, graphOutput;
#if defined(GMX_QEQ_TORCH_CUDA)
    std::optional<c10::cuda::CUDAStream> inferenceStream;
    // Declared last: destroy graph before tensors whose storage it references.
    std::unique_ptr<at::cuda::CUDAGraph> cudaGraph;
#endif
    Model(const GmxQeqOptions& options) :
        module(torch::jit::load(options.modelPath, torch::kCPU)),
        device(options.device), count(options.numAtoms)
    {
        if (count <= 0 || !options.elements || !options.massesU)
        { throw std::runtime_error("Missing QEq elements/masses or empty region"); }
        if (device.is_cuda() && !torch::cuda::is_available())
        { throw std::runtime_error("CUDA QEq requested but this LibTorch/runtime has no CUDA device"); }
        if (options.torchThreads < 1) { throw std::runtime_error("QEq threads must be positive"); }
        at::set_num_threads(options.torchThreads);
        // Keep integer buffers (neighbor/index tables) integer. ScriptModule's
        // C++ to(device,dtype) also casts those buffers, unlike Python double().
        module.to(device);
        for (const auto& parameter : module.parameters())
        {
            if (parameter.is_floating_point() && parameter.scalar_type() != torch::kFloat64)
            { throw std::runtime_error("This bridge expects a float64 QEq TorchScript model"); }
        }
        module.eval();
        // NNMod.forward enumerates ModuleDict children in this exact order.
        const auto networks = module.attr("nnmod").toModule().attr("elemental_nets").toModule();
        std::unordered_map<std::string, int64_t> types;
        for (const auto& child : networks.named_children())
        { types.emplace(child.name, static_cast<int64_t>(types.size())); }
        if (types.size() != static_cast<size_t>(module.attr("gausswidth").toTensor().numel()))
        { throw std::runtime_error("QEq element order and Gaussian widths disagree"); }
        std::vector<int64_t> indices;
        std::istringstream labels(options.elements);
        std::string label;
        while (labels >> label)
        {
            const auto found = types.find(label);
            if (found == types.end()) { throw std::runtime_error("Element absent from QEq model: " + label); }
            indices.push_back(found->second);
        }
        if (indices.size() != static_cast<size_t>(count))
        { throw std::runtime_error("QEq element count differs from selected region"); }
        for (int i = 0; i < count; ++i)
        {
            if (!std::isfinite(options.massesU[i]) || options.massesU[i] <= 0)
            { throw std::runtime_error("QEq mass must be finite and positive (u)"); }
        }
        species = torch::from_blob(indices.data(), {count}, torch::kInt64).clone().to(device);
        masses = torch::from_blob(const_cast<double*>(options.massesU), {count}, torch::kFloat64).clone().to(device);
        pbc = torch::tensor({options.pbc[0], options.pbc[1], options.pbc[2]}, torch::kInt32).to(device);
        const std::string graph = module.get_method("forward").graph()->toString();
        gradients = graph.find("aten::grad(") != std::string::npos;
        totalCharge = module.attr("totcharge").toDouble();
        // GMX_QEQ_REANN OPTIMIZATION: specialization must never silently use
        // another atom order or periodic mode. Validate once on CPU, not in
        // the CUDA graph. Ordinary user TorchScript models keep the old path.
        staticModel = module.hasattr("qeq_static_species");
        if (staticModel)
        {
            const auto expected = module.attr("qeq_static_species").toTensor().cpu().contiguous();
            const auto expectedPbc = module.attr("qeq_static_pbc").toTensor().cpu().contiguous();
            if (expected.scalar_type() != torch::kInt64 || expected.numel() != count
                || expectedPbc.scalar_type() != torch::kInt32 || expectedPbc.numel() != 3)
            { throw std::runtime_error("Invalid static QEq model metadata"); }
            for (int i = 0; i < count; ++i)
            {
                if (expected.data_ptr<int64_t>()[i] != indices[i])
                { throw std::runtime_error("Static QEq model requires its exported atom/element order"); }
            }
            for (int d = 0; d < 3; ++d)
            {
                if (expectedPbc.data_ptr<int>()[d] != 0 || options.pbc[d] != 0)
                { throw std::runtime_error("Static QEq specialization requires model PBC 0 0 0"); }
            }
            if (gradients) { throw std::runtime_error("Static QEq optimization is inference-only"); }
#if defined(GMX_QEQ_TORCH_CUDA)
            if (device.is_cuda())
            {
                // Existing module/constants were copied on the default stream.
                // One startup barrier makes their ownership safe on our stream.
                torch::cuda::synchronize(device.index());
                inferenceStream = c10::cuda::getStreamFromPool(false, device.index());
                c10::cuda::CUDAStreamGuard guard(*inferenceStream);
                cartBuffer = torch::empty({count,3}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
                boxBuffer = torch::empty({3,3}, cartBuffer.options());
                hostCart = torch::empty({count,3}, torch::TensorOptions().dtype(torch::kFloat64).pinned_memory(true));
                hostBox = torch::empty({3,3}, hostCart.options());
                std::fprintf(stderr,"QEq static CUDA optimization enabled: persistent inputs; graph capture after 20 warmup calls\n");
            }
#endif
        }
    }

    torch::Tensor evaluate(const double* xyz, const double* cell)
    {
#if defined(GMX_QEQ_TORCH_CUDA)
        c10::cuda::OptionalCUDAStreamGuard streamGuard;
        if (inferenceStream)
        {
            streamGuard.reset_stream(*inferenceStream);
            std::memcpy(hostCart.data_ptr<double>(),xyz,count*3*sizeof(double));
            std::memcpy(hostBox.data_ptr<double>(),cell,9*sizeof(double));
            cartBuffer.copy_(hostCart,true);
            boxBuffer.copy_(hostBox,true);
            if (!cudaGraph && predictions >= 20)
            {
                inferenceStream->synchronize();
                cudaGraph = std::make_unique<at::cuda::CUDAGraph>();
                cudaGraph->capture_begin();
                try
                {
                    graphOutput = unpack(module.forward({pbc,cartBuffer,boxBuffer,species,masses}));
                }
                catch (...)
                {
                    // End capture on failure as well, then surface the error.
                    // Never continue with a partially captured/stale result.
                    try { cudaGraph->capture_end(); } catch (...) {}
                    cudaGraph.reset();
                    throw;
                }
                cudaGraph->capture_end();
                std::fprintf(stderr,"QEq static CUDA graph captured; replaying every MD charge evaluation\n");
            }
            if (cudaGraph)
            {
                cudaGraph->replay();
                ++predictions;
                // Exactly one blocking return is required by current CPU MD
                // integration. It also protects pinned input reuse next call.
                return graphOutput.detach().to(torch::kCPU,torch::kFloat64).contiguous().view({-1});
            }
            auto result = unpack(module.forward({pbc,cartBuffer,boxBuffer,species,masses}));
            ++predictions;
            return result.detach().to(torch::kCPU,torch::kFloat64).contiguous().view({-1});
        }
#endif
        // ORIGINAL | per-call tensor transfers retained for ordinary models.
        auto cart = torch::from_blob(const_cast<double*>(xyz), {count, 3}, torch::kFloat64).to(device);
        auto box = torch::from_blob(const_cast<double*>(cell), {3, 3}, torch::kFloat64).to(device);
        return unpack(module.forward({pbc,cart,box,species,masses})).detach().to(torch::kCPU,torch::kFloat64).contiguous().view({-1});
    }

    static torch::Tensor unpack(const torch::jit::IValue& value)
    {
        if (value.isTensor()) { return value.toTensor(); }
        if (value.isTuple() && !value.toTuple()->elements().empty())
        { return value.toTuple()->elements()[0].toTensor(); }
        throw std::runtime_error("QEq forward must return a charge Tensor or a nonempty tuple");
    }
};
}
extern "C" int gmx_qeq_abi_version() { return 1; }
extern "C" void* gmx_qeq_create(const GmxQeqOptions* options, double* total,
                                 char* error, size_t size)
{
    try
    {
        if (!options || options->abiVersion != 1)
        { throw std::runtime_error("Unsupported GROMACS QEq ABI"); }
        auto model = std::make_unique<Model>(*options);
        *total = model->module.attr("totcharge").toDouble();
        return model.release();
    }
    catch (const std::exception& e) { putError(error, size, e.what()); return nullptr; }
}
extern "C" int gmx_qeq_predict(void* pointer, const double* xyz, const double* cell,
                                double* charges, double* multiplier, double* milliseconds,
                                char* error, size_t size)
{
    try
    {
        auto& model = *static_cast<Model*>(pointer);
        const auto start = std::chrono::steady_clock::now();
        at::AutoGradMode mode(model.gradients);
        // ORIGINAL | auto cart = torch::from_blob(const_cast<double*>(xyz), {model.count, 3}, torch::kFloat64).to(model.device);
        // ORIGINAL | auto box = torch::from_blob(const_cast<double*>(cell), {3, 3}, torch::kFloat64).to(model.device);
        // ORIGINAL | auto value = model.module.forward({model.pbc, cart, box, model.species, model.masses});
        // ORIGINAL | torch::Tensor result;
        // ORIGINAL | if (value.isTensor()) { result = value.toTensor(); }
        // ORIGINAL | else if (value.isTuple() && !value.toTuple()->elements().empty())
        // ORIGINAL | { result = value.toTuple()->elements()[0].toTensor(); }
        // ORIGINAL | else { throw std::runtime_error("QEq forward must return a charge Tensor or a nonempty tuple"); }
        // ORIGINAL | result = result.detach().to(torch::kCPU, torch::kFloat64).contiguous().view({-1});
        auto result = model.evaluate(xyz,cell);
        // QEq's augmented solve returns N charges and one Lagrange multiplier.
        if (result.numel() != model.count && result.numel() != model.count + 1)
        { throw std::runtime_error("QEq result must contain N or N+1 values"); }
        const auto* data = result.data_ptr<double>();
        double sum = 0;
        for (int i = 0; i < model.count; ++i)
        {
            if (!std::isfinite(data[i])) { throw std::runtime_error("QEq returned a nonfinite charge"); }
            charges[i] = data[i]; // elementary charge e, NO Amber scaling
            sum += data[i];
        }
        // ORIGINAL | if (std::abs(sum - model.module.attr("totcharge").toDouble()) > 1e-5)
        if (std::abs(sum - model.totalCharge) > 1e-5)
        { throw std::runtime_error("QEq atomic charges violate the model total charge"); }
        if (multiplier) { *multiplier = result.numel() == model.count + 1 ? data[model.count] : 0; }
        if (milliseconds)
        { *milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count(); }
        return 0;
    }
    catch (const std::exception& e) { putError(error, size, e.what()); return 1; }
}
extern "C" void gmx_qeq_destroy(void* pointer) { delete static_cast<Model*>(pointer); }
