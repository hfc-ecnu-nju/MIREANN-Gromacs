// GMX_MIREANN NEW FILE: original MIREANN TorchScript inference and charge-response VJP.
// Local extension, LGPL-2.1-or-later. Reference: supplied MIREANN/ee/PES.py and pes_cpp.cpp.
// The supplied .pt is authoritative: do not reconstruct its formula from training code.
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h> // GMX_MIREANN_GPU_AUDIT
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "mireann_abi.h"
#include "mireann_classical_abi.h" // GMX_MIREANN_GPU_AUDIT: no Torch types in ABI
#include "reann_async_buffers.h" // GMX_REANN_ASYNC: reusable packed pinned/device buffers

namespace
{
void errorText(char* error, size_t size, const char* message)
{
    if (error && size) { std::snprintf(error, size, "%s", message); }
}
struct MireannModel
{
    torch::jit::Module module;
    torch::Device device;
    torch::Tensor pbc, species, masses, energy, charges, background;
    int count, numMm = 0;
    double total;
    bool pending = false;
    bool backgroundResponse = true; // GMX_REANN_ASYNC: old C clients keep original behavior
    // GMX_REANN_ASYNC: input storage remains alive until the matching backward.
    // Execution is declared after buffers: its destructor drains work first.
    gmx::reann_async::PackedBuffer inputBatch, potentialBatch, outputBatch;
    gmx::reann_async::Execution execution;
    // GMX_MIREANN_GPU_AUDIT: original forward remains available unchanged.
    // Only the private energy/charge method omits unused internal gradients.
    bool energyChargeOnly = true;
#if defined(GMX_MIREANN_CUDA_CLASSICAL)
    std::unique_ptr<void,decltype(&gmx_mireann_cuda_classical_destroy)> classical{
        nullptr,&gmx_mireann_cuda_classical_destroy};
    int classicalDevice=-1;
#endif

    // The public forward detaches its input coordinates before computing forces.
    // Store references to those SAME internal leaves, without modifying any formula,
    // weight, input, output or sign. This permits an exact dq/dR vector-Jacobian product
    // after GROMACS has computed the long-range electrostatic potential.
    // ORIGINAL | void exposeLeaf(int argument, const char* name)
    void exposeLeaf(const std::shared_ptr<torch::jit::Graph>& graph,int argument,const char* name)
    {
        // ORIGINAL | const auto graph = module.get_method("forward").graph();
        torch::jit::Value* leaf = nullptr;
        for (auto* node : graph->nodes())
        {
            if (node->kind() == c10::Symbol::fromQualString("aten::clone"))
            {
                auto* parent = node->input(0)->node();
                if (parent->kind() == c10::Symbol::fromQualString("aten::detach")
                    && parent->input(0) == graph->inputs().at(argument))
                {
                    if (leaf) { throw std::runtime_error("Ambiguous MIREANN coordinate leaf"); }
                    leaf = node->output();
                }
            }
        }
        if (!leaf)
        {
            throw std::runtime_error("MIREANN model lacks the supported detach/clone coordinate interface; cannot compute a consistent charge-response force");
        }
        module.register_attribute(name, c10::TensorType::get(), torch::empty({0}));
        auto* store = graph->create(c10::prim::SetAttr, {graph->inputs().at(0), leaf}, 0);
        store->s_(c10::attr::name, name);
        store->insertBefore(graph->return_node());
        graph->lint();
    }

    // GMX_MIREANN_GPU_AUDIT: preserve the supplied file and public forward.
    // A copied graph retains the EXACT energy and charge expressions. DCE
    // removes only unused outputs. Explicitly remove aten::grad only after all
    // of its outputs are dead; never discard live or model-dependent gradients.
    void prepareEnergyChargeMethod()
    {
        auto graph=module.get_method("forward").graph()->copy();
        exposeLeaf(graph,2,"_gmx_mireann_qm_leaf");
        exposeLeaf(graph,3,"_gmx_mireann_mm_leaf");
        auto* original=graph->outputs().at(0)->node();
        if(original->kind()!=c10::prim::TupleConstruct || original->inputs().size()!=4)
        { throw std::runtime_error("Unsupported MIREANN return graph"); }
        auto* tuple=graph->createTuple({original->input(0),original->input(3)});
        tuple->insertBefore(graph->return_node());
        graph->eraseOutput(0);graph->registerOutput(tuple->output());
        torch::jit::EliminateDeadCode(graph);
        std::vector<torch::jit::Node*> deadGradients;
        for(auto* node:graph->nodes())
        {
            if(node->kind()==c10::Symbol::fromQualString("aten::grad")
               && std::all_of(node->outputs().begin(),node->outputs().end(),
                    [](torch::jit::Value* value){return value->uses().empty();}))
            { deadGradients.push_back(node); }
        }
        for(auto* node:deadGradients){node->destroy();}
        torch::jit::EliminateDeadCode(graph);graph->lint();
        auto* function=module.type()->compilation_unit()->create_function(
            c10::QualifiedName(*module.type()->name(),"_gmx_mireann_energy_charge"),graph);
        module.type()->addMethod(function);
    }

    explicit MireannModel(const GmxQeqOptions& options) :
        module(torch::jit::load(options.modelPath, torch::kCPU)),
        device(options.device ? options.device : "cpu"), count(options.numAtoms),
        total(module.attr("totcharge").toDouble()),
        inputBatch(device), potentialBatch(device), outputBatch(device), execution(device)
    {
        if (options.abiVersion != 1 || count < 1 || options.torchThreads < 1)
        { throw std::runtime_error("Invalid MIREANN create options"); }
        if (device.type() != torch::kCPU && device.type() != torch::kCUDA)
        { throw std::runtime_error("MIREANN supports cpu or cuda:N"); }
        if (device.is_cuda() && !torch::cuda::is_available())
        { throw std::runtime_error("MIREANN CUDA device requested but LibTorch CUDA is unavailable"); }
        const auto& schema = module.get_method("forward").function().getSchema();
        if (schema.arguments().size() != 8 || schema.returns().size() != 1)
        { throw std::runtime_error("MIREANN requires seven inputs: pbc, QM coordinates, MM coordinates, MM q(e), cell, species, mass; the five-input gas QEq model cannot be used"); }
        const auto tuple = schema.returns()[0].type()->cast<c10::TupleType>();
        if (!tuple || tuple->elements().size() != 4)
        { throw std::runtime_error("MIREANN must return energy, QM gradient, MM gradient, QM charges"); }
        torch::set_num_threads(options.torchThreads);
        for (auto parameter : module.parameters())
        {
            if (parameter.is_floating_point() && parameter.scalar_type() != torch::kFloat64)
            { throw std::runtime_error("MIREANN currently requires a double-precision model"); }
            parameter.set_requires_grad(false);
        }
        std::unordered_map<std::string, int64_t> elementIndex;
        int64_t index = 0;
        for (const auto& child : module.attr("nnmod").toModule().attr("elemental_nets").toModule().named_children())
        { elementIndex.emplace(child.name, index++); }
        std::istringstream input(options.elements ? options.elements : "");
        std::string label;
        std::vector<int64_t> indices;
        std::vector<double> massValues;
        while (input >> label)
        {
            const auto found = elementIndex.find(label);
            if (found == elementIndex.end() || indices.size() >= static_cast<size_t>(count))
            { throw std::runtime_error("MIREANN element absent from model or too many elements: " + label); }
            const double mass = options.massesU[indices.size()];
            if (!std::isfinite(mass) || mass <= 0) { throw std::runtime_error("Invalid QM topology mass"); }
            indices.push_back(found->second);
            massValues.push_back(mass);
        }
        if (indices.size() != static_cast<size_t>(count)) { throw std::runtime_error("Wrong MIREANN QM element count"); }
        // Preserve integer neighbor buffers; never convert an entire module to double.
        module.to(device);
        module.eval();
        species = torch::tensor(indices, torch::TensorOptions().dtype(torch::kInt64).device(device));
        masses = torch::tensor(massValues, torch::TensorOptions().dtype(torch::kFloat64).device(device));
        pbc = torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        // ORIGINAL | exposeLeaf(2, "_gmx_mireann_qm_leaf");
        // ORIGINAL | exposeLeaf(3, "_gmx_mireann_mm_leaf");
        // Diagnostic switch reproduces the old double-backward path for paired
        // tests. It never changes energies, charge predictions or force terms.
        energyChargeOnly=std::getenv("GMX_MIREANN_KEEP_MODEL_BACKWARD")==nullptr;
        if(energyChargeOnly){prepareEnergyChargeMethod();}
        else
        {
            auto graph=module.get_method("forward").graph();
            exposeLeaf(graph,2,"_gmx_mireann_qm_leaf");
            exposeLeaf(graph,3,"_gmx_mireann_mm_leaf");
        }
        execution.adoptInitialization(); // GMX_REANN_ASYNC: constructor stream -> model stream
    }
    // ORIGINAL | torch::Tensor input(const double* data, std::initializer_list<int64_t> shape)
    // ORIGINAL | { auto result=torch::empty(shape,torch::TensorOptions().dtype(torch::kFloat64));
    // ORIGINAL |   if(result.numel()){std::memcpy(result.data_ptr<double>(),data,result.numel()*sizeof(double));}
    // ORIGINAL |   return result.to(device); }
    // GMX_REANN_ASYNC: phi has its own buffer; never overwrite forward storage
    // while autograd can still refer to it. No per-call tensor storage allocation.
    torch::Tensor input(const double* data, std::initializer_list<int64_t> shape)
    {
        int64_t count = 1;
        for (auto dimension : shape) { count *= dimension; }
        potentialBatch.reserve(count);
        potentialBatch.put(0, data, count);
        potentialBatch.upload(count);
        return potentialBatch.view(0, count, shape);
    }
};
void copyFinite(const torch::Tensor& tensor, double* destination, int64_t expected, const char* name)
{
    if (!tensor.defined() || tensor.numel() != expected || !torch::isfinite(tensor).all().item<bool>())
    { throw std::runtime_error(std::string("Invalid MIREANN ") + name); }
    auto host = tensor.detach().to(torch::kCPU, torch::kFloat64).contiguous();
    if (expected) { std::memcpy(destination, host.data_ptr<double>(), expected*sizeof(double)); }
}
}

extern "C" void* gmx_mireann_create(const GmxQeqOptions* options, double* totalChargeE,
                                    char* error, size_t size)
{
    try
    {
        if (!options) { throw std::runtime_error("Null MIREANN create options"); }
        auto model = std::make_unique<MireannModel>(*options);
        if (totalChargeE) { *totalChargeE = model->total; }
        return model.release();
    }
    catch (const std::exception& exception) { errorText(error, size, exception.what()); return nullptr; }
}
extern "C" int gmx_mireann_predict(void* pointer, int numMm, const double* qm, const double* mm,
                                   const double* mmQ, const double* box, double* qmQ,
                                   double* energy, double* elapsed, char* error, size_t size)
{
    try
    {
        auto& model = *static_cast<MireannModel*>(pointer);
        if (numMm < 0) { throw std::runtime_error("Negative MIREANN MM count"); }
        if (model.pending) { throw std::runtime_error("MIREANN prediction would overwrite an unfinished force graph"); }
        const auto start = std::chrono::steady_clock::now();
        gmx::reann_async::Execution::Guard streamGuard(model.execution);
        at::AutoGradMode gradients(true);
        model.pending = false;
        model.numMm = numMm;
        // ORIGINAL | model.background=model.input(mmQ,{numMm});
        // GMX_REANN_ASYNC: one pinned H2D batch, same values/order/units.
        const int64_t qmSize=3LL*model.count, mmSize=3LL*numMm;
        const int64_t mmOffset=qmSize, chargeOffset=mmOffset+mmSize;
        const int64_t cellOffset=chargeOffset+numMm, inputSize=cellOffset+9;
        model.inputBatch.reserve(inputSize);
        model.inputBatch.put(0,qm,qmSize);
        model.inputBatch.put(mmOffset,mm,mmSize);
        model.inputBatch.put(chargeOffset,mmQ,numMm);
        model.inputBatch.put(cellOffset,box,9);
        model.inputBatch.upload(inputSize);
        model.background = model.inputBatch.view(chargeOffset,numMm,{numMm}).detach();
        // ORIGINAL | model.background.set_requires_grad(true);
        model.background.set_requires_grad(model.backgroundResponse);
        // ORIGINAL | const auto output = model.module.forward({model.pbc, model.input(qm, {model.count,3}),
        // ORIGINAL |           model.input(mm, {numMm,3}), model.background, model.input(box, {3,3}),
        // ORIGINAL |           model.species, model.masses}).toTuple()->elements();
        // ORIGINAL | inputs={pbc,input(qm,...),input(mm,...),background,input(box,...),species,masses};
        const std::vector<torch::jit::IValue> inputs={model.pbc,
            model.inputBatch.view(0,qmSize,{model.count,3}),
            model.inputBatch.view(mmOffset,mmSize,{numMm,3}),model.background,
            model.inputBatch.view(cellOffset,9,{3,3}),model.species,model.masses};
        const auto output=(model.energyChargeOnly
            ? model.module.get_method("_gmx_mireann_energy_charge")(inputs)
            : model.module.forward(inputs)).toTuple()->elements();
        model.energy = output.at(0).toTensor();
        // ORIGINAL | model.charges = output.at(3).toTensor().reshape({-1});
        model.charges=output.at(model.energyChargeOnly ? 1 : 3).toTensor().reshape({-1});
        // ORIGINAL | copyFinite(model.energy,energy,1,"electrostatic energy");
        // ORIGINAL | copyFinite(model.charges,qmQ,model.count,"QM charges");
        // GMX_REANN_ASYNC: energy and charges share one D2H result/fence.
        model.outputBatch.reserve(1+model.count);
        model.outputBatch.queue(0,model.energy,1,"electrostatic energy");
        model.outputBatch.queue(1,model.charges,model.count,"QM charges");
        model.outputBatch.download(1+model.count,model.execution);
        model.outputBatch.get(0,energy,1);
        model.outputBatch.get(1,qmQ,model.count);
        double sum = 0;
        for (int i=0; i<model.count; ++i) { sum += qmQ[i]; }
        if (std::abs(sum-model.total) > 1e-5) { throw std::runtime_error("MIREANN violates the QM total charge"); }
        // GMX_MIREANN_GPU_AUDIT: private method intentionally has no gradients.
        // The only backward, including the residual VJP, is in forces() below.
        // ORIGINAL | if (!output.at(1).isTensor() || !output.at(2).isTensor())
        // ORIGINAL | { throw std::runtime_error("MIREANN must return defined QM and MM energy gradients"); }
        // ORIGINAL | if (output.at(1).toTensor().numel() != model.count*3
        // ORIGINAL |     || output.at(2).toTensor().numel() != numMm*3)
        // ORIGINAL | { throw std::runtime_error("MIREANN coordinate gradient shape mismatch"); }
        if (!model.energyChargeOnly && (!output.at(1).isTensor() || !output.at(2).isTensor()))
        { throw std::runtime_error("MIREANN must return defined QM and MM energy gradients"); }
        if (!model.energyChargeOnly && (output.at(1).toTensor().numel() != model.count*3
            || output.at(2).toTensor().numel() != numMm*3))
        { throw std::runtime_error("MIREANN coordinate gradient shape mismatch"); }
        model.pending = true;
        if (elapsed) { *elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); }
        return 0;
    }
    catch (const std::exception& exception) { errorText(error,size,exception.what()); return 1; }
}
extern "C" int gmx_mireann_forces(void* pointer, const double* phi, double scale,
                                  double* qmForces, double* mmForces, double* bgDerivative,
                                  char* error, size_t size)
{
    try
    {
        auto& model=*static_cast<MireannModel*>(pointer);
        if (!model.pending || !std::isfinite(scale) || scale<=0)
        { throw std::runtime_error("MIREANN response requires one preceding successful prediction and an explicit energy scale"); }
        gmx::reann_async::Execution::Guard streamGuard(model.execution);
        at::AutoGradMode enableGradients(true);
        // GMX_MIREANN: the model's returned QM/MM gradients ALREADY contain
        // dq/dR for its own local energy. Re-differentiating that same energy
        // below replaces those returned gradients; it is never added to them.
        // phi contains ONLY d(U_GROMACS - U_bare_local)/dq. Thus q*phi supplies
        // the residual periodic-electrostatic response, not a second copy of
        // the model's local charge response. This scalar is for a VJP only;
        // it must not be added to the reported physical energy.
        auto weightedEnergy = scale*model.energy + (model.charges*model.input(phi,{model.count})).sum();
        // ORIGINAL | grad(weightedEnergy,{qmLeaf,mmLeaf,background},{},false,false,true);
        // GMX_REANN_ASYNC: with a hard sphere the caller never uses dE/dq_MM.
        // Do not traverse that extra backward branch. This does NOT disable
        // dq_QM/dR or the force on any QM/MM coordinate.
        std::vector<torch::Tensor> leaves={model.module.attr("_gmx_mireann_qm_leaf").toTensor(),
                                         model.module.attr("_gmx_mireann_mm_leaf").toTensor()};
        if(model.backgroundResponse){leaves.push_back(model.background);}
        auto gradient = torch::autograd::grad({weightedEnergy},leaves,{},false,false,true);
        // Official MIREANN outputs dE/dR; forces are -dE/dR. Coordinates are A,
        // weightedEnergy is kJ/mol, hence the factor -10 for GROMACS force units.
        // ORIGINAL | copyFinite(-10.0*gradient.at(0),qmForces,model.count*3,"QM response forces");
        // ORIGINAL | copyFinite(-10.0*gradient.at(1),mmForces,model.numMm*3,"MM response forces");
        // ORIGINAL | if(gradient.at(2).defined()){copyFinite(gradient.at(2),bgDerivative,model.numMm,"background-charge derivatives");}
        // ORIGINAL | else{std::fill_n(bgDerivative,model.numMm,0.0);}
        // GMX_REANN_ASYNC: pack all force outputs on the device, then transfer
        // and validate once. The required pre-integration host fence remains.
        const int64_t nq=3LL*model.count,nm=3LL*model.numMm,total=nq+nm+model.numMm;
        model.outputBatch.reserve(total);
        model.outputBatch.queue(0,gradient.at(0),nq,"QM response forces",-10.);
        model.outputBatch.queue(nq,gradient.at(1),nm,"MM response forces",-10.);
        if(model.backgroundResponse && gradient.at(2).defined())
        { model.outputBatch.queue(nq+nm,gradient.at(2),model.numMm,"background-charge derivatives"); }
        else
        {
            torch::NoGradGuard noGrad;
            model.outputBatch.view(nq+nm,model.numMm,{model.numMm}).zero_();
        }
        model.outputBatch.download(total,model.execution);
        model.outputBatch.get(0,qmForces,nq);
        model.outputBatch.get(nq,mmForces,nm);
        model.outputBatch.get(nq+nm,bgDerivative,model.numMm);
        model.pending=false;
        // Release step-specific autograd storage; the model itself stays loaded.
        model.energy=torch::Tensor(); model.charges=torch::Tensor(); model.background=torch::Tensor();
        model.module.setattr("_gmx_mireann_qm_leaf",torch::empty({0}));
        model.module.setattr("_gmx_mireann_mm_leaf",torch::empty({0}));
        return 0;
    }
    catch (const std::exception& exception) { errorText(error,size,exception.what()); return 1; }
}
extern "C" void gmx_mireann_destroy(void* pointer) { delete static_cast<MireannModel*>(pointer); }

extern "C" int gmx_mireann_set_background_response(void* pointer,int enabled,char* error,size_t size)
{
    try
    {
        auto& model=*static_cast<MireannModel*>(pointer);
        if(model.pending){throw std::runtime_error("Cannot change MIREANN derivative mode during a pending step");}
        model.backgroundResponse=enabled!=0;
        return 0;
    }
    catch(const std::exception& exception){errorText(error,size,exception.what());return 1;}
}

// GMX_MIREANN_GPU_AUDIT: GPU acceleration of the SAME classical derivatives.
// CPU-only LibTorch and builds without CUDA retain the original CPU formula.
extern "C" int gmx_mireann_classical(void* pointer,const GmxMireannClassicalInput* input,
    GmxMireannClassicalOutput* output,int* usedGpu,char* error,size_t size)
{
    try
    {
        *usedGpu=0;
#if defined(GMX_MIREANN_CUDA_CLASSICAL)
        auto& model=*static_cast<MireannModel*>(pointer);
        if(input->cudaDevice>=0)
        {
            if(!model.classical)
            { model.classical.reset(gmx_mireann_cuda_classical_create(input->cudaDevice));model.classicalDevice=input->cudaDevice; }
            if(model.classicalDevice!=input->cudaDevice)
            { throw std::runtime_error("MIREANN classical CUDA device cannot change during a run"); }
            gmx_mireann_cuda_classical_compute(model.classical.get(),input,output);
            *usedGpu=1;
        }
#else
        (void)pointer;(void)input;(void)output;
#endif
        return 0;
    }
    catch(const std::exception& exception){errorText(error,size,exception.what());return 1;}
}
