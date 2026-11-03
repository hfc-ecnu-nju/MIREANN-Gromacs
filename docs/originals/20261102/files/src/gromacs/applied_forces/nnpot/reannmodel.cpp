// GMX_REANN_PES NEW FILE BEGIN: native NNPot adapter, LGPL-2.1-or-later.
// Reuses the public REANN five-input, energy/force TorchScript interface.
// No Python interpreter, external process, training, or model conversion in MD.
#include "torchmodel.h" // Torch headers must precede GROMACS DIM macros.
#include "reannmodel.h"
#include <sstream> // GMX_MIREANN_PSEUDOBOND model-only species

#include <cmath>
#include <set>
#include <unordered_map>
#include <vector>

// GMX_REANN_PES REVIEW FIX: declare CUDA runtime APIs, as the upstream Torch adapter does.
#include "gromacs/hardware/device_information.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/atomprop.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"
#include "nnpotoptions.h"

namespace gmx
{
namespace
{
torch::Device reannDevice(const std::string& requested, bool validationOnly)
{
    if (requested != "auto" && requested != "cpu" && requested != "gpu")
    {
        GMX_THROW(InconsistentInputError("nnpot-reann-device must be auto, cpu or gpu"));
    }
    // grompp can validate a CUDA model on a CPU login node.
    if (validationOnly || requested == "cpu") { return torch::Device(torch::kCPU); }
    if (!torch::cuda::is_available())
    {
        if (requested == "gpu")
        {
            GMX_THROW(InconsistentInputError("REANN GPU inference requested, but CUDA is unavailable"));
        }
        return torch::Device(torch::kCPU);
    }
    int index = 0;
#if GMX_GPU_CUDA
    if (cudaGetDevice(&index) != cudaSuccess)
    {
        GMX_THROW(InternalError("Cannot query the active GROMACS CUDA device for REANN"));
    }
#endif
    return torch::Device(torch::kCUDA, index);
}
} // namespace

struct ReannModel::Impl
{
    torch::jit::Module model;
    torch::Device device;
    torch::Tensor species, masses, pbc;
    std::vector<int> atomicNumbers;
    double energyScale;

    Impl(const NNPotParameters& params, const MpiComm& comm, bool validationOnly) :
        model(torch::jit::load(params.modelFileName_, torch::kCPU)),
        device(reannDevice(params.reannDevice_, validationOnly)),
        energyScale(params.reannEnergyScale_)
    {
        if (comm.isParallel())
        {
            GMX_THROW(NotImplementedError("REANN currently requires one rank (-ntmpi 1)"));
        }
        if (params.embeddingScheme_ != NNPotEmbedding::Mechanical || !params.linkFrontier_.empty())
        {
            GMX_THROW(NotImplementedError(
                    "REANN molecular PES requires mechanical embedding and whole-molecule NNP regions; "
                    "electrostatic-model embedding and link atoms are not supported"));
        }
        if (!std::isfinite(energyScale) || energyScale <= 0)
        {
            GMX_THROW(InconsistentInputError(
                    "Specify nnpot-reann-energy-scale: kJ/mol per model energy unit. "
                    "For eV use 96.4853321233; kcal/mol 4.184; kJ/mol 1. Units cannot be inferred from .pt"));
        }
        for (const auto& input : params.modelInput_)
        {
            if (!input.empty())
            {
                GMX_THROW(InconsistentInputError(
                        "REANN supplies its own five inputs; leave nnpot-model-input1..9 empty"));
            }
        }
        const auto& schema = model.get_method("forward").function().getSchema();
        if (schema.arguments().size() != 6 || schema.returns().size() != 1)
        {
            GMX_THROW(InconsistentInputError("REANN forward must accept pbc, cart, cell, species, mass"));
        }
        auto outputType = schema.returns()[0].type();
        if (const auto optional = outputType->cast<c10::OptionalType>())
        {
            outputType = optional->getElementType();
        }
        const auto tuple = outputType->cast<c10::TupleType>();
        if (!tuple || tuple->elements().size() != 2)
        {
            GMX_THROW(InconsistentInputError("REANN PES must return (energy, forces), not a charge model"));
        }
        for (const auto& parameter : model.parameters())
        {
            if (parameter.is_floating_point() && parameter.scalar_type() != torch::kFloat64)
            {
                GMX_THROW(InconsistentInputError("REANN adapter requires a float64 PES model"));
            }
        }
        const auto networks = model.attr("nnmod").toModule().attr("elemental_nets").toModule();
        AtomProperties atomProperties;
        std::unordered_map<int, int64_t> typeMap;
        for (const auto& child : networks.named_children())
        {
            const int z = atomProperties.atomNumberFromElement(child.name.c_str());
            const auto type = static_cast<int64_t>(typeMap.size());
            if (z <= 0 || !typeMap.emplace(z, type).second)
            {
                GMX_THROW(InconsistentInputError("REANN elemental_nets contains an unrecognized/duplicate element"));
            }
        }
        // GMX_MIREANN_PSEUDOBOND: COX input_reann uses F with carbon mass.
        // Override only model labels, never the physical topology atomic Z.
        std::vector<int> modelNumbers;
        std::istringstream labels(params.reannElements_);
        std::string label;
        while (labels>>label)
        {
            const int z=atomProperties.atomNumberFromElement(label.c_str());
            if (z<=0) { GMX_THROW(InconsistentInputError("Unknown nnpot-reann-elements label")); }
            modelNumbers.push_back(z);
        }
        if (!modelNumbers.empty() && (params.boundary_!="pseudo-bond" || modelNumbers.size()!=params.nnpIndices_.size()))
        { GMX_THROW(InconsistentInputError("nnpot-reann-elements requires pseudo-bond and one label per QM input atom")); }
        std::vector<int64_t> types;
        std::vector<double> massValues;
        std::set<int> seen;
        for (const auto id : params.nnpIndices_)
        {
            if (id < 0 || id >= params.atoms_.nr || !seen.insert(id).second)
            {
                GMX_THROW(InconsistentInputError("Invalid/duplicate atom index in REANN NNP region"));
            }
            const auto& atom = params.atoms_.atom[id];
            // ORIGINAL | const auto found = typeMap.find(atom.atomnumber);
            const int modelZ=modelNumbers.empty() ? atom.atomnumber : modelNumbers[types.size()];
            const auto found = typeMap.find(modelZ);
            if (found == typeMap.end() || !std::isfinite(atom.m) || atom.m <= 0)
            {
                GMX_THROW(InconsistentInputError(
                        "REANN requires positive topology masses and atomic numbers present in elemental_nets"));
            }
            types.push_back(found->second);
            atomicNumbers.push_back(atom.atomnumber);
            massValues.push_back(atom.m);
        }
        if (types.empty()) { GMX_THROW(InconsistentInputError("REANN atom group is empty")); }
        const auto count = static_cast<int64_t>(types.size());
        species = torch::from_blob(types.data(), {count}, torch::kInt64).clone().to(device);
        masses = torch::from_blob(massValues.data(), {count}, torch::kFloat64).clone().to(device);
        // A molecular QM region in a periodic solvent is still a nonperiodic
        // model. REANN internally makes its coordinates whole using the box.
        pbc = torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        // Do not to(device,dtype): it would also cast integer neighbor buffers.
        model.to(device);
        model.eval();
    }
};

ReannModel::ReannModel(const NNPotParameters& params, const MDLogger& logger,
                       const MpiComm& mpiComm, bool validationOnly) :
    impl_(std::make_unique<Impl>(params, mpiComm, validationOnly))
{
    GMX_LOG(logger.info).appendTextFormatted(
            "GMX_REANN_PES: native NNPot, %zu atoms, device=%s, float64; "
            "nm->Angstrom; E scale %.12g, F scale %.12g; molecular model PBC=0 0 0.",
            impl_->atomicNumbers.size(), impl_->device.str().c_str(),
            impl_->energyScale, impl_->energyScale*10.0);
}

ReannModel::~ReannModel() = default;

void ReannModel::evaluateModel(gmx_enerdata_t* enerd, ArrayRef<RVec> forces,
                              ArrayRef<const int> indexLookup, ArrayRef<const int>,
                              ArrayRef<const std::string>, ArrayRef<RVec> positions,
                              ArrayRef<int> atomNumbers, ArrayRef<int>, ArrayRef<RVec>,
                              ArrayRef<RVec>, ArrayRef<real>, real,
                              ArrayRef<const LinkFrontierAtom> linkFrontier,
                              matrix* box, PbcType*)
{
    auto& state = *impl_;
    const auto n = static_cast<int64_t>(state.atomicNumbers.size());
    if (!box || !linkFrontier.empty() || positions.ssize() != n
        || atomNumbers.ssize() != n || indexLookup.ssize() != n)
    {
        GMX_THROW(InconsistentInputError("REANN input shape, box or link frontier is inconsistent"));
    }
    for (int64_t i = 0; i < n; ++i)
    {
        if (atomNumbers[i] != state.atomicNumbers[i] || indexLookup[i] < 0
            || indexLookup[i] >= forces.ssize())
        {
            GMX_THROW(InconsistentInputError("REANN atom ordering/local force mapping changed unexpectedly"));
        }
        for (int d = 0; d < DIM; ++d)
        {
            if (!std::isfinite(positions[i][d]))
            { GMX_THROW(InconsistentInputError("Nonfinite REANN input coordinate")); }
        }
    }
    const auto& b = *box;
    const double determinant = b[0][0]*(b[1][1]*b[2][2]-b[1][2]*b[2][1])
                             -b[0][1]*(b[1][0]*b[2][2]-b[1][2]*b[2][0])
                             +b[0][2]*(b[1][0]*b[2][1]-b[1][1]*b[2][0]);
    if (!std::isfinite(determinant) || determinant <= 1e-12)
    {
        GMX_THROW(InconsistentInputError("REANN needs a finite, positive-volume model box"));
    }
    auto cart = torch::from_blob(positions.data()->as_vec(), {n,DIM}, torchRealType)
                        .to(state.device,torch::kFloat64)*10.0;
    auto cell = torch::from_blob(*box, {DIM,DIM}, torchRealType)
                        .to(state.device,torch::kFloat64)*10.0;
    // REANN computes autograd.grad internally, even though its outputs detach.
    // A NoGradGuard or InferenceMode here would disable the required force graph.
    at::AutoGradMode enableGrad(true);
    const auto output = state.model.forward({state.pbc,cart,cell,state.species,state.masses});
    if (!output.isTuple() || output.toTuple()->elements().size() != 2)
    { GMX_THROW(InternalError("REANN PES did not return (energy,forces)")); }
    const auto& values = output.toTuple()->elements();
    if (!values[0].isTensor() || !values[1].isTensor())
    { GMX_THROW(InternalError("REANN energy and forces must both be tensors")); }
    const auto energy = values[0].toTensor();
    const auto rawForce = values[1].toTensor().to(state.device,torch::kFloat64);
    if (energy.numel() != 1 || rawForce.dim() != 2 || rawForce.size(0) != n || rawForce.size(1) != DIM)
    { GMX_THROW(InternalError("REANN returned an invalid energy/force shape")); }
    torch::NoGradGuard noGrad;
    // Molecular virial only. Match REANN's own whole-molecule/COM transform,
    // including its fractional rounding, rather than using wrapped MD positions.
    auto tensorVirial = torch::zeros({DIM,DIM},torch::kFloat64);
    if (computeVirial_)
    {
        const auto fractional = torch::matmul(cart,torch::linalg_inv(cell));
        auto whole = torch::matmul(fractional-torch::round(fractional-fractional[0]),cell);
        whole = whole-(state.masses.unsqueeze(1)*whole).sum(0)/state.masses.sum();
        tensorVirial = (-0.5*torch::matmul(whole.t(),rawForce.detach())*state.energyScale)
                               .to(torch::kCPU,torch::kFloat64).contiguous();
    }
    const auto hostForce = (rawForce.detach()*state.energyScale*10.0)
                                   .to(torch::kCPU,torch::kFloat64).contiguous();
    const double hostEnergy = energy.detach().to(torch::kCPU,torch::kFloat64).item<double>()*state.energyScale;
    if (!std::isfinite(hostEnergy) || !torch::isfinite(hostForce).all().item<bool>()
        || !torch::isfinite(tensorVirial).all().item<bool>())
    { GMX_THROW(InternalError("REANN returned nonfinite energy, forces or virial")); }
    // Validate target precision before touching any engine output.
    if (!std::isfinite(static_cast<real>(hostEnergy)))
    { GMX_THROW(InternalError("REANN energy exceeds GROMACS real precision")); }
    const auto f = hostForce.accessor<double,2>();
    const auto v = tensorVirial.accessor<double,2>();
    for (int64_t i = 0; i < n; ++i)
    {
        for (int d = 0; d < DIM; ++d)
        {
            if (!std::isfinite(static_cast<real>(f[i][d])))
            { GMX_THROW(InternalError("REANN force exceeds GROMACS real precision")); }
        }
    }
    for (int i = 0; i < DIM; ++i)
    {
        for (int j = 0; j < DIM; ++j)
        {
            if (!std::isfinite(static_cast<real>(v[i][j])))
            { GMX_THROW(InternalError("REANN virial exceeds GROMACS real precision")); }
        }
    }
    enerd->term[InteractionFunction::NeuralNetworkPotentialEnergy] = hostEnergy;
    for (int64_t i = 0; i < n; ++i)
    {
        for (int d = 0; d < DIM; ++d) { forces[indexLookup[i]][d] += f[i][d]; }
    }
    for (int i = 0; i < DIM; ++i)
    {
        for (int j = 0; j < DIM; ++j) { virial_[i][j] = v[i][j]; }
    }
}
} // namespace gmx
// GMX_REANN_PES NEW FILE END
