// Optional in-process C++/LibTorch QEq-REANN backend (Linux/WSL).
#include "dynamicchargeprovider.h"
#include "qeq_reann_abi.h"
// GMX_MIREANN: separate seven-input backend, same validated charge commit.
#include "mireann.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif
namespace gmx::dynamiccharge
{
#if !defined(_WIN32)
namespace
{
// GMX_QEQ_REANN: locate the backend beside libgromacs, in build AND install trees.
// No model/library environment variables are used by the MDP interface.
void backendAnchor() {}
std::string builtBackendPath()
{
#if defined(GMX_QEQ_REANN_BUILT)
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&backendAnchor), &info) || !info.dli_fname)
    { throw std::runtime_error("Cannot locate the integrated QEq backend beside libgromacs"); }
    return (std::filesystem::path(info.dli_fname).parent_path() / GMX_QEQ_BACKEND_FILENAME).string();
#else
    throw std::runtime_error("QEq support was not compiled; configure -DGMX_QEQ_REANN=ON");
#endif
}
const char* required(const char* key)
{
    const char* value = std::getenv(key);
    if (!value || !*value) { throw std::runtime_error(std::string("Missing ") + key); }
    return value;
}
struct Library
{
    // LibTorch distributions can use the old libstdc++ ABI while GROMACS uses
    // the new ABI. Some std::regex templates have identical mangled names but
    // different layouts. Resolve plugin dependencies before global GROMACS
    // symbols; the only boundary between the two libraries is our C ABI.
    static int flags()
    {
        int result = RTLD_NOW | RTLD_LOCAL;
#if defined(RTLD_DEEPBIND)
        result |= RTLD_DEEPBIND;
#endif
        // CUDA/LibTorch register process-lifetime hooks and thread-local
        // destructors. Unloading their code after releasing a model can crash
        // those hooks at process exit. Keep the library mapped, but free every
        // model/tensor normally through gmx_qeq_destroy.
#if defined(RTLD_NODELETE)
        result |= RTLD_NODELETE;
#endif
        return result;
    }
    explicit Library(const char* path) : pointer(dlopen(path, flags()))
    {
        if (!pointer) { throw std::runtime_error(std::string("Cannot load QEq library: ") + dlerror()); }
    }
    ~Library() { dlclose(pointer); }
    template<typename T> T symbol(const char* name)
    {
        dlerror();
        void* value = dlsym(pointer, name);
        if (const char* error = dlerror()) { throw std::runtime_error(error); }
        return reinterpret_cast<T>(value);
    }
    void* pointer;
};
class QeqPredictor final : public ChargePredictor
{
public:
    explicit QeqPredictor(const Frame& frame, const QeqOptions* configured = nullptr) :
        library_(configured ? builtBackendPath().c_str() : required("GMX_DYNAMIC_CHARGE_QEQ_LIBRARY")),
        evaluate_(library_.symbol<decltype(&gmx_qeq_predict)>("gmx_qeq_predict")),
        model_(nullptr, library_.symbol<decltype(&gmx_qeq_destroy)>("gmx_qeq_destroy"))
    {
        if (library_.symbol<decltype(&gmx_qeq_abi_version)>("gmx_qeq_abi_version")() != 1)
        { throw std::runtime_error("Unsupported QEq plugin ABI"); }
        std::stringstream elements;
        if (configured)
        {
            std::istringstream labels(configured->elements);
            std::string symbol;
            size_t i = 0;
            elements.precision(17);
            while (labels >> symbol)
            {
                if (i >= frame.region.size()) { throw std::runtime_error("Too many QEq elements"); }
                elements << symbol << ' ' << frame.massesU.at(frame.region[i++]) << '\n';
            }
        }
        else
        {
            // Retained legacy interface for old numerical regression tests.
            std::ifstream input(required("GMX_DYNAMIC_CHARGE_ELEMENTS"));
            if (!input) { throw std::runtime_error("Cannot open QEq element file"); }
            elements << input.rdbuf();
        }
        const std::map<std::string, double> massTable = {
            {"H",1.008},{"C",12.011},{"N",14.007},{"O",15.999},{"F",18.998403},
            {"P",30.973762},{"S",32.06},{"Cl",35.45},{"Br",79.904},{"I",126.90447},
            {"Na",22.989769},{"Mg",24.305},{"Ca",40.078},{"Si",28.085}};
        std::string line, labels;
        std::vector<double> masses;
        while (std::getline(elements, line))
        {
            std::istringstream row(line.substr(0, line.find('#')));
            std::string symbol, value, extra;
            if (!(row >> symbol)) { continue; }
            double mass;
            if (row >> value)
            {
                size_t used = 0;
                mass = std::stod(value, &used);
                if (used != value.size() || (row >> extra))
                { throw std::runtime_error("QEq elements require one 'symbol [mass_u]' per line"); }
            }
            else
            {
                const auto found = massTable.find(symbol);
                if (found == massTable.end()) { throw std::runtime_error("Supply mass_u for " + symbol); }
                mass = found->second;
            }
            if (!std::isfinite(mass) || mass <= 0) { throw std::runtime_error("Invalid QEq mass_u"); }
            labels += symbol + " ";
            masses.push_back(mass);
        }
        if (masses.size() != frame.region.size())
        { throw std::runtime_error("QEq element count must match region order and size"); }
        GmxQeqOptions options{};
        options.abiVersion = 1;
        options.numAtoms = static_cast<int>(masses.size());
        options.modelPath = configured ? configured->model.c_str() : required("GMX_DYNAMIC_CHARGE_MODEL");
        options.device = configured ? configured->device.c_str() : std::getenv("GMX_DYNAMIC_CHARGE_DEVICE");
        if (!options.device) { options.device = "cpu"; }
        options.elements = labels.c_str();
        options.massesU = masses.data();
        options.torchThreads = configured ? configured->threads : 1;
        if (const char* text = configured ? nullptr : std::getenv("GMX_DYNAMIC_CHARGE_TORCH_THREADS"))
        {
            size_t used = 0;
            options.torchThreads = std::stoi(text, &used);
            if (used != std::string(text).size() || options.torchThreads < 1)
            { throw std::runtime_error("Invalid QEq Torch thread count"); }
        }
        if (const char* text = configured ? configured->pbc.c_str() : std::getenv("GMX_DYNAMIC_CHARGE_MODEL_PBC"))
        {
            std::istringstream values(text);
            std::string extra;
            for (int& flag : options.pbc)
            {
                if (!(values >> flag) || (flag != 0 && flag != 1))
                { throw std::runtime_error("Model PBC requires three 0/1 flags"); }
            }
            if (values >> extra) { throw std::runtime_error("Too many model PBC flags"); }
        }
        char error[4096]{};
        double total = 0;
        model_.reset(library_.symbol<decltype(&gmx_qeq_create)>("gmx_qeq_create")(
                &options, &total, error, sizeof(error)));
        if (!model_) { throw std::runtime_error(error); }
        double initial = 0;
        for (int id : frame.region) { initial += frame.chargesE[id]; }
        if (!std::isfinite(total) || std::abs(total - initial) > 1e-5)
        { throw std::runtime_error("Topology region charge differs from QEq totcharge; initialize it consistently (no implicit projection)"); }
        xyz_.resize(masses.size() * 3);
        const char* profilePath = configured
                ? (configured->profile.empty() ? nullptr : configured->profile.c_str())
                : std::getenv("GMX_DYNAMIC_CHARGE_PROFILE");
        if (const char* path = profilePath)
        {
            profile_.open(path);
            if (!profile_) { throw std::runtime_error("Cannot open QEq profile output"); }
        }
        std::fprintf(stderr, "QEq-REANN C++ backend: device=%s, atoms=%d, model_total=%.9g e; input nm->Angstrom, output q(e)\n",
                     options.device, options.numAtoms, total);
    }
    ~QeqPredictor() override
    {
        if (profile_.is_open())
        {
            profile_.precision(17);
            profile_ << "calls,warmup_calls,measured_calls,mean_model_ms,mean_bridge_ms\n"
                     << calls_ << ',' << std::min(calls_, 20LL) << ',' << measured_ << ','
                     << (measured_ ? modelMs_/measured_ : 0) << ','
                     << (measured_ ? bridgeMs_/measured_ : 0) << '\n';
        }
    }
    std::vector<double> predict(const Frame& frame) override
    {
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < frame.region.size(); ++i)
        {
            const auto& position = frame.positionsNm[frame.region[i]];
            for (int d = 0; d < 3; ++d) { xyz_[i*3+d] = position[d] * 10.0; }
        }
        double cell[9];
        for (int i = 0; i < 3; ++i)
        { for (int j = 0; j < 3; ++j) { cell[3*i+j] = frame.boxNm[i][j] * 10.0; } }
        const double det = cell[0]*(cell[4]*cell[8]-cell[5]*cell[7])
                         -cell[1]*(cell[3]*cell[8]-cell[5]*cell[6])
                         +cell[2]*(cell[3]*cell[7]-cell[4]*cell[6]);
        if (!std::isfinite(det) || std::abs(det) < 1e-10)
        { throw std::runtime_error("QEq needs an invertible model cell even for a nonperiodic cluster"); }
        std::vector<double> charges(frame.region.size());
        double multiplier = 0, milliseconds = 0;
        char error[4096]{};
        if (evaluate_(model_.get(), xyz_.data(), cell, charges.data(), &multiplier,
                      &milliseconds, error, sizeof(error)))
        { throw std::runtime_error(error); }
        ++calls_;
        if (calls_ > 20)
        {
            ++measured_;
            modelMs_ += milliseconds;
            bridgeMs_ += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        return charges;
    }
private:
    Library library_;
    decltype(&gmx_qeq_predict) evaluate_;
    std::unique_ptr<void, decltype(&gmx_qeq_destroy)> model_;
    std::vector<double> xyz_;
    std::ofstream profile_;
    long long calls_ = 0, measured_ = 0;
    double modelMs_ = 0, bridgeMs_ = 0;
};
}
#endif
std::unique_ptr<ChargePredictor> makeQeqReannPredictor(const Frame& frame)
{
#if !defined(_WIN32)
    return std::make_unique<QeqPredictor>(frame);
#else
    throw std::runtime_error("The QEq shared-library backend currently requires Linux/WSL");
#endif
}
std::unique_ptr<ChargePredictor> makeQeqReannPredictor(const Frame& frame, const QeqOptions& options)
{
#if !defined(_WIN32)
    // GMX_MIREANN: the seven-input model cannot use the gas-phase QEq ABI.
    if (options.modelType=="mireann") { return makeMireannPredictor(frame,options,builtBackendPath()); }
    return std::make_unique<QeqPredictor>(frame, &options);
#else
    throw std::runtime_error("The QEq shared-library backend currently requires Linux/WSL");
#endif
}
}
