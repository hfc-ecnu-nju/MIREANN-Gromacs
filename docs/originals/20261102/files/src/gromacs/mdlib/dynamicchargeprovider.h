// Local extension, LGPL-2.1-or-later, as for GROMACS.
#ifndef GMX_MDLIB_DYNAMICCHARGEPROVIDER_H
#define GMX_MDLIB_DYNAMICCHARGEPROVIDER_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "dynamicchargeoptions.h"

namespace gmx::dynamiccharge
{
// Global topology order, zero-based indices; nm, ps and elementary charge.
// Positions are periodic images: a model must implement PBC using boxNm.
struct Frame
{
    std::int64_t step;
    double timePs;
    std::array<std::array<double, 3>, 3> boxNm;
    std::vector<std::array<double, 3>> positionsNm;
    std::vector<double> chargesE;
    std::vector<double> massesU; // GMX_QEQ_REANN: topology masses (u), not model guesses.
    std::vector<int> atomTypes; // force-field type indices, NOT atomic numbers
    std::vector<int> region;
};

class ChargePredictor
{
public:
    virtual ~ChargePredictor() = default;
    virtual std::vector<double> predict(const Frame& frame) = 0;
};
std::unique_ptr<ChargePredictor> makeQeqReannPredictor(const Frame& frame);
std::unique_ptr<ChargePredictor> makeQeqReannPredictor(const Frame& frame, const QeqOptions& options);

// Per-simulation storage, owned by t_forcerec (never process-global/static).
// Mapping and buffers survive force calls; atom reordering invalidates them.
struct RuntimeState;

inline std::vector<int> readRegion(const std::string& path)
{
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("Cannot open dynamic charge region: " + path); }
    std::vector<int> region;
    std::set<int> seen;
    std::string line;
    while (std::getline(in, line))
    {
        line = line.substr(0, line.find('#'));
        std::istringstream words(line);
        std::string word;
        while (words >> word)
        {
            std::size_t used = 0;
            const long long id = std::stoll(word, &used);
            if (used != word.size() || id <= 0 || id > std::numeric_limits<int>::max()
                || !seen.insert(static_cast<int>(id - 1)).second)
            { throw std::runtime_error("Region requires unique positive 1-based atom numbers"); }
            region.push_back(static_cast<int>(id - 1));
        }
    }
    if (region.empty()) { throw std::runtime_error("Dynamic charge region is empty"); }
    return region;
}

struct RuntimeState
{
    RuntimeState(const char* regionFile, const char* tableDirectory, const char* auditPath)
        : RuntimeState(readRegion(regionFile), tableDirectory, auditPath)
    {
    }
    RuntimeState(std::vector<int> region, const char* tableDirectory, const char* auditPath)
    {
        frame.region = std::move(region);
        if (tableDirectory)
        {
            useTables = true;
            tables = tableDirectory;
        }
        if (auditPath)
        {
            audit.open(auditPath, std::ios::app);
            if (!audit) { throw std::runtime_error("Cannot open dynamic charge audit log"); }
            audit.precision(17);
        }
    }

    template<typename GlobalIndex>
    void rebuildMapping(int numLocal, std::int64_t partition, GlobalIndex globalIndex)
    {
        // With one rank, global physical indices must fit in the local atom
        // count; the latter may additionally include DD filler particles.
        globalToLocal.assign(numLocal, -1);
        int numGlobal = 0;
        for (int local = 0; local < numLocal; ++local)
        {
            const int global = globalIndex(local);
            if (global < 0) { continue; }
            if (global >= numLocal || globalToLocal[global] != -1)
            { throw std::runtime_error("Invalid single-rank global atom mapping"); }
            globalToLocal[global] = local;
            numGlobal = std::max(numGlobal, global + 1);
        }
        globalToLocal.resize(numGlobal);
        for (int local : globalToLocal)
        {
            if (local < 0) { throw std::runtime_error("Incomplete single-rank global atom mapping"); }
        }
        regionLocal.resize(frame.region.size());
        outsideRegion.assign(numGlobal, true);
        for (std::size_t i = 0; i < frame.region.size(); ++i)
        {
            const int id = frame.region[i];
            if (id >= numGlobal)
            { throw std::runtime_error("Dynamic charge region atom is outside topology"); }
            regionLocal[i] = globalToLocal[id];
            outsideRegion[id] = false;
        }
        frame.positionsNm.resize(numGlobal);
        frame.chargesE.resize(numGlobal);
        frame.massesU.resize(numGlobal);
        frame.atomTypes.resize(numGlobal);
        localCount = numLocal;
        partitionCount = partition;
        initialized = true;
        // GMX_MIREANN_GPU_AUDIT: physical exclusions are cached in global order.
        // Refresh after any local mapping rebuild; the GPU uploads once/revision.
        mireannExclusions.clear();
        ++mappingRevision;
    }

    Frame frame{};
    std::vector<int> globalToLocal;
    std::vector<int> regionLocal;
    std::vector<unsigned char> outsideRegion;
    int localCount = -1;
    std::int64_t partitionCount = -1;
    bool initialized = false;
    std::vector<unsigned char> mireannExclusions;
    std::uint64_t mappingRevision=0;
    double outsideCharge = 0;
    double outsideChargeSquared = 0;
    bool useTables = false;
    std::string tables;
    std::ofstream audit;
    std::unique_ptr<ChargePredictor> model;
};

// ML integration point. Return one charge per frame.region, in that order.
// Default is identity (plumbing only). Replace this function with inference,
// or pass a table directory for deterministic per-step integration tests.
// Keep prediction stateless with respect to MD history for reproducible restarts.
inline std::vector<double> predictCharges(const Frame& frame, const char* tableDirectory)
{
    std::vector<double> result;
    if (tableDirectory)
    {
        const std::string path = std::string(tableDirectory) + "/"
                                 + std::to_string(frame.step) + ".charges";
        std::ifstream in(path);
        if (!in) { throw std::runtime_error("Missing charge prediction table: " + path); }
        std::string token;
        while (in >> token)
        {
            std::size_t used = 0;
            double value = std::stod(token, &used);
            if (used != token.size()) { throw std::runtime_error("Invalid charge table value"); }
            result.push_back(value);
        }
    }
    else
    {
        for (int id : frame.region) { result.push_back(frame.chargesE.at(id)); }
    }
    return result;
}

inline void validatePrediction(const Frame& frame, const std::vector<double>& prediction)
{
    if (prediction.size() != frame.region.size())
    { throw std::runtime_error("Prediction size does not match region size"); }
    double oldSum = 0, newSum = 0;
    for (std::size_t i = 0; i < prediction.size(); ++i)
    {
        if (!std::isfinite(prediction[i]))
        { throw std::runtime_error("Nonfinite predicted charge"); }
        oldSum += frame.chargesE.at(frame.region[i]);
        newSum += prediction[i];
    }
    if (!std::isfinite(oldSum) || !std::isfinite(newSum) || std::abs(newSum - oldSum) > 1e-5)
    { throw std::runtime_error("Predicted region charge must preserve its total charge (1e-5 e)"); }
}
} // namespace gmx::dynamiccharge
#endif
