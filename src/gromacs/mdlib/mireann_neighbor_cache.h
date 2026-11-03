/* GMX_MIREANN_BUFFERED_LIST (2026-10-26), LGPL-2.1-or-later.
 * Candidate-list validity only. Native AnalysisNeighborhood builds the list;
 * native pbc_dx supplies distances. No cached images or cached 25-A membership.
 */
#ifndef GMX_MDLIB_MIREANN_NEIGHBOR_CACHE_H
#define GMX_MDLIB_MIREANN_NEIGHBOR_CACHE_H
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace gmx::dynamiccharge
{
class MireannNeighborCache
{
public:
    using Box = std::array<std::array<double,3>,3>;

    template<class Positions, class QmMask, class Distance>
    bool needsRebuild(std::int64_t step, bool mdNeighborSearch, int interval,
                      double cutoffNm, double listCutoffNm, int center,
                      const Positions& current, const QmMask& qmMask,
                      const Box& box, Distance distance) const
    {
        if (!valid_ || current.size()!=reference_.size() || box!=box_
            || step<buildStep_ || listCutoffNm<=cutoffNm)
        { return true; }
        // interval=0 follows the actual native MD neighbor-search flag, not a
        // guessed modulo of the step number (important for restart/nstlist).
        if ((interval==0 && mdNeighborSearch)
            || (interval>0 && step-buildStep_>=interval))
        { return true; }
        const double skin=listCutoffNm-cutoffNm;
        double lengthScale=std::max(1.,listCutoffNm);
        for(const auto& row:box) for(double value:row)
        { lengthScale=std::max(lengthScale,std::abs(value)); }
        const double margin=64.*std::numeric_limits<float>::epsilon()*lengthScale;
        if(skin<=margin){return true;}
        const double centerMove=distance(current[center],reference_[center]);
        double maxMmMove=0;
        for(std::size_t atom=0;atom<current.size();++atom)
        {
            if(!qmMask[atom])
            {
                maxMmMove=std::max(maxMmMove,distance(current[atom],reference_[atom]));
                // Triangle inequality on the periodic cell: a pair initially
                // outside r_list cannot reach r_cut while this sum < skin.
                // Include a conservative margin for native real arithmetic.
                if(maxMmMove+centerMove>=skin-margin)
                { return true; }
            }
        }
        return false;
    }

    template<class Positions>
    void commit(std::int64_t step, const Positions& current, const Box& box)
    {
        reference_.resize(current.size());
        for(std::size_t i=0;i<current.size();++i) for(int d=0;d<3;++d)
        { reference_[i][d]=current[i][d]; }
        box_=box;buildStep_=step;valid_=true;++rebuilds_;
    }

    std::vector<int>& candidates() { return candidates_; }
    std::uint64_t rebuilds() const { return rebuilds_; }
private:
    bool valid_=false;
    std::int64_t buildStep_=0;
    Box box_{};
    std::vector<std::array<double,3>> reference_;
    std::vector<int> candidates_;
    std::uint64_t rebuilds_=0;
};
} // namespace gmx::dynamiccharge
#endif
