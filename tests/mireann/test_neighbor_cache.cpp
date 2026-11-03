// SPDX-License-Identifier: LGPL-2.1-or-later
// GMX_MIREANN_BUFFERED_LIST property checks against full current-frame searches.
#include "mireann_neighbor_cache.h"
#include <cassert>
#include <iostream>
#include <random>
#include <set>
using gmx::dynamiccharge::MireannNeighborCache;
using Vec=std::array<double,3>;
int main()
{
    std::mt19937 generator(72484);
    std::uniform_real_distribution<double> uniform(0.,9.);
    std::normal_distribution<double> motion(0.,.001);
    std::vector<Vec> x(800);
    for(auto& atom:x) for(auto& value:atom){value=uniform(generator);}
    x[0]={.01,4.,4.}; // a center moving through a periodic boundary
    std::vector<unsigned char> qm(x.size(),false);qm[0]=true;
    MireannNeighborCache::Box box{{{9.,0.,0.},{0.,9.,0.},{0.,0.,9.}}};
    auto distance=[&box](const Vec& a,const Vec& b)
    {
        double sum=0;
        for(int d=0;d<3;++d)
        { double v=a[d]-b[d];v-=std::round(v/box[d][d])*box[d][d];sum+=v*v; }
        return std::sqrt(sum);
    };
    MireannNeighborCache cache;
    int frames=0;
    auto check=[&](int step,bool md,int interval)
    {
        if(cache.needsRebuild(step,md,interval,2.5,3.,0,x,qm,box,distance))
        {
            auto& candidate=cache.candidates();candidate.clear();
            for(int atom=1;atom<int(x.size());++atom)
            { if(distance(x[atom],x[0])<=3.){candidate.push_back(atom);} }
            cache.commit(step,x,box);
        }
        std::set<int> filtered,full;
        for(int atom:cache.candidates())if(distance(x[atom],x[0])<=2.5){filtered.insert(atom);}
        for(int atom=1;atom<int(x.size());++atom)
        { if(distance(x[atom],x[0])<=2.5){full.insert(atom);} }
        assert(filtered==full);++frames;
    };
    for(int step=0;step<200;++step)
    {
        for(auto& atom:x)for(auto& value:atom){value+=motion(generator);}
        x[0][0]-=.003;
        for(auto& atom:x)for(int d=0;d<3;++d)
        { atom[d]-=std::floor(atom[d]/box[d][d])*box[d][d]; }
        check(step,step%20==0,0);
    }
    assert(cache.rebuilds()==10);
    // A previously excluded particle crosses the full skin before the schedule:
    x[1]={x[0][0]+3.1,x[0][1],x[0][2]};check(200,true,20);
    const auto before=cache.rebuilds();
    x[1][0]-=.7;check(201,false,20);assert(cache.rebuilds()==before+1);
    // Simultaneous center/particle motion must use the SUM, not either alone.
    x[1]={x[0][0]+3.1,x[0][1],x[0][2]};check(220,true,20);
    const auto beforePair=cache.rebuilds();
    x[0][0]+=.31;x[1][0]-=.31;check(221,false,20);
    assert(cache.rebuilds()==beforePair+1);
    const auto beforeBox=cache.rebuilds();box[0][0]=9.1;check(222,false,20);
    assert(cache.rebuilds()==beforeBox+1);
    const auto beforeReset=cache.rebuilds();check(0,false,20);
    assert(cache.rebuilds()==beforeReset+1);
    std::cout<<"PASS "<<frames<<" current-frame comparisons, periodic crossing, skin exhaustion, "
             <<"combined displacement, changed cell and reset step\n";
}
