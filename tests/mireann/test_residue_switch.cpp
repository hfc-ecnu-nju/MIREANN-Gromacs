// SPDX-License-Identifier: LGPL-2.1-or-later
#include "mireann_residue_switch.h"
#include <cassert>
#include <iostream>
#include <random>
using namespace gmx::dynamiccharge;
int main()
{
    using Vec=std::array<double,3>;
    double maximum=0;int comparisons=0;
    std::mt19937 engine(421);std::uniform_real_distribution<double> delta(-.12,.12);
    for(int sample=0;sample<100;++sample)
    {
        std::vector<Vec> x{{2.40+delta(engine),delta(engine),delta(engine)},
                           {2.46+delta(engine),.04+delta(engine),delta(engine)},
                           {2.52+delta(engine),delta(engine),.08+delta(engine)}};
        const auto a=evaluateResidueSwitch(x,2.5,.2);
        assert(a.weight>=0 && a.weight<=1);
        for(size_t i=0;i<x.size();++i)for(int d=0;d<3;++d)
        {
            auto plus=x,minus=x;const double h=1e-7;plus[i][d]+=h;minus[i][d]-=h;
            const double fd=(evaluateResidueSwitch(plus,2.5,.2).weight-evaluateResidueSwitch(minus,2.5,.2).weight)/(2*h);
            maximum=std::max(maximum,std::abs(fd-a.gradientsPerNm[i][d]));++comparisons;
        }
    }
    assert(evaluateResidueSwitch({Vec{2.29,0,0},Vec{2.6,0,0}},2.5,.2).weight==1);
    assert(evaluateResidueSwitch({Vec{2.5,0,0},Vec{2.6,0,0}},2.5,.2).weight==0);
    assert(maximum<2e-7);
    std::cout<<"PASS comparisons="<<comparisons<<" max_weight_gradient_error_per_nm="<<maximum<<'\n';
}
