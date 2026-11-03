// GMX_MIREANN_PHYSICS_AUDIT: cutoff continuity of the combined PME reference.
#include "mirecoulomb.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main()
{
    const double c=138.93545764438198,qi=-.7,qj=.417,rc=1.2,alpha=.8;
    const double shift=std::erfc(alpha*rc)/rc,h=1e-7;
    auto residual=[&](double r,bool corrected)
    {
        const double real=r<rc ? c*qi*qj*(std::erfc(alpha*r)/r-shift) : 0;
        const auto reference=gmx::dynamiccharge::embeddingCoulombPair({r,0,0},qi,qj,c,rc,alpha,corrected ? shift : 0);
        return real-reference.energyKjMol;
    };
    const double oldJump=residual(rc+h,false)-residual(rc-h,false);
    const double newJump=residual(rc+h,true)-residual(rc-h,true);
    if(std::abs(oldJump)<1e-2 || std::abs(newJump)>1e-5){throw std::runtime_error("PME shift continuity regression");}
    for(double r:{.7,1.19,1.21,2.0})
    {
        const auto p=gmx::dynamiccharge::embeddingCoulombPair({r,0,0},qi,qj,c,rc,alpha,shift);
        const double fd=-(gmx::dynamiccharge::embeddingCoulombPair({r+h,0,0},qi,qj,c,rc,alpha,shift).energyKjMol
                        -gmx::dynamiccharge::embeddingCoulombPair({r-h,0,0},qi,qj,c,rc,alpha,shift).energyKjMol)/(2*h);
        if(std::abs(fd-p.forceOnQmKjNm[0])>1e-6){throw std::runtime_error("Shift/Gaussian force derivative");}
    }
    std::cout<<"SHIFT CONTINUITY PASS old_jump="<<oldJump<<" corrected_difference="<<newJump<<'\n';
}
