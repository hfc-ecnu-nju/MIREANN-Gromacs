// GMX_MIREANN: numerical verification of the production subtraction kernel.
// Test the actual production subtraction kernel, including its charge derivative.
#include "mirecoulomb.h"
#include <algorithm>
#include <iostream>
using gmx::dynamiccharge::embeddingCoulombPair;
using Vec = std::array<double,3>;
int main()
{
    constexpr double c=138.93545764438198, qi=-0.7, qj=0.417, h=1e-6, rc=1.0;
    double maximum=0;
    // alpha=0.8 deliberately exposes the Gaussian term: at a usual PME alpha
    // the missing term in the old Amber force can be hidden by its small size.
    for (double alpha : {0.8,3.3}) for (double distance : {0.5,0.99,1.01,1.5,2.0})
    {
        const Vec x{distance,0.12,-0.07};
        auto pair=embeddingCoulombPair(x,qi,qj,c,rc,alpha);
        for (int d=0;d<3;++d)
        {
            auto a=x,b=x; a[d]+=h;b[d]-=h;
            const double fd=-(embeddingCoulombPair(a,qi,qj,c,rc,alpha).energyKjMol
                              -embeddingCoulombPair(b,qi,qj,c,rc,alpha).energyKjMol)/(2*h);
            maximum=std::max(maximum,std::abs(fd-pair.forceOnQmKjNm[d]));
            if (std::abs(fd-pair.forceOnQmKjNm[d])>2e-7)
            { throw std::runtime_error("Gaussian derivative / force sign / units"); }
        }
        const double qfd=(embeddingCoulombPair(x,qi+h,qj,c,rc,alpha).energyKjMol
                         -embeddingCoulombPair(x,qi-h,qj,c,rc,alpha).energyKjMol)/(2*h);
        if (std::abs(qfd-pair.potentialAtQmKjPerE)>2e-7)
        { throw std::runtime_error("Subtraction charge response"); }
    }
    std::cout<<"ERF/GAUSSIAN SUBTRACTION PASS max_force_error="<<maximum<<'\n';
}
