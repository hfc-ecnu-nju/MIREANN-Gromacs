/* GMX_MIREANN_RESIDUE_SWITCH (2026-10-26), LGPL-2.1-or-later.
 * Experimental whole-residue, C2 switching; lengths in nm. No model weights
 * or physical MM charges are changed. This only weights model background and
 * its matching reference; the caller must add the complete weight-chain force.
 */
#ifndef GMX_MDLIB_MIREANN_RESIDUE_SWITCH_H
#define GMX_MDLIB_MIREANN_RESIDUE_SWITCH_H
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace gmx::dynamiccharge
{
struct ResidueSwitchResult
{
    double weight=1;
    std::vector<std::array<double,3>> gradientsPerNm;
};
inline ResidueSwitchResult evaluateResidueSwitch(
        const std::vector<std::array<double,3>>& imagesNm,double outerNm,double widthNm)
{
    if(imagesNm.empty() || !std::isfinite(outerNm) || !std::isfinite(widthNm)
       || widthNm<=0 || widthNm>=outerNm)
    { throw std::runtime_error("Invalid MIREANN residue switch geometry"); }
    const size_t n=imagesNm.size();
    ResidueSwitchResult out;out.gradientsPerNm.assign(n,{0,0,0});
    std::vector<double> radial(n),s(n),dsdr(n),prefix(n+1,1),suffix(n+1,1);
    double logarithm=0;
    for(size_t i=0;i<n;++i)
    {
        const auto& x=imagesNm[i];radial[i]=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
        if(!std::isfinite(radial[i])){throw std::runtime_error("Nonfinite residue switch coordinate");}
        // If any atom is in the inner sphere the COMPLETE residue has weight 1.
        if(radial[i]<=outerNm-widthNm){return out;}
        if(radial[i]>=outerNm){s[i]=0;dsdr[i]=0;continue;}
        const double u=(outerNm-radial[i])/widthNm;
        s[i]=std::clamp(u*u*u*(10-15*u+6*u*u),0.0,1.0);
        dsdr[i]=-30*u*u*(1-u)*(1-u)/widthNm;
        if(s[i]==1){return out;}
        logarithm+=std::log1p(-s[i]);
    }
    // Smooth union of atom spheres: w_res=1-product_i(1-s_i). A nearest-atom
    // min() would introduce a derivative kink when the nearest atom changes.
    out.weight=std::clamp(-std::expm1(logarithm),0.0,1.0);
    for(size_t i=0;i<n;++i){prefix[i+1]=prefix[i]*(1-s[i]);}
    for(size_t i=n;i>0;--i){suffix[i-1]=suffix[i]*(1-s[i-1]);}
    for(size_t i=0;i<n;++i)
    {
        const double coefficient=prefix[i]*suffix[i+1]*dsdr[i]/radial[i];
        for(int d=0;d<3;++d){out.gradientsPerNm[i][d]=coefficient*imagesNm[i][d];}
    }
    return out;
}
} // namespace gmx::dynamiccharge
#endif
