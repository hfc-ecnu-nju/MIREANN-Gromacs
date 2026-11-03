/* GMX_MIREANN NEW FILE: electrostatic replacement after charge commit and PME.
 * Local extension, LGPL-2.1-or-later.
 */
#ifndef GMX_MDLIB_MIREANN_H
#define GMX_MDLIB_MIREANN_H
#include "dynamicchargeprovider.h"
#include "mireann_classical_abi.h" // GMX_MIREANN_GPU_AUDIT
struct t_inputrec;
struct t_forcerec;
struct gmx_localtop_t;
struct gmx_enerdata_t;
namespace gmx { class ForceWithVirial; }
namespace gmx::dynamiccharge
{
struct MireannEnvironment
{
    std::vector<int> mmGlobal; // global atom IDs, one nearest image per MM atom
    std::vector<std::array<double,3>> qmNm, mmNm; // relative to specified QM center
    std::vector<double> mmChargesE, weights, weightDerivativePerNm;
    double modelEnergyKj=0;
    int centerRegionIndex=0;
};
struct MireannForces
{
    std::vector<std::array<double,3>> qm, mm; // kJ/(mol nm)
    std::vector<double> backgroundDerivative; // kJ/(mol e)
};
class MireannPredictor : public ChargePredictor
{
public:
    virtual const MireannEnvironment& environment() const=0;
    virtual MireannForces forces(const std::vector<double>& potentialKjPerE)=0;
    virtual void record(const Frame&,double bareEnergyKj)=0;
    virtual bool classical(const GmxMireannClassicalInput&,GmxMireannClassicalOutput&)=0;
};
std::unique_ptr<ChargePredictor> makeMireannPredictor(const Frame&,const QeqOptions&,const std::string& backend);
void applyMireannCorrection(RuntimeState&,const t_inputrec&,const gmx_localtop_t&,
                            t_forcerec&,gmx::ForceWithVirial&,gmx_enerdata_t*,bool computeVirial);
} // namespace gmx::dynamiccharge
#endif
