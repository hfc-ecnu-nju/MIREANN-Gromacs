// SPDX-License-Identifier: LGPL-2.1-or-later
/* GMX_DUAL_HAMILTONIAN: fixed-s full-endpoint interpolation, LGPL-2.1-or-later.
 * No charge interpolation. The two native endpoint calculations own independent
 * topology/PME/model state; these buffers retain both physical charge arrays.
 */
#ifndef GMX_MDLIB_DUALHAMILTONIAN_CORE_H
#define GMX_MDLIB_DUALHAMILTONIAN_CORE_H
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace gmx::dualhamiltonian
{
inline void require(bool ok,const std::string& message)
{ if(!ok){throw std::runtime_error("Dual Hamiltonian: "+message);} }
inline void finite(const std::vector<double>& a)
{ for(double x:a){require(std::isfinite(x),"nonfinite endpoint data");} }
struct Endpoint
{
    std::vector<double> forces,charges,terms,pairs;
    std::array<double,9> virial{};
    void validate(std::size_t n) const
    {
        require(forces.size()==3*n && charges.size()==n,"endpoint atom count mismatch");
        finite(forces);finite(charges);finite(terms);finite(pairs);
        for(double v:virial){require(std::isfinite(v),"nonfinite endpoint virial");}
    }
};
class ChargeBuffers
{
public:
    void commit(const std::vector<double>& rawMM,const std::vector<double>& predictedMLMM)
    {
        finite(rawMM);finite(predictedMLMM);
        require(!rawMM.empty() && rawMM.size()==predictedMLMM.size(),"charge array sizes differ");
        if(qMM_.empty()){qMM_=rawMM;}
        else{require(qMM_==rawMM,"MM endpoint charges changed after initialization");}
        qMLMM_=predictedMLMM;
        // Keep separate sums and squares: each PME endpoint uses its own values.
        sumMM_=sumMLMM_=squareMM_=squareMLMM_=0;
        for(std::size_t i=0;i<qMM_.size();++i)
        {
            sumMM_+=qMM_[i];sumMLMM_+=qMLMM_[i];
            squareMM_+=qMM_[i]*qMM_[i];squareMLMM_+=qMLMM_[i]*qMLMM_[i];
        }
    }
    const std::vector<double>& qMM() const{return qMM_;}
    const std::vector<double>& qMLMM() const{return qMLMM_;}
    double sumMM() const{return sumMM_;} double sumMLMM() const{return sumMLMM_;}
    double squareMM() const{return squareMM_;} double squareMLMM() const{return squareMLMM_;}
private:
    std::vector<double> qMM_,qMLMM_;
    double sumMM_=0,sumMLMM_=0,squareMM_=0,squareMLMM_=0;
};
inline double mix(double s,double mm,double ml)
{
    require(std::isfinite(s)&&s>=0&&s<=1,"s must be finite and in [0,1]");
    require(std::isfinite(mm)&&std::isfinite(ml),"nonfinite endpoint observable");
    // Exact endpoint recovery, without cancellation from an unrelated endpoint.
    return s==0 ? mm : s==1 ? ml : (1-s)*mm+s*ml;
}
inline void mixInPlace(double s,Endpoint& mm,const Endpoint& ml)
{
    mm.validate(mm.charges.size());ml.validate(mm.charges.size());
    require(mm.terms.size()==ml.terms.size()&&mm.pairs.size()==ml.pairs.size(),"energy layouts differ");
    for(std::size_t i=0;i<mm.forces.size();++i){mm.forces[i]=mix(s,mm.forces[i],ml.forces[i]);}
    for(std::size_t i=0;i<mm.terms.size();++i){mm.terms[i]=mix(s,mm.terms[i],ml.terms[i]);}
    for(std::size_t i=0;i<mm.pairs.size();++i){mm.pairs[i]=mix(s,mm.pairs[i],ml.pairs[i]);}
    for(std::size_t i=0;i<9;++i){mm.virial[i]=mix(s,mm.virial[i],ml.virial[i]);}
    // Deliberately leave mm.charges unchanged: no "mixed charge" exists.
}
struct Request
{
    std::int64_t step=0;double time=0;
    std::array<double,9> box{};
    std::vector<double> coordinates,rawCharges,masses;
};
}
#endif
