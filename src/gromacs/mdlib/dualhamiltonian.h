// SPDX-License-Identifier: LGPL-2.1-or-later
/* GMX_DUAL_HAMILTONIAN: native MM driver + unchanged ML/MM force endpoint. */
#ifndef GMX_MDLIB_DUALHAMILTONIAN_H
#define GMX_MDLIB_DUALHAMILTONIAN_H
#include <memory>
#include <cstdint>
#include <string>
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/vectypes.h"
struct t_inputrec;struct t_mdatoms;struct gmx_enerdata_t;struct t_trxframe;struct t_forcerec;struct t_commrec;
namespace gmx
{
class IMDModule;
namespace dualhamiltonian
{
struct Options{bool active=false;double s=1;std::string socket,energyLog="dual-energy.csv";};
class Client;class Worker;
Options readOptions(const t_inputrec& ir);
std::unique_ptr<IMDModule> makeModule();
bool workerRequested();
std::shared_ptr<Worker> makeWorker(const t_mdatoms& atoms,const t_commrec& cr);
bool nextFrame(Worker& worker,t_trxframe* frame,int natoms);
double frameTime(const Worker& worker);
void completeForce(t_forcerec& fr,const t_inputrec& ir,const t_mdatoms& atoms,const t_commrec& cr,
                   ArrayRef<const RVec> coordinates,ArrayRef<RVec> forces,
                   const matrix box,tensor virial,gmx_enerdata_t* energy,
                   std::int64_t step,double time,bool computeVirial);
}
}
#endif
