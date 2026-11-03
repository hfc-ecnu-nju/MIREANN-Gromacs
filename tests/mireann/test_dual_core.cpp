// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualhamiltonian_socket.h"
#include <cassert>
#include <iostream>
#include <future>
using namespace gmx::dualhamiltonian;
int main()
{
    ChargeBuffers q;q.commit({.2,-.2},{.6,-.6});q.commit({.2,-.2},{.7,-.7});
    assert(q.qMM()[0]==.2&&q.qMLMM()[0]==.7);
    assert(std::abs(q.squareMM()-.08)<1e-15&&std::abs(q.squareMLMM()-.98)<1e-15);
    bool caught=false;try{q.commit({.3,-.3},{.7,-.7});}catch(...){caught=true;}assert(caught);
    Endpoint a,b;a.charges={.2,-.2};b.charges={.7,-.7};
    a.forces={1,2,3,-1,-2,-3};b.forces={-3,6,5,3,-6,-5};
    a.terms={2,3};b.terms={4,8};a.pairs={2};b.pairs={7};
    for(double s:{0.,.25,.5,.75,1.})
    {
        auto result=a;mixInPlace(s,result,b);
        for(std::size_t i=0;i<6;++i){assert(result.forces[i]==mix(s,a.forces[i],b.forces[i]));}
        assert(result.charges==a.charges);
        // Two coordinate-dependent endpoint energies; weighted force must be
        // the gradient of the weighted energy, not a force-only interpolation.
        const double x=.71,h=1e-6;
        auto energy=[s](double r){return mix(s,.5*3*r*r,2*r*r*r);};
        double fd=-(energy(x+h)-energy(x-h))/(2*h);
        assert(std::abs(fd-mix(s,-3*x,-6*x*x))<1e-8);
    }
    caught=false;try{mix(-.1,0,1);}catch(...){caught=true;}assert(caught);
    // Linear charge interpolation does NOT equal linear Hamiltonian mixing.
    assert(std::abs(mix(.5,.2*-.2,.7*-.7)-(.45*-.45))>.01);
#ifndef _WIN32
    const auto path=std::string("/tmp/gmx-dual-unit-")+std::to_string(getpid())+".sock";
    auto worker=std::async(std::launch::async,[&]{
        Channel server(path,true);Request r;
        assert(server.next(&r,2)&&r.step==17&&r.coordinates[0]==.5);
        server.respond(b,r.step,r.time);
        assert(!server.next(&r,2));
    });
    {
        Channel client(path,false);Request r;r.step=17;r.time=.017;r.coordinates={.5,0,0,1,0,0};r.rawCharges=a.charges;r.masses={1,1};
        auto received=client.evaluate(r);assert(received.charges==b.charges&&received.forces==b.forces&&received.terms==b.terms);
    }
    worker.get();assert(access(path.c_str(),F_OK)!=0);
#endif
    std::cout<<"PASS: separate fixed/dynamic charges, exact endpoints, weighted forces/energies, finite differences, IPC framing and EOF.\n";
}
