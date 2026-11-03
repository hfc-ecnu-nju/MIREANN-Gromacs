// SPDX-License-Identifier: LGPL-2.1-or-later
/* Same-host local IPC for two native force evaluations; no worker integration.
 * Strict framing, size checks, finite-data checks and bounded blocking. No shell
 * commands, remote access, arbitrary file overwrite or numerical fallback.
 */
#ifndef GMX_MDLIB_DUALHAMILTONIAN_SOCKET_H
#define GMX_MDLIB_DUALHAMILTONIAN_SOCKET_H
#include "dualhamiltonian_core.h"
#include <cerrno>
#include <chrono>
#include <memory>
#include <thread>
#ifndef _WIN32
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#endif
namespace gmx::dualhamiltonian
{
#ifndef _WIN32
class Socket
{
public:
    explicit Socket(int fd=-1):fd_(fd){}
    ~Socket(){if(fd_>=0){::close(fd_);}}
    Socket(const Socket&)=delete;Socket& operator=(const Socket&)=delete;
    int fd() const{return fd_;}
    void reset(int fd){if(fd_>=0){::close(fd_);}fd_=fd;}
    void timeout() const
    {
        timeval t{300,0};
        require(::setsockopt(fd_,SOL_SOCKET,SO_RCVTIMEO,&t,sizeof(t))==0,"cannot set receive timeout");
        require(::setsockopt(fd_,SOL_SOCKET,SO_SNDTIMEO,&t,sizeof(t))==0,"cannot set send timeout");
    }
    void send(const void* data,std::size_t bytes) const
    {
        auto* p=static_cast<const char*>(data);
        while(bytes)
        {
            const auto n=::send(fd_,p,bytes,MSG_NOSIGNAL);
            if(n<0&&errno==EINTR){continue;}
            require(n>0,"endpoint send failed or timed out");p+=n;bytes-=n;
        }
    }
    bool receive(void* data,std::size_t bytes,bool allowCleanEof=false) const
    {
        auto* p=static_cast<char*>(data);bool first=true;
        while(bytes)
        {
            const auto n=::recv(fd_,p,bytes,0);
            if(n<0&&errno==EINTR){continue;}
            if(n==0&&first&&allowCleanEof){return false;}
            require(n>0,"truncated endpoint response, worker exited, or receive timed out");
            first=false;p+=n;bytes-=n;
        }
        return true;
    }
private:
    int fd_=-1;
};
inline sockaddr_un address(const std::string& path)
{
    sockaddr_un a{};a.sun_family=AF_UNIX;
    require(!path.empty()&&path.size()<sizeof(a.sun_path),"socket path is empty or too long");
    std::memcpy(a.sun_path,path.c_str(),path.size()+1);return a;
}
struct Header
{
    std::uint64_t magic=0x314C415544584D47ULL,version=1,kind=0,sequence=0,natoms=0,nterms=0,npairs=0;
    std::int64_t step=0;double time=0;
};
inline void checkHeader(const Header& h)
{
    require(h.magic==0x314C415544584D47ULL&&h.version==1,"incompatible endpoint protocol");
    require(h.natoms>0&&h.natoms<100000000&&h.nterms<10000&&h.npairs<10000000,"invalid endpoint dimensions");
    require(std::isfinite(h.time),"invalid endpoint time");
}
inline void sendVector(const Socket& s,const std::vector<double>& a)
{if(!a.empty()){s.send(a.data(),a.size()*sizeof(double));}}
inline void receiveVector(const Socket& s,std::vector<double>& a,std::size_t size)
{a.resize(size);if(size){s.receive(a.data(),size*sizeof(double));}finite(a);}
class Channel
{
public:
    explicit Channel(const std::string& path,bool listener):path_(listener?path:"")
    {
        const auto a=address(path);
        if(listener)
        {
            Socket server(::socket(AF_UNIX,SOCK_STREAM,0));require(server.fd()>=0,"cannot create listener");
            // An existing path is an error. Never unlink another process's socket.
            require(::bind(server.fd(),reinterpret_cast<const sockaddr*>(&a),sizeof(a))==0,"cannot bind socket (path may exist)");
            owned_=true;require(::lstat(path.c_str(),&identity_)==0,"cannot inspect owned socket");
            require(::chmod(path.c_str(),0600)==0,"cannot restrict socket permissions");
            require(::listen(server.fd(),1)==0,"cannot listen");
            // The launching script supervises startup/connection timeout.
            connection_.reset(::accept(server.fd(),nullptr,nullptr));
            require(connection_.fd()>=0,"cannot accept local force client");
        }
        else
        {
            const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            for(;;)
            {
                Socket candidate(::socket(AF_UNIX,SOCK_STREAM,0));require(candidate.fd()>=0,"cannot create client");
                if(::connect(candidate.fd(),reinterpret_cast<const sockaddr*>(&a),sizeof(a))==0)
                {connection_.reset(::dup(candidate.fd()));break;}
                require(errno==ENOENT||errno==ECONNREFUSED,"cannot connect to endpoint socket");
                require(std::chrono::steady_clock::now()<until,"endpoint socket connection timed out");
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        connection_.timeout();
    }
    ~Channel()
    {
        if(owned_)
        {
            struct stat now{};
            if(::lstat(path_.c_str(),&now)==0&&S_ISSOCK(now.st_mode)&&now.st_dev==identity_.st_dev&&now.st_ino==identity_.st_ino)
            {::unlink(path_.c_str());}
        }
    }
    Endpoint evaluate(const Request& r)
    {
        const auto n=r.rawCharges.size();require(r.coordinates.size()==3*n&&r.masses.size()==n,"invalid request arrays");
        finite(r.coordinates);finite(r.rawCharges);finite(r.masses);
        Header h;h.kind=1;h.sequence=++sequence_;h.natoms=n;h.step=r.step;h.time=r.time;
        connection_.send(&h,sizeof(h));connection_.send(r.box.data(),9*sizeof(double));
        sendVector(connection_,r.coordinates);sendVector(connection_,r.rawCharges);sendVector(connection_,r.masses);
        Header reply;connection_.receive(&reply,sizeof(reply));checkHeader(reply);
        require(reply.kind==2&&reply.sequence==h.sequence&&reply.natoms==n&&reply.step==h.step&&reply.time==h.time,"stale or mismatched endpoint frame");
        Endpoint e;receiveVector(connection_,e.forces,3*n);receiveVector(connection_,e.charges,n);
        receiveVector(connection_,e.terms,reply.nterms);receiveVector(connection_,e.pairs,reply.npairs);
        connection_.receive(e.virial.data(),9*sizeof(double));e.validate(n);return e;
    }
    bool next(Request* r,std::size_t expectedAtoms)
    {
        Header h;if(!connection_.receive(&h,sizeof(h),true)){return false;}checkHeader(h);
        require(h.kind==1&&h.natoms==expectedAtoms&&h.sequence==sequence_+1,"invalid request sequence or atom count");
        sequence_=h.sequence;pending_=h;r->step=h.step;r->time=h.time;
        connection_.receive(r->box.data(),9*sizeof(double));for(double v:r->box){require(std::isfinite(v),"invalid box");}
        receiveVector(connection_,r->coordinates,3*h.natoms);receiveVector(connection_,r->rawCharges,h.natoms);
        receiveVector(connection_,r->masses,h.natoms);return true;
    }
    void respond(const Endpoint& e,std::int64_t step,double time)
    {
        e.validate(pending_.natoms);require(step==pending_.step&&time==pending_.time,"worker returned the wrong frame");
        Header h=pending_;h.kind=2;h.nterms=e.terms.size();h.npairs=e.pairs.size();
        connection_.send(&h,sizeof(h));sendVector(connection_,e.forces);sendVector(connection_,e.charges);
        sendVector(connection_,e.terms);sendVector(connection_,e.pairs);connection_.send(e.virial.data(),9*sizeof(double));
    }
private:
    Socket connection_;std::string path_;bool owned_=false;struct stat identity_{};
    std::uint64_t sequence_=0;Header pending_{};
};
#else
class Channel
{
public:
    Channel(const std::string&,bool){throw std::runtime_error("Dual native endpoint IPC currently requires Linux/WSL");}
    Endpoint evaluate(const Request&){return {};}
    bool next(Request*,std::size_t){return false;}
    void respond(const Endpoint&,std::int64_t,double){}
};
#endif
}
#endif
