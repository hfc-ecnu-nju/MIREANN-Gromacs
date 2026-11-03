"""One integrating MM driver + one native ML/MM force worker, same host only.

The worker uses -rerun solely to enter the native force loop; no trajectory is
read in worker mode and no second integrator runs. Both children are supervised.
No existing processes are attached, signalled or reused.
"""
from pathlib import Path
import argparse,json,os,subprocess,time,signal

def run(config,run_dir,ntomp=4,nb='cpu',pme='cpu',timeout=86400):
    c=json.loads(Path(config).read_text());root=Path(run_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    socket=Path(c['socket']);socket.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if socket.exists():raise RuntimeError('Socket already exists; refusing to unlink or reuse it.')
    # The server creates a mode-0600 socket; the local directory is private.
    if socket.parent.stat().st_uid!=os.getuid() or socket.parent.stat().st_mode&0o077:
        raise RuntimeError('Socket parent must be owned by this user and mode 0700.')
    worker_dir=root/'mlmm-worker';driver_dir=root/'mixed-md';worker_dir.mkdir();driver_dir.mkdir()
    env=os.environ.copy();env['OMP_NUM_THREADS']=str(ntomp);env['GMX_DISABLE_GPU_GRAPH']='1'
    env.pop('GMX_DUAL_HAMILTONIAN_WORKER',None)
    worker_env=env.copy();worker_env['GMX_DUAL_HAMILTONIAN_WORKER']=str(socket)
    common=['-ntmpi','1','-ntomp',str(ntomp),'-update','cpu','-nb',nb,'-pme',pme,'-pin','off']
    worker=[c['gmx'],'mdrun','-s',c['mlmm_tpr'],'-rerun',c['coordinates'],'-deffnm','endpoint',*common]
    driver=[c['gmx'],'mdrun','-s',c['mm_tpr'],'-deffnm','mixed',*common]
    state=dict(state='starting',s=c['s'],socket=str(socket),worker_argv=worker,driver_argv=driver)
    def save(): (root/'status.json').write_text(json.dumps(state,indent=2))
    save();children=[];started=time.monotonic()
    try:
        with (worker_dir/'console.log').open('w') as wlog,(driver_dir/'console.log').open('w') as dlog:
            wp=subprocess.Popen(worker,cwd=worker_dir,env=worker_env,stdout=wlog,stderr=subprocess.STDOUT);children.append(wp)
            state['worker_pid']=wp.pid;save()
            until=time.monotonic()+180
            while not socket.exists():
                if wp.poll() is not None:raise RuntimeError('Worker exited before opening its socket; see mlmm-worker/console.log')
                if time.monotonic()>until:raise TimeoutError('Worker startup timed out')
                time.sleep(.1)
            dp=subprocess.Popen(driver,cwd=driver_dir,env=env,stdout=dlog,stderr=subprocess.STDOUT);children.append(dp)
            state.update(state='running',driver_pid=dp.pid);save()
            while dp.poll() is None:
                if wp.poll() is not None:raise RuntimeError('Force worker exited while MD was active')
                if time.monotonic()-started>timeout:raise TimeoutError('This run exceeded its requested timeout')
                time.sleep(.2)
            if dp.returncode:raise RuntimeError('MM integrating driver failed; see mixed-md/console.log')
            # Driver close/EOF cleanly finishes native worker rerun.
            code=wp.wait(timeout=90)
            if code:raise RuntimeError('Native worker did not exit successfully')
            state.update(state='complete',driver_exit_code=dp.returncode,worker_exit_code=code,elapsed_seconds=time.monotonic()-started);save()
    except BaseException as e:
        state.update(state='failed',error=str(e));save();raise
    finally:
        # Only Popen objects created by this launcher may be terminated.
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    return root

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--ntomp',type=int,default=4);p.add_argument('--nb',choices=['cpu','gpu'],default='cpu');p.add_argument('--pme',choices=['cpu','gpu'],default='cpu');p.add_argument('--timeout',type=float,default=86400)
    a=p.parse_args()
    if a.ntomp<1:p.error('--ntomp must be positive')
    print(run(a.config,a.run_dir,a.ntomp,a.nb,a.pme,a.timeout))
