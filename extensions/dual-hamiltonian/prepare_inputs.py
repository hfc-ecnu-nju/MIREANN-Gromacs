"""Prepare independent fixed-s MM/MLMM input files from the ORIGINAL topology.

The ML/MM Hamiltonian options are preserved; raw TOP charges are never edited.
Run this in the existing input directory so relative model paths are resolved.
"""
from pathlib import Path
import argparse,json,os,re,subprocess,uuid,shlex,shutil

def entries(text):
    return {k.strip().lower().replace('_','-'):v.split(';')[0].strip()
            for k,v in re.findall(r'^\s*([^;\n=]+)=(.*)$',text,re.M)}

def prepare(gmx,mdp,top,coordinates,index,s,out,compile_tpr=True):
    if not 0<=s<=1:raise ValueError('s must lie in [0,1]')
    # GMX_RELEASE_REVIEW_20261102: accept a PATH command without treating it as a local filename.
    gmx=shutil.which(str(gmx)) or str(Path(gmx).resolve())
    if compile_tpr and not Path(gmx).is_file():raise FileNotFoundError(gmx)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    mdp=Path(mdp).resolve();top=Path(top).resolve();coordinates=Path(coordinates).resolve();index=Path(index).resolve()
    text=mdp.read_text();kv=entries(text)
    if kv.get('integrator','md').lower()!='md':raise ValueError('This version supports integrator=md only')
    if kv.get('pcoupl','no').lower()!='no':raise ValueError('Initial dual version supports fixed-box NVE/NVT, not NPT')
    if kv.get('mireann-active','no').lower() not in ['yes','true']:raise ValueError('Provide the existing active MIREANN MD input')
    if kv.get('mireann-charge-response','yes').lower() not in ['yes','true']:
        raise ValueError('Energy-consistent dual Hamiltonian requires the existing full charge-response=yes endpoint')
    if kv.get('free-energy','no').lower()!='no':raise ValueError('Native FEP must be off; s is a separate fixed coupling')
    socket='/tmp/gmx-dual-'+uuid.uuid4().hex[:12]+'/endpoint.sock'
    def update(source,replacements,drop_ml=False):
        result=[];done=set()
        for line in source.splitlines():
            match=re.match(r'^\s*([^;=]+)=(.*)$',line)
            if match:
                key=match[1].strip().lower().replace('_','-')
                if drop_ml and key.startswith(('mireann-','qeq-reann-','nnpot-','dual-hamiltonian-')):continue
                if key in replacements:
                    if key not in done:result.append(key+' = '+str(replacements[key]));done.add(key)
                    continue
            result.append(line)
        result.extend(k+' = '+str(v) for k,v in replacements.items() if k not in done)
        return '\n'.join(result)+'\n'
    # Model locations become absolute; otherwise parameters and model weights
    # remain those of the supplied existing ML/MM endpoint.
    mlvalues={'dual-hamiltonian-active':'no','nstcalcenergy':1,'nsteps':-1}
    for key in ['mireann-model','nnpot-modelfile']:
        if key in kv:
            model=Path(kv[key]);model=model if model.is_absolute() else mdp.parent/model
            if not model.is_file():raise FileNotFoundError(model)
            mlvalues[key]=str(model.resolve())
    mmvalues={'nnpot-active':'no','mireann-active':'no','qeq-reann-active':'no','nstcalcenergy':1,
              'dual-hamiltonian-active':'yes','dual-hamiltonian-s':format(s,'.17g'),
              'dual-hamiltonian-socket':socket,'dual-hamiltonian-energy-log':'dual-energy.csv'}
    (out/'mm.mdp').write_text(update(text,mmvalues,drop_ml=True));(out/'mlmm.mdp').write_text(update(text,mlvalues))
    commands=[]
    for name in ['mm','mlmm']:
        command=[str(Path(gmx).resolve()),'grompp','-f',str(out/(name+'.mdp')),'-p',str(top),'-c',str(coordinates),'-n',str(index),'-o',str(out/(name+'.tpr')),'-po',str(out/(name+'.processed.mdp'))]
        commands.append(command)
        if compile_tpr:
            with (out/(name+'-grompp.log')).open('w') as f:
                ret=subprocess.run(command,cwd=mdp.parent,stdout=f,stderr=subprocess.STDOUT)
            if ret.returncode:raise RuntimeError(name+' grompp failed; inspect its log; warnings are not automatically ignored')
    config=dict(gmx=str(Path(gmx).resolve()),s=s,socket=socket,mm_tpr=str(out/'mm.tpr'),mlmm_tpr=str(out/'mlmm.tpr'),coordinates=str(coordinates),
                original_topology=str(top),original_mlmm_mdp=str(mdp),commands=commands,
                rule='MM retains original TOP charges and every classical interaction. MLMM preserves supplied endpoint. Only full endpoint energies and forces are mixed.')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'grompp-commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+ 'cd -- '+shlex.quote(str(mdp.parent))+'\n'+'\n'.join(shlex.join(c) for c in commands)+'\n')
    return out/'config.json'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['gmx','mdp','top','coordinates','index','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--s',type=float,required=True);p.add_argument('--no-grompp',action='store_true');a=p.parse_args()
    print(prepare(a.gmx,a.mdp,a.top,a.coordinates,a.index,a.s,a.out,not a.no_grompp))
