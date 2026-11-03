#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Compile and run model-independent bridge checks; Linux/WSL, C++17.

Outputs are created in a new directory. No model, simulation or GPU is needed.
"""
from pathlib import Path
import argparse, json, shutil, subprocess, tempfile, sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--out')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = Path(args.out).resolve() if args.out else Path(tempfile.mkdtemp(prefix='gmx-mireann-check-'))
    if args.out: out.mkdir(parents=True, exist_ok=False)
    cxx = shutil.which(args.cxx)
    if not cxx: raise RuntimeError('C++ compiler not found: ' + args.cxx)
    cases = [root/'tests/dynamiccharge/mireann_subtraction_check.cpp',
             root/'tests/dynamiccharge/mireann_shift_check.cpp',
             *sorted((root/'tests/mireann').glob('test_*.cpp'))]
    results = []
    for src in cases:
        binary = out/src.stem
        compile_cmd = [cxx, '-std=c++17', '-O2', '-pthread', '-Wall', '-Wextra',
                       '-I', str(root/'src/gromacs/mdlib'), str(src), '-o', str(binary)]
        built = subprocess.run(compile_cmd, text=True, capture_output=True, timeout=120)
        (out/(src.stem+'-build.log')).write_text(built.stdout+built.stderr)
        if built.returncode: raise RuntimeError('Compilation failed: ' + src.name)
        ran = subprocess.run([str(binary)], text=True, capture_output=True, timeout=60)
        (out/(src.stem+'.log')).write_text(ran.stdout+ran.stderr)
        if ran.returncode: raise RuntimeError('Check failed: ' + src.name)
        results.append({'test': src.name, 'exit_code': ran.returncode, 'output': ran.stdout.strip()})
        print(src.name + ': PASS', flush=True)
    checked = subprocess.run([sys.executable, str(root/'tests/mireann/test_inputs.py')],
                             text=True, capture_output=True, timeout=60)
    (out/'test_inputs.log').write_text(checked.stdout+checked.stderr)
    if checked.returncode: raise RuntimeError('Input preparation checks failed')
    results.append({'test':'test_inputs.py','exit_code':checked.returncode,'output':checked.stdout+checked.stderr})
    (out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print('Results:', out)

if __name__ == '__main__': main()
