#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Check physical MM charge persistence in a MIREANN environment CSV.

Use mireann-environment-log for a short run spanning neighbor-list rebuilds.
q_e is the physical MM charge; model_q_e may vary with switching weights.
Only atoms that occur at least twice can be checked. This does not audit atoms
outside the model/reference environment and is not an NVE-conservation test.
"""
from pathlib import Path
import argparse, csv, json, math

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--tolerance',type=float,default=1e-12)
    args=parser.parse_args()
    seen={}; steps=set(); maximum=0.; rows=0
    for row in csv.DictReader(Path(args.csv).open()):
        q=float(row['q_e']); atom=int(row['atom_1based']); step=int(row['step'])
        if not math.isfinite(q): raise RuntimeError('Nonfinite physical MM charge')
        steps.add(step); rows+=1
        if atom not in seen: seen[atom]=[q,1]
        else:
            maximum=max(maximum,abs(q-seen[atom][0]));seen[atom][1]+=1
    checked=sum(count>1 for _,count in seen.values())
    if len(steps)<2 or not checked: raise RuntimeError('Insufficient cross-step observations')
    result={'rows':rows,'steps':len(steps),'repeated_atoms':checked,'max_charge_change_e':maximum}
    print(json.dumps(result,indent=2))
    if maximum>args.tolerance: raise RuntimeError('Physical MM charges changed across force evaluations')

if __name__=='__main__':main()
