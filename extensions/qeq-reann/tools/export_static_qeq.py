# GMX_QEQ_REANN NEW FILE: offline exporter, not part of the MD Python runtime.
"""Export a fixed-element, nonperiodic charge-only REANN specialization.

Usage: python export_static_qeq.py --model REANN_CHA_DOUBLE.pt \
  --output REANN_CHA_STATIC_DOUBLE.pt --elements C H H H C H H H

Only offline export uses Python. MD inference remains native C++/LibTorch.
All trained weights remain float64 and unchanged; no retraining or time-step
skipping. For small fixed regions only (2..64 atoms, model PBC 0 0 0).
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from qeq_static_model import StaticCharge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--elements', nargs='+', required=True)
    args = parser.parse_args()
    if not 2 <= len(args.elements) <= 64:
        parser.error('This all-pair specialization is limited to 2..64 atoms')
    if args.model.resolve() == args.output.resolve() or args.output.exists():
        parser.error('Output must be a new file; the original model is never overwritten')
    model = torch.jit.load(str(args.model), map_location='cpu').eval()
    if 'aten::grad(' in str(model.forward.graph):
        parser.error('An energy/force model cannot use the charge-only specialization')
    for parameter in model.parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float64:
            parser.error('This exporter expects the supplied float64 charge model')
    names = [name for name, _ in model.nnmod.elemental_nets.named_children()]
    if any(symbol not in names for symbol in args.elements):
        parser.error('An element is absent from the trained model')
    species = torch.tensor([names.index(symbol) for symbol in args.elements],dtype=torch.int64)
    candidate = torch.jit.script(StaticCharge(model,species).eval())
    candidate.save(str(args.output))
    audit = dict(source_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                 output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                 elements=args.elements,trained_element_order=names,model_pbc=[0,0,0],
                 precision='float64',weights='unchanged',
                 note='Validate on representative coordinates before a new composition is used for MD.')
    args.output.with_suffix('.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
