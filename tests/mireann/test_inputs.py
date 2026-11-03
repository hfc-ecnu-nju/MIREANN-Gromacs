#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Model-free checks for endpoint preparation and portable command paths."""
from pathlib import Path
import importlib.util, json, os, shutil, tempfile, unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('dual_prepare', ROOT/'extensions/dual-hamiltonian/prepare_inputs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class InputPreparation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gmx-dual-input-test-')
        self.root = Path(self.temp.name)
        self.mdp = self.root/'original inputs/md.mdp'
        self.mdp.parent.mkdir()
        (self.mdp.parent/'PES.pt').write_bytes(b'placeholder-not-a-model')
        (self.mdp.parent/'EE.pt').write_bytes(b'placeholder-not-a-model')
        self.mdp.write_text('integrator=md\npcoupl=no\nmireann-active=yes\nmireann-charge-response=yes\n'
                            'mireann-model=EE.pt\nmireann-atoms=1 2\nnnpot-active=yes\n'
                            'nnpot-modelfile=PES.pt\npull=yes\npull-coord1-k=100\n')
        self.top=self.root/'system.top'
        self.top.write_text('; raw TOP charges must never be rewritten\n')
        self.gro=self.root/'start.gro'; self.gro.write_text('placeholder\n')
        self.ndx=self.root/'qm.ndx'; self.ndx.write_text('[ QM ]\n1 2\n')
    def tearDown(self): self.temp.cleanup()
    def prepare(self, s):
        return module.prepare('python3',self.mdp,self.top,self.gro,self.ndx,s,self.root/'prepared',False)
    def test_separate_endpoints_and_paths(self):
        before=self.top.read_bytes()
        config=self.prepare(.5)
        c=json.loads(config.read_text())
        self.assertEqual(c['gmx'],str(Path(shutil.which('python3')).resolve()))
        mm=module.entries((config.parent/'mm.mdp').read_text())
        ml=module.entries((config.parent/'mlmm.mdp').read_text())
        self.assertEqual(mm['mireann-active'],'no')
        self.assertNotIn('mireann-atoms',mm)
        self.assertEqual(mm['dual-hamiltonian-s'],'0.5')
        self.assertEqual(ml['mireann-active'],'yes')
        self.assertEqual(mm['pull-coord1-k'],ml['pull-coord1-k'])
        self.assertTrue(Path(ml['mireann-model']).is_absolute())
        self.assertEqual(self.top.read_bytes(),before)
        generated=(config.parent/'grompp-commands.sh').read_text()
        self.assertIn('cd -- ',generated)
        self.assertIn('original inputs',generated)
    def test_invalid_s_rejected(self):
        for s in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):self.prepare(s)
    def test_npt_rejected(self):
        self.mdp.write_text(self.mdp.read_text().replace('pcoupl=no','pcoupl=c-rescale'))
        with self.assertRaises(ValueError):self.prepare(.5)

if __name__=='__main__': unittest.main()
