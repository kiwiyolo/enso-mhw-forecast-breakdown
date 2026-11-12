"""Assemble the already regenerated SI, retaining S7's original export."""
from pathlib import Path
import json, shutil, sys
ROOT=Path(__file__).resolve().parent/'supplementary'
sys.path.insert(0,str(ROOT))
import package_si
for ext in ['pdf','png','svg']:
    src=ROOT/f'matched_s5/figures/Supplementary_Fig_05.{ext}'
    if src.exists():shutil.copy2(src,ROOT/f'figures/Supplementary_Fig_05.{ext}')
records=json.loads((ROOT/'figure_registry.json').read_text())
next(r for r in records if r['number']==5).update(json.loads((ROOT/'matched_s5/caption.json').read_text()))
package_si.package(records)
