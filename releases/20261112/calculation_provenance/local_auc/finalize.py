"""Finalize immutable manifests after render/package/validation have exited."""
from pathlib import Path
import json,hashlib
import fitz,numpy as np
ROOT=Path(__file__).resolve().parent
SI=ROOT.parent/'Supplementary_Figures_20261111_v3'
PRIOR=ROOT.parent/'Supplementary_Figures_20261111_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for i in range(2,16):
    name=f'Supplementary_Fig_{i:02d}.pdf'
    ok=sha(SI/'figures'/name)==sha(PRIOR/'figures'/name)
    assert ok,name
    checks.append(dict(test='unchanged unrelated SI artwork '+name,passed=ok))
with fitz.open(SI/'figures/Supplementary_Fig_01.pdf') as a,fitz.open(PRIOR/'figures/Supplementary_Fig_01.pdf') as b:
    clip=fitz.Rect(0,0,a[0].rect.width,a[0].rect.height*.5)
    aa=a[0].get_pixmap(dpi=150,alpha=False,clip=clip);bb=b[0].get_pixmap(dpi=150,alpha=False,clip=clip)
    delta=int(np.abs(np.frombuffer(aa.samples,np.uint8).astype(int)-np.frombuffer(bb.samples,np.uint8).astype(int)).max())
    assert delta==0,delta
    checks.append(dict(test='S1a,b pixel-identical to v2',passed=True,maximum_channel_difference=delta))
(SI/'qa/unchanged_artwork.json').write_text(json.dumps(checks,indent=2))
for folder in [ROOT,SI]:
    manifest=[dict(path=str(p.relative_to(folder)),sha256=sha(p)) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='release_manifest.json' and '__pycache__' not in p.parts and p.suffix!='.log']
    (folder/'release_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(folder.name,len(manifest),'files',flush=True)
