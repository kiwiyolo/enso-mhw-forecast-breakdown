"""Publish a new immutable figure directory; never overwrite an older release."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parent
TARGET=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures/Figure3_compact_layout_20261104_v21')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
qa=json.loads((ROOT/'QA.json').read_text())
assert qa['pass'], 'Publication requires all numerical and export checks to pass.'
manifest=json.loads((ROOT/'manifest.json').read_text())
assert all(sha(ROOT/e['path'])==e['sha256'] for e in manifest), 'Manifest must be current.'
assert not TARGET.exists(), 'Refusing to overwrite an existing release.'
TARGET.mkdir()
for e in manifest:
    p=TARGET/e['path'];p.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/e['path'],p)
shutil.copy2(ROOT/'manifest.json',TARGET/'manifest.json')
assert all(sha(TARGET/e['path'])==e['sha256'] for e in manifest)
bundle=TARGET/'Figure3_reproduction.zip'
with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for e in manifest:z.write(TARGET/e['path'],e['path'])
    z.write(TARGET/'manifest.json','manifest.json')
result={'target':str(TARGET),'files_verified':len(manifest),'all_hashes_match':True,
        'QA_pass':qa['pass'],'new_directory_only':True,'bundle_sha256':sha(bundle),
        'manuscript_and_other_figures_changed':False}
(TARGET/'publication_QA.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
