"""Non-overwriting review release with hashes and complete source tables."""
from pathlib import Path
import json,hashlib,shutil,zipfile
ROOT=Path(__file__).resolve().parent
TARGET=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures/Figure4_full_hierarchy_percent_20261104_v11')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not TARGET.exists(),'Never overwrite an existing release'
    q=json.loads((ROOT/'QA.json').read_text());assert all(c['passed'] for c in q['checks'])
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    # The source figure, arrays and prior audit must remain unchanged.
    for x in manifest:assert sha(x['source'])==x['sha256'],x['source']
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in {'manifest.json','publication.json'} and p.suffix!='.zip']
    entries=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in files]
    (ROOT/'manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
    TARGET.mkdir()
    for p in files:
        d=TARGET/p.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d);assert sha(p)==sha(d)
    shutil.copy2(ROOT/'manifest.json',TARGET/'manifest.json')
    bundle=TARGET/'Figure4_full_hierarchy_percent_20261104_v11.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for e in entries:z.write(TARGET/e['path'],e['path'])
        z.write(TARGET/'manifest.json','manifest.json')
    r=dict(target=str(TARGET),files=len(entries),verified=True,originals_unchanged=True,archive=str(bundle),sha256=sha(bundle),
           status='Full review-layout delivered; archived-input validity and physical process attribution NOT validated')
    for p in [ROOT/'publication.json',TARGET/'publication.json']:p.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
