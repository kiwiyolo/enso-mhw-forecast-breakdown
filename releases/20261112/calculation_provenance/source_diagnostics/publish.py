"""Publish independently verified v16 without changing older releases."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,yaml
ROOT=Path(__file__).resolve().parent
roots=yaml.safe_load(Path('/data3/luoq/FEIS/configs/paths.yaml').read_text())['roots']
paper=Path(roots['source'])/'research/papers/mhw-predictability'
out=paper/'Figures/Figure2_relative_atmospheric_loss_20261029_v16'
data=Path(roots['data'])/'processed/research/mhw-predictability/figure2-relative-atmospheric-loss-20261029-v16'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
qa=json.loads((ROOT/'validation_QA.json').read_text())
assert qa['a_source_image_bytes_identical'] and qa['independently_recomputed']
assert qa['coefficient_count']==81 and qa['plotted_b_curve_rows']==27
assert qa['previous_values_exactly_preserved'] and qa['reference_sensitivity_points']==6
lineage=json.loads((ROOT/'source_data/input_lineage.json').read_text())
assert all(sha(Path(r['path']))==r['sha256'] for r in lineage)
assert not out.exists() and not data.exists(), 'Refusing overwrite'
data.mkdir(parents=True,exist_ok=False);shutil.copytree(ROOT/'source_data',data/'source_data')
out.mkdir(parents=True,exist_ok=False);shutil.copytree(ROOT/'figures',out/'figures')
for name in ['calculate.py','build_figure.py','validate.py','publish.py','metric-contract.json',
             'decision-log.csv','Figure2.md','QA.md','numerical_QA.json','render_QA.json','validation_QA.json']:
    shutil.copy2(ROOT/name,out/name)
(out/'source_data').symlink_to(data/'source_data',target_is_directory=True)
records=[]
for owner,folder in [('paper',out),('data',data)]:
    for p in sorted(folder.rglob('*')):
        if p.is_symlink() or not p.is_file():continue
        rel=p.relative_to(folder);assert sha(p)==sha(ROOT/rel)
        records.append(dict(owner=owner,path=str(rel),bytes=p.stat().st_size,sha256=sha(p)))
manifest=dict(version=out.name,created_utc=datetime.now(timezone.utc).isoformat(),
    data_root=str(data),baseline='Figure2_relative_atmospheric_loss_20261029_v15',
    a_image_payload_identical=True,b_data_identical=True,raw_input_modified=False,prior_versions_preserved=True,
    manuscript_modified=False,metric='Historical-normalized pattern-correlation decline and its ratio to the SST decline',
    records=records)
(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert all(sha(Path(r['path']))==r['sha256'] for r in lineage)
print(json.dumps(dict(output=str(out),data=str(data),verified_files=len(records)),indent=2))
