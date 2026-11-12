"""Check bundled data, signed accounting and the complete figure inventory."""
from pathlib import Path
import json,hashlib,re
import pandas as pd
import numpy as np
from scipy.stats import linregress
import fitz
ROOT=Path(__file__).resolve().parent
checks={}
def check(k,v):checks[k]=bool(v)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=pd.read_csv(ROOT/'main13/source_data/Fig1/Figure1_source_events.csv')
h=d[d.Event!='2023/24'];e=d[d.Event=='2023/24'].iloc[0];f=linregress(h.Peak_Nino34,h.Mean_AUC)
check('historical_events_nine',len(h)==9)
check('figure1_observed_auc',abs(e.Mean_AUC-.5959687883)<1e-7)
check('figure1_expected_auc',abs(f.intercept+f.slope*e.Peak_Nino34-.7050185255)<1e-7)
r=pd.read_csv(ROOT/'main4/source_data/regional_map_shortfall.csv',keep_default_na=False)
net=r.Net_shortfall.sum();positive=100*r.Positive_shortfall.sum()/net;negative=-100*r.Above_benchmark.sum()/net
check('figure4_net_denominator',abs(net-.1065143704)<1e-9)
check('signed_percentages_close_to_100',abs(positive+negative-100)<1e-9)
check('region_net_shares_close_to_100',abs(r.Net_share_pct.sum()-100)<1e-9)
q=pd.read_csv(ROOT/'supplementary/derived_data/S01c_all_seven_regions.csv',keep_default_na=False)
check('SI1_all_seven_regions',len(q)==7 and set(q.region)==set(['TP','NP','SP','TA','NA','SA','IO']))
figures=[ROOT/f'main13/figures/Figure{i}.pdf' for i in (1,2,3)]+[ROOT/'main4/figures/Figure4.pdf']
figures += [ROOT/f'supplementary/figures/Supplementary_Fig_{i:02d}.pdf' for i in range(1,18)]
for p in figures:
    check(str(p.relative_to(ROOT))+' exists',p.exists())
    if p.exists():
        with fitz.open(p) as doc:check(str(p.relative_to(ROOT))+' single page',len(doc)==1 and bool(doc[0].get_text().strip()))
with fitz.open(ROOT/'supplementary/Supplementary_Figures_with_captions.pdf') as d:
    check('SI_all_17_captioned_pages',len(d)==17)
    check('SI_caption_numbers_1_to_17',all(f'Supplementary Fig. {i+1} |' in p.get_text() for i,p in enumerate(d)))
if (ROOT/'checksums.json').exists():
    manifest=json.loads((ROOT/'checksums.json').read_text())
    for p,h in manifest.items():check('checksum '+p,(ROOT/p).is_file() and sha(ROOT/p)==h)
out={'checks':checks,'passed':sum(checks.values()),'total':len(checks),'positive_percent':positive,'negative_percent':negative,
     'scientific_submission_clearance':False,'scope':'Snapshot integrity and displayed statistics; not validation of archived initialization selection.'}
(ROOT/'validation.json').write_text(json.dumps(out,indent=2)+'\n')
print('Checks',sum(checks.values()),'/',len(checks))
assert all(checks.values()),[k for k,v in checks.items() if not v]
