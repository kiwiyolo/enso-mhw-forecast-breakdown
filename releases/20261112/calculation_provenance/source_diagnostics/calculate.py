"""Historical-normalized pattern-fidelity loss, retaining every archived coefficient."""
from pathlib import Path
import hashlib,json,shutil,math
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'source_data';DATA.mkdir(exist_ok=True)
roots=yaml.safe_load(Path('/data3/luoq/FEIS/configs/paths.yaml').read_text())['roots']
paper=Path(roots['source'])/'research/papers/mhw-predictability'
previous=paper/'Figures/Figure2_relative_atmospheric_loss_20261029_v15'
source=previous/'source_data/pattern_correlations.csv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copy2(source,DATA/'pattern_correlations.csv')
table=pd.read_csv(source)
assert len(table)==81 and not table.duplicated(['Process','Event','Lead']).any()
assert np.isfinite(table.Pattern_correlation).all() and table.Pattern_correlation.between(-1,1).all()
processes=['SST','Zonal wind stress','Convection proxy']
events=['1997/98','2015/16','2023/24']
assert set(table.Process)==set(processes) and set(table.Event)==set(events)
for _,g in table.groupby(['Process','Event']):assert sorted(g.Lead)==list(range(1,10))

def loss(historical,target):
    return 100*(historical-target)/historical if np.isfinite([historical,target]).all() and historical>0 else float('nan')
assert loss(.8,.4)==50 and loss(.8,.8)==0 and loss(.4,.8)==-100
assert math.isnan(loss(0,.2)) and math.isnan(loss(-.2,.2))

rows=[];leadrows=[];sens=[]
for process in processes:
    t=table[table.Process.eq(process)].pivot(index='Lead',columns='Event',values='Pattern_correlation').sort_index()
    hist=(t['1997/98']+t['2015/16'])/2
    target=t['2023/24']
    h=float(hist.mean());q=float(target.mean())
    rows.append(dict(Process=process,Historical_r=h,Target_r=q,
        Absolute_r_decline=h-q,Relative_decline_percent=loss(h,q)))
    for lead in t.index:
        leadrows.append(dict(Process=process,Lead=int(lead),Historical_r=float(hist[lead]),
            Target_r=float(target[lead]),Relative_decline_percent=loss(float(hist[lead]),float(target[lead]))))
    for event in events[:2]:
        h_event=float(t[event].mean())
        sens.append(dict(Process=process,Historical_event=event,Historical_r=h_event,
            Target_r=q,Relative_decline_percent=loss(h_event,q)))
summary=pd.DataFrame(rows)
sst=float(summary.loc[summary.Process.eq('SST'),'Relative_decline_percent'].iloc[0])
assert sst>0
summary['Relative_to_SST']=summary.Relative_decline_percent/sst
summary['Excess_over_SST_percentage_points']=summary.Relative_decline_percent-sst
sensitivity=pd.DataFrame(sens)
for event in events[:2]:
    value=float(sensitivity.loc[sensitivity.Process.eq('SST') & sensitivity.Historical_event.eq(event),'Relative_decline_percent'].iloc[0])
    assert value>0
    mask=sensitivity.Historical_event.eq(event)
    sensitivity.loc[mask,'Relative_to_SST']=sensitivity.loc[mask,'Relative_decline_percent']/value
summary.to_csv(DATA/'relative_loss_summary.csv',index=False)
pd.DataFrame(leadrows).to_csv(DATA/'relative_loss_by_lead.csv',index=False)
sensitivity.to_csv(DATA/'historical_reference_sensitivity.csv',index=False)
paths=[source,previous/'figures/Figure2.pdf',previous/'MANIFEST.json',
       paper/'Figures/Revised_20260921/Figure3/Figure3_process_forecast_skill.csv']
assert sha(source)==sha(paths[-1])
(DATA/'input_lineage.json').write_text(json.dumps([dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in paths],indent=2)+'\n')
qa=dict(status='PASS arithmetic and completeness',input_coefficients=81,summary_rows=3,lead_rows=27,
    input_exclusions=0,all_values_finite=bool(np.isfinite(summary.select_dtypes('number')).all().all()),
    positive_denominators=True,unit_tests_passed=True,raw_inputs_modified=False,
    no_significance_test=True)
(ROOT/'numerical_QA.json').write_text(json.dumps(qa,indent=2)+'\n')
print(summary.to_string(index=False))
