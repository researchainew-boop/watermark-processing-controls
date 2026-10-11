"""Validation-only wider gray-band search for the literal gated score."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from hard_gate_analysis import gated, THRESHOLD
from aiwstudy.decision import DecisionPolicy, decision_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/extension/hard_gate'

def main():
    validation = pd.read_csv(ROOT/'outputs/conference_clean/validation_predictions.csv',keep_default_na=False)
    validation['fusion_score'] = np.where(validation.active_score>=THRESHOLD,validation.fusion,validation.probe)
    grid = [round(x,2) for x in np.arange(.02,.50,.01)]
    rows = []
    for margin in grid:
        metrics = decision_metrics(validation,DecisionPolicy(THRESHOLD,margin))
        groups = metrics['group_positive_rate']
        edited = groups['R-T']['positive_rate']; clean = groups['R-C']['positive_rate']
        rows.append({'margin':margin,'coverage':metrics['coverage'],
            'balanced_accuracy_decided':metrics['balanced_accuracy_decided'],
            'edited_real_fpr':edited,'clean_real_fpr':clean,'gap':edited-clean,
            'feasible':metrics['coverage']>=.70 and edited<=.05 and edited-clean<=.02})
    feasible = [x for x in rows if x['feasible']]
    selected = max(feasible or rows,key=lambda x:(-max(x['gap'],0),-max(x['edited_real_fpr']-.05,0),x['balanced_accuracy_decided'],x['coverage']))
    pd.DataFrame(rows).to_csv(OUT/'validation_margin_grid.csv',index=False)
    policy = {'threshold':THRESHOLD,'selected':selected,'margin_grid':grid,'selection_data':'corrected seen validation only','constraints_feasible':bool(feasible)}
    (OUT/'policy.json').write_text(json.dumps(policy,indent=2),encoding='utf-8')
    test = pd.read_csv(ROOT/'outputs/conference_clean/test_predictions.csv',keep_default_na=False)
    test['fusion_score'],_ = gated(test)
    result = decision_metrics(test,DecisionPolicy(THRESHOLD,selected['margin']))
    (OUT/'policy_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(policy,result,flush=True)

if __name__=='__main__': main()
