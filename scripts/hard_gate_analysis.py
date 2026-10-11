"""Literal 13/16 gate and paired clean marking-prevalence diagnostic.

Retrospective old-test analysis. Parameters are frozen before fresh inference.
No training, test-driven threshold selection, or attack-prevalence extrapolation.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experimental_reliability import prepare_auc, weighted_auc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/extension/hard_gate'
THRESHOLD = 13 / 16


def gated(frame, prefix=''):
    enabled = frame.active_score.to_numpy() >= THRESHOLD
    return (np.where(enabled, frame[prefix+'fusion'], frame[prefix+'probe']),
            np.where(enabled, frame[prefix+'fusion_logit'], frame[prefix+'probe_logit']))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'agreement_threshold': THRESHOLD, 'bits_required': 13, 'payload_bits': 16,
        'rule': 'existing fusion if agreement >=13/16, otherwise exact passive probe',
        'retrospective_old_test': True, 'fresh_test_tuning_allowed': False,
        'marked_shares': [i/10 for i in range(11)],
        'prevalence_scope': 'Clean paired unmarked/marked AI; real clean controls held fixed. No claim about unobserved unmarked attacks.',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    freeze = OUT / 'protocol.json'
    if not freeze.exists():
        freeze.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    frame = pd.read_csv(ROOT/'outputs/conference_clean/test_predictions.csv', keep_default_na=False)
    result, output = {}, frame.copy()
    rng = np.random.default_rng(20261009)
    for name, prefix in [('resnet',''), ('clip','clip_')]:
        prob, rank = gated(frame, prefix)
        output[name+'_gated'] = prob
        output[name+'_gated_logit'] = rank
        if not np.array_equal(prob[frame.active_score < THRESHOLD], frame.loc[frame.active_score < THRESHOLD,prefix+'probe'].to_numpy()):
            raise AssertionError('Fallback changed probe scores')
        slices = {'full': np.ones(len(frame), dtype=bool),
            'reduced_recovery': ((frame.attack=='crop') | ((frame.attack=='lowres') & (frame.severity==2)) | frame.attack.isin(['regenerate','spectral'])).to_numpy()}
        summary = {}
        for label, mask in slices.items():
            sub = frame.loc[mask]
            y = sub.label.to_numpy()
            scores = {'probe': sub[prefix+'probe_logit'].to_numpy(), 'original_fusion': sub[prefix+'fusion_logit'].to_numpy(), 'gated': rank[mask]}
            prepared = {key: prepare_auc(y,val) for key,val in scores.items()}
            codes, sources = pd.factorize(sub.source_id, sort=True)
            strata = [np.unique(codes[sub.generator.to_numpy()==g]) for g in sorted(sub.generator.unique())]
            values = {key: [] for key in scores}
            for _ in range(2000):
                w = np.zeros(len(sources))
                for group in strata:
                    w[group] = np.bincount(rng.integers(len(group),size=len(group)),minlength=len(group))
                for key in scores:
                    values[key].append(weighted_auc(prepared[key],w[codes]))
            summary[label] = {key: {'auc': float(roc_auc_score(y,val)), 'ci95': np.percentile(values[key],[2.5,97.5]).tolist()} for key,val in scores.items()}
            summary[label]['gated_gain_ci95'] = np.percentile(np.array(values['gated'])-values['probe'],[2.5,97.5]).tolist()
            summary[label]['sources'] = len(sources)
        clean = frame.loc[frame.group.isin(['R-C','S-U','S-W'])].copy()
        _, cleanrank = gated(clean,prefix)
        prevalence = []
        for share in protocol['marked_shares']:
            weights = np.where(clean.group=='S-U',1-share,np.where(clean.group=='S-W',share,1.))
            systems = {'probe': clean[prefix+'probe_logit'].to_numpy(), 'original_fusion': clean[prefix+'fusion_logit'].to_numpy(), 'gated': cleanrank}
            row = {'marked_share': share}
            for key,val in systems.items():
                row[key+'_auc'] = float(roc_auc_score(clean.label,val,sample_weight=weights))
            row['gated_gain'] = row['gated_auc']-row['probe_auc']
            prevalence.append(row)
        summary['clean_marked_prevalence'] = prevalence
        result[name] = summary
        print(name,summary['full'],summary['reduced_recovery'],flush=True)
    output.to_csv(OUT/'test_predictions.csv',index=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
