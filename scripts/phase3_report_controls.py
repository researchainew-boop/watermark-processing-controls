"""Summarize completed diagnostic controls without rerunning inference or fits."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from phase3_prepare_controls import digest, ROOT, OUT


def main():
    final=json.loads((OUT/'split_seeds/complete.json').read_text())
    assert final['rows_per_source']=={'train':13,'validation':13,'test_AI':30,'test_real':16}
    assert final['original_frozen_heads_unchanged']
    frozen=json.loads((OUT/'unmarked_frozen_results.json').read_text())
    assert frozen['expanded_test_rows']==46424 and frozen['expanded_test_sources']==1680
    frame=pd.read_csv(OUT/'split_seeds/mean_std_metrics.csv',keep_default_na=False)
    assignments=pd.read_csv(OUT/'split_seeds/assignments.csv',keep_default_na=False)
    source_counts=[]
    for seed,part in assignments.groupby('seed'):
        counts=part.groupby(['generator','split']).size().reset_index(name='source_count')
        total=part.groupby('split').size().to_dict()
        assert total=={'test':1843,'train':4102,'val':878}
        oi=part.loc[part.generator.eq('openimages')].groupby('split').size().to_dict()
        assert oi=={'test':150,'train':700,'val':150}
        assert part.loc[part.generator.eq('midjourney'),'split'].eq('test').all()
        source_counts.extend({'seed':int(seed),**row} for row in counts.to_dict('records'))
    summary={'completed':True,'original_frozen_heads_unchanged':True,
        'five_split_sources':6823,'source_counts_per_seed':{'train':4102,'validation':878,'test':1843},
        'test_rows_per_seed':49158,'AI_sources_per_test':1405,'real_sources_per_test':438,
        'OpenImages_sources_per_seed':{'train':700,'validation':150,'test':150},
        'unmarked_frozen':{key:frozen[key] for key in ['expanded_test_overall','matched_marked_edits','matched_unmarked_edits','paired_AI_probe_effects']},
        'mean_std':frame.to_dict('records'),
        'standard_deviation':'Sample SD across five split seeds; overlapping tests and an identical Midjourney holdout. It is neither an uncertainty interval nor an independent external test.',
        'measurement_provenance':{'frozen_report_sha256':digest(OUT/'unmarked_frozen_results.json'),
            'seed_metrics_sha256':digest(OUT/'split_seeds/per_seed_metrics.csv'),
            'five_seed_completion_sha256':digest(OUT/'split_seeds/complete.json')},
        'interpretation':'Retrospective diagnostic study, excluding the previously frozen 192-source sample. Every derivative follows its source split; probes fit training and fusion fits validation only.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(source_counts).to_csv(OUT/'split_seeds/source_counts.csv',index=False)
    lines=['# Completed Phase 3 detection controls','',
        'The original heads and the 192-source frozen sample remain unchanged. '
        'This report describes retrospective controls and repeated source splits; it does not turn the source pool into a fresh test set.','',
        '## Matched unmarked edits','',
        'The same fourteen transformations were applied directly to all 1,396 unmarked AI images in the original test. '
        'The corresponding marked copies use the same 284 real sources and attack settings. '
        'Adding the 19,544 copies gives 46,424 rows from 1,680 sources.','']
    for name in ['resnet','clip']:
        m=frozen['matched_marked_edits'][name+'_probe'];u=frozen['matched_unmarked_edits'][name+'_probe'];e=frozen['paired_AI_probe_effects'][name]
        lines.append(f"- {name.upper()} probe: marked-edit AUC {m['auc']:.4f}, unmarked-edit AUC {u['auc']:.4f}; test-ROC TPR at 5% FPR {m['tpr_at_test_roc_fpr_5pct']:.4f} and {u['tpr_at_test_roc_fpr_5pct']:.4f}. The mean marked-minus-unmarked probability response is {e['mean_probability_marked_minus_unmarked']:.6f}, with source-clustered 95% interval [{e['mean_probability_delta_ci95'][0]:.6f}, {e['mean_probability_delta_ci95'][1]:.6f}].")
    lines += ['', 'These comparisons reduce the concern that the probe merely reads the embedded mark. '
        'They do not remove collection, scene, codec, or backbone pretraining confounds.','',
        '## Five split seeds and a second real collection','',
        'The source pool contains the original 5,823 screened images plus 1,000 separately selected Open Images photographs. '
        'Each split contains 4,102 training, 878 validation, and 1,843 test sources. '
        'Open Images contributes 700/150/150 sources, and all 962 Midjourney sources remain in test. '
        'Training and validation each use thirteen original conditions. '
        'Tests include all sixteen original conditions and fourteen additional unmarked edits per AI source: 49,158 rows, with thirty per AI source and sixteen per real source.','',
        'Unmarked edited copies never enter training or calibration. '
        'The probe fits training only, fusion fits validation only, and the payload gate remains 13/16. '
        'The three originally held-out settings stay excluded from training and calibration.','']
    overall=frame.loc[frame.real_source.eq('overall') & ~frame.backbone.eq('active')]
    for row in overall.to_dict('records'):
        lines.append(f"- {row['backbone'].upper()} / {row['variant']} / {row['system']}: AUC {row['auc_mean']:.4f} ± {row['auc_std']:.4f}; test-ROC TPR at 5% FPR {row['tpr_at_test_roc_fpr_5pct_mean']:.4f} ± {row['tpr_at_test_roc_fpr_5pct_std']:.4f}.")
    lines += ['', '## Results by real source','']
    for row in frame.loc[frame.real_source.isin(['coco','openimages']) & frame.variant.eq('full') & frame.system.eq('probe')].to_dict('records'):
        lines.append(f"- {row['backbone'].upper()} probe versus {row['real_source']}: AUC {row['auc_mean']:.4f} ± {row['auc_std']:.4f}; test-ROC TPR at 5% FPR {row['tpr_at_test_roc_fpr_5pct_mean']:.4f} ± {row['tpr_at_test_roc_fpr_5pct_std']:.4f}.")
    lines += ['', 'The ± quantities are sample standard deviations over five splits, not confidence intervals. '
        'Test sources overlap across splits and the same Midjourney holdout is reused. '
        'Collection-specific performance and the fixed validation-threshold error rates are retained in the underlying reports.','',
        '## Artifacts','',
        '- `outputs/phase3/controls/unmarked_frozen_results.json`: unchanged-head comparisons and paired source intervals.',
        '- `outputs/phase3/controls/split_seeds/mean_std_metrics.csv`: all systems and ablations, separately for COCO and Open Images.',
        '- `outputs/phase3/controls/split_seeds/source_counts.csv`: source counts by generator and split for every seed.',
        '- `outputs/phase3/controls/split_seeds/complete.json`: checksums, software versions, source assignment, and original-head preservation.',
        '- `outputs/phase3/controls/summary.json`: compact machine-readable report.','']
    (ROOT/'docs/phase3_controls.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'summary':str(OUT/'summary.json'),'five_seed_rows':len(frame),'seeds':final['seeds']}),flush=True)


if __name__=='__main__':main()
