"""Inspect saved measurements and the final PDF; never fit or infer models."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

def rows(relative):
    with (ROOT / relative).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))

def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): value.update(block)
    return value.hexdigest()

def verify_hashes(base, mapping):
    for relative, expected in mapping.items():
        assert digest(base/relative)==expected, relative

def check_mean_std(records, summaries, keys, metrics):
    grouped={}
    for row in records:
        grouped.setdefault(tuple(row[key] for key in keys),[]).append(row)
    for row in summaries:
        values=grouped[tuple(row[key] for key in keys)]
        assert len(values)==5
        for metric in metrics:
            numbers=np.array([float(value[metric]) for value in values])
            assert math.isclose(float(row[metric+'_mean']),numbers.mean(),abs_tol=1e-12)
            assert math.isclose(float(row[metric+'_std']),numbers.std(ddof=1),abs_tol=1e-12)

def check_interval(value, bounded=True):
    assert len(value)==2 and all(math.isfinite(float(number)) for number in value)
    assert float(value[0])<=float(value[1])
    if bounded: assert 0<=float(value[0])<=float(value[1])<=1

def check_phase4():
    """Validate saved new evidence, independent of paper compilation/layout."""
    location=ROOT/'outputs/phase4/size_controlled'
    size=read('outputs/phase4/size_controlled/complete.json')
    protocol=read('outputs/phase4/size_controlled/protocol.json')
    amendment=read('outputs/phase4/size_controlled/jpeg_protocol_amendment.json')
    variants={'full','clean_repeat','no_matched_reals','jpeg_only'}
    assert set(size['variants'])==variants and size['fit_count']==40 and size['seed_count']==5
    assert sorted(size['backbones'])==['clip','resnet']
    assert size['all_converged'] and size['earlier_heads_and_inputs_unchanged']
    assert size['protocol_frozen_before_all_new_fits'] and size['amendment_frozen_before_JPEG_fits']
    assert size['protocol_sha256']==digest(location/'protocol.json')
    assert size['amendment_sha256']==digest(location/'jpeg_protocol_amendment.json')
    assert size['initial_script_snapshot_sha256']==protocol['script_sha256']==digest(location/'initial_script_snapshot.py')
    verify_hashes(ROOT,protocol['input_sha256'])
    verify_hashes(ROOT,protocol['preserved_head_sha256'])
    verify_hashes(location,size['output_sha256'])
    fits=read('outputs/phase4/size_controlled/convergence.json')
    assert len(fits)==40 and len({(r['seed'],r['backbone'],r['variant']) for r in fits})==40
    original_counts={r['seed']:r for r in protocol['counts']}
    for result in fits:
        assert result['converged'] and all(int(n)<result['iteration_cap'] for n in result['n_iter'])
        cutoff=amendment['frozen_at_utc'] if result['variant']=='jpeg_only' else protocol['frozen_at_utc']
        assert cutoff<result['completed_at_utc']
        estimated_fit_start=datetime.fromisoformat(result['completed_at_utc'])-timedelta(seconds=result['fit_seconds'])
        assert datetime.fromisoformat(cutoff)<estimated_fit_start
        counts=original_counts[result['seed']]
        reduced=result['variant']=='no_matched_reals'
        assert result['rows']==counts['no_matched_reals_rows' if reduced else 'full_rows']
        assert result['class_counts']==counts['reduced_class_counts' if reduced else 'full_class_counts']
        head=location/str(result['seed'])/f"{result['backbone']}_{result['variant']}.npz"
        assert digest(head)==result['model_sha256']
        for label in ['0','1']:
            expected=result['rows']/(2*result['class_counts'][label])
            assert math.isclose(result['balanced_class_weights'][label],expected,abs_tol=1e-12)
        selected=json.loads(head.with_name(head.stem+'_threshold.json').read_text(encoding='utf-8'))
        assert 0<=selected['validation_fpr']<=.05 and math.isfinite(selected['logit_threshold'])
    records=rows('outputs/phase4/size_controlled/per_seed_metrics.csv')
    summaries=rows('outputs/phase4/size_controlled/mean_std_metrics.csv')
    assert len(records)==120 and len(summaries)==24
    measures=['auc','tpr_at_validation_fpr5','test_fpr_at_validation_fpr5','tpr_at_test_roc_fpr5']
    check_mean_std(records,summaries,['backbone','variant','real_source'],measures)
    assert all(int(row['test_sources'])==1843 and int(row['test_rows'])==49158 for row in records if row['real_source']=='overall')
    differences=rows('outputs/phase4/size_controlled/paired_source_bootstrap.csv')
    assert len(differences)==144
    for row in differences:
        check_interval([row['ci95_low'],row['ci95_high']],bounded=False)
        if row['seed']=='five_seed_mean':
            other=row['contrast'].removeprefix('full_minus_')
            selected=[value for value in records if value['backbone']==row['backbone'] and value['real_source']=='overall']
            full={value['seed']:float(value[row['metric']]) for value in selected if value['variant']=='full'}
            control={value['seed']:float(value[row['metric']]) for value in selected if value['variant']==other}
            estimate=np.mean([full[seed]-control[seed] for seed in full])
            assert math.isclose(float(row['estimate']),estimate,abs_tol=1e-12)

    domain_root=ROOT/'outputs/phase4/domain_generalization'
    domain=read('outputs/phase4/domain_generalization/summary.json')
    domain_complete=read('outputs/phase4/domain_generalization/complete.json')
    assert domain_complete['original_heads_unchanged'] and not domain_complete['thresholds_selected_on_test']
    assert domain_complete['diagnostic_systems']==18 and len(domain['models'])==6
    assert domain['test_sources']==1705 and domain['test_rows']==46950
    assert domain['test_real_sources_each_collection']==150 and domain['test_AI_sources']==1405
    verify_hashes(domain_root,domain_complete['head_sha256'])
    verify_hashes(domain_root,domain_complete['result_sha256'])
    scope=rows('outputs/phase4/domain_generalization/source_assignments.csv')
    regimes=['coco_only','openimages_only','joint']
    common_test=None
    for regime in regimes:
        roles={role:{row['source_id'] for row in scope if row['regime']==regime and row['role']==role}
               for role in ['train','fusion_fit','calibration','test']}
        for left in roles:
            for right in roles:
                if left!=right: assert not roles[left].intersection(roles[right])
        if common_test is None: common_test=roles['test']
        else: assert roles['test']==common_test
        for role,expected_split in [('train','train'),('fusion_fit','val'),('calibration','val'),('test','test')]:
            assert all(row['locked_split']==expected_split for row in scope if row['regime']==regime and row['role']==role)
        for role,count in [('train',700),('fusion_fit',75),('calibration',75)]:
            reals=[row for row in scope if row['regime']==regime and row['role']==role and row['label']=='0']
            assert len(reals)==len({row['source_id'] for row in reals})==count
            if regime!='joint': assert {row['collection'] for row in reals}=={regime.removesuffix('_only')}
    assert {(model['backbone'],model['regime']) for model in domain['models'].values()}=={(name,regime) for name in ['clip','resnet'] for regime in regimes}
    for model in domain['models'].values():
        assert model['thresholds']['real_sources']==75
        for system in ['probe','fusion','gated']:
            cutoff=model['thresholds']['systems'][system]
            measured=model['systems'][system]
            assert 0<=cutoff['calibration_fpr']<=.05 and not measured['test_threshold_retuned']
            assert cutoff['threshold_logit']==measured['threshold_logit']
            for key in ['auc_ci95','test_fpr_ci95','test_tpr_ci95']: check_interval(measured[key])
    frozen=read('outputs/phase4/domain_generalization/frozen_seed_thresholds.json')
    verify_hashes(ROOT,frozen['head_sha256'])
    assert len(frozen['head_sha256'])==4 and len(frozen['systems'])==6
    assert frozen['scope_sha256']==digest(domain_root/'frozen_calibration_scope.csv')
    assert all(0<=system['validation_fpr']<=.05 for system in frozen['systems'].values())

    watermark_root=ROOT/'outputs/phase4/watermark_reporting'
    watermark=read('outputs/phase4/watermark_reporting/analysis_audit.json')
    assert watermark['all_checks_passed'] and watermark['no_new_inference_or_test_calibration']
    assert watermark['measurement_rows']==watermark['saved_fixed_rule_verified_flags_recomputed']==42432
    assert watermark['transformed_covers']==21216 and watermark['covers']==1632
    assert watermark['source_counts']=={'photograph':204,'unmarked_ai':204}
    assert watermark['conditions']==13 and watermark['decoders_per_transformed_cover']==2
    assert watermark['analysis_plan_sha256']==digest(watermark_root/'analysis_plan.json')
    verify_hashes(ROOT,watermark['input_sha256'])
    verify_hashes(watermark_root,watermark['outputs_sha256'])
    table=rows('outputs/phase4/watermark_reporting/table_ready.csv')
    assert len(table)==14 and all(int(row['n_sources_per_origin'])==204 for row in table)
    assert {row['condition'] for row in table}=={'identity','jpeg_s2','crop_s1','lowres_s2','regenerate_s2','spectral_s1','overall_equal_condition_mean'}

    fresh_root=ROOT/'outputs/phase4/fresh_replication'
    fresh_protocol=read('outputs/phase4/fresh_replication/protocol.json')
    fresh=read('outputs/phase4/fresh_replication/summary.json')
    selected=read('outputs/phase4/fresh_replication/selection_complete.json')
    derived=read('outputs/phase4/fresh_replication/derivation_complete.json')
    exclusions=read('outputs/phase4/fresh_replication/native_exclusion_audit.json')
    assert fresh['source_count']==selected['sources']==400 and fresh['images']==derived['images']==2560
    assert not fresh['models_refitted'] and not fresh['thresholds_retuned'] and fresh['model_hashes_unchanged']
    assert fresh_protocol['test_retuning'] is False and fresh_protocol['test_inference_passes']==1
    assert fresh_protocol['explicit_native_id_exclusions'] and fresh_protocol['calibration_head_hashes_verified']
    assert selected['prior_native_id_intersections']==0 and selected['exact_or_ahash_overlap_screen_passed']
    assert selected['detector_inference_performed'] is False
    assert selected['native_exclusion_audit_sha256']==digest(fresh_root/'native_exclusion_audit.json')
    assert exclusions['excluded_ids_sha256']==digest(fresh_root/'excluded_native_ids.csv')
    assert exclusions['Dalle_original_eligible_exports_replayed']==exclusions['Midjourney_original_eligible_exports_replayed']==1000
    assert fresh_protocol['frozen_at_utc']<selected['completed_at_utc']<derived['completed_at_utc']
    assert frozen['locked_at_utc']<=fresh_protocol['frozen_at_utc']
    assert fresh['protocol_sha256']==selected['protocol_sha256']==digest(fresh_root/'protocol.json')
    assert selected['manifest_sha256']==digest(fresh_root/'sources.csv')
    assert fresh['manifest_sha256']==derived['manifest_sha256']==digest(fresh_root/'manifest.csv')
    assert fresh['calibration_sha256']==fresh_protocol['calibration_sha256']==digest(domain_root/'frozen_seed_thresholds.json')
    assert fresh['bootstrap_draws_sha256']==digest(fresh_root/'bootstrap_draws.npz')
    expected={Path(key).name:value for key,value in frozen['head_sha256'].items()}
    assert fresh_protocol['model_hashes']==expected and len(expected)==4
    verify_hashes(ROOT/'outputs/phase3/controls/split_seeds/20260920',expected)
    source_rows=rows('outputs/phase4/fresh_replication/sources.csv')
    manifest_rows=rows('outputs/phase4/fresh_replication/manifest.csv')
    assert len(source_rows)==len({row['source_id'] for row in source_rows})==400
    assert Counter(row['collection'] for row in source_rows)==Counter({name:80 for name in ['coco','openimages','stable-diffusion','dalle-3','midjourney']})
    assert len(manifest_rows)==len({row['image_id'] for row in manifest_rows})==2560
    assert set(row['source_id'] for row in manifest_rows)=={row['source_id'] for row in source_rows}
    per_source=Counter(row['source_id'] for row in manifest_rows)
    assert all(per_source[row['source_id']]==(8 if row['label']=='1' else 4) for row in source_rows)
    prior_ids={(row['collection'],row['original_id']) for row in rows('outputs/phase4/fresh_replication/excluded_native_ids.csv')}
    assert not prior_ids.intersection((row['collection'],row['original_id']) for row in source_rows)
    records=fresh['measurements']
    assert len(records)==60 and len(fresh['paired_comparisons'])==60
    predictions=rows('outputs/phase4/fresh_replication/predictions.csv')
    assert len(predictions)==2560 and {row['image_id'] for row in predictions}=={row['image_id'] for row in manifest_rows}
    for row in predictions:
        for name in ['resnet','clip']:
            chosen=name+('_fusion' if float(row['active_score'])>=13/16 else '_probe')
            assert float(row[name+'_gated'])==float(row[chosen])
    lookup={}
    for measured in records:
        assert measured['threshold_logit']==frozen['systems'][measured['system']]['threshold_logit']
        for key in ['auc_ci95','test_TPR_ci95','test_FPR_ci95']: check_interval(measured[key])
        subset=[row for row in predictions if (row['label']=='0' or row['mark']==measured['AI_mark'])
                and (measured['attack']=='overall' or row['attack']==measured['attack'])]
        y=np.array([int(row['label']) for row in subset])
        scores=np.array([float(row[measured['system']]) for row in subset])
        accepted=scores>=measured['threshold_logit']
        assert measured['sources']==400 and measured['images']==len(subset)==(1600 if measured['attack']=='overall' else 400)
        assert math.isclose(measured['auc'],roc_auc_score(y,scores),abs_tol=1e-12)
        assert math.isclose(measured['test_TPR'],accepted[y==1].mean(),abs_tol=1e-12)
        assert math.isclose(measured['test_FPR'],accepted[y==0].mean(),abs_tol=1e-12)
        lookup[(measured['attack'],measured['AI_mark'],measured['system'])]=measured
    for comparison in fresh['paired_comparisons']:
        assert comparison['paired_source_draws']
        left,right=comparison['comparison'].split('-minus-')
        key=comparison['attack'],comparison['AI_mark']
        left=lookup[(*key,comparison['backbone']+'_'+left)]
        right=lookup[(*key,comparison['backbone']+'_'+right)]
        for measure,field in [('auc','auc_difference'),('test_TPR','fixed_threshold_TPR_difference'),('test_FPR','fixed_threshold_FPR_difference')]:
            assert math.isclose(comparison[field],left[measure]-right[measure],abs_tol=1e-12)
            check_interval(comparison[field+'_ci95'],bounded=False)
    started=[]
    for name in ['resnet','clip']:
        inference=read(f'outputs/phase4/fresh_replication/{name}_inference.json')
        assert inference['images']==2560 and inference['manifest_sha256']==fresh['manifest_sha256']
        assert inference['embedding_sha256']==digest(fresh_root/f'{name}_embeddings.npz')
        timestamp=inference.get('test_inference_started_at_utc',inference.get('inference_started_at_utc'))
        if timestamp is not None:
            assert selected['completed_at_utc']<timestamp<=inference['completed_at_utc']
            started.append(timestamp)
        else:
            assert selected['completed_at_utc']<inference['completed_at_utc']
        assert inference['completed_at_utc']<=fresh['completed_at_utc']
    return {'size_controlled_fits':40,'size_controlled_variants':4,'size_controlled_split_seeds':5,
            'domain_probe_fits':6,'domain_fusion_fits':6,'independent_domain_calibration_sources_per_regime':75,
            'watermark_fixed_flags_rechecked':42432,'watermark_control_sources_each_origin':204,
            'new_replication_sources':400,'new_replication_images':2560,'new_replication_frozen_heads':4,
            'new_replication_native_ID_intersections':0,'new_replication_test_start_times_recorded':len(started),
            'phase4_protocols_hashes_and_saved_intervals_checked':True}

def resource_fonts(resources):
    """Also inspect embedded fonts inside imported scientific PDF figures."""
    if resources is None: return
    resources=resources.get_object()
    font_map=resources.get('/Font',{})
    if hasattr(font_map,'get_object'): font_map=font_map.get_object()
    for reference in font_map.values():
        yield reference.get_object()
    objects=resources.get('/XObject',{})
    if hasattr(objects,'get_object'): objects=objects.get_object()
    for reference in objects.values():
        obj=reference.get_object()
        if obj.get('/Subtype')=='/Form':
            yield from resource_fonts(obj.get('/Resources'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visual-reviewed', action='store_true',
                        help='Record only after inspecting all final rendered pages.')
    args = parser.parse_args()
    sources = rows('outputs/conference_clean/manifest_sources.csv')
    kept = {r['source_id'] for r in sources}
    assert len(sources) == len(kept) == 5823
    pairs = rows('outputs/conference_clean/duplicate_pairs.csv')
    assert not any(p['left_source_id'] in kept and p['right_source_id'] in kept for p in pairs)
    variants = rows('outputs/conference_clean/manifest_derived.csv')
    assert len(variants) == 80739
    test = [r for r in variants if r['split'] == 'test']
    assert len(test) == 26880 and len({r['source_id'] for r in test}) == 1680
    assert set(Counter(r['source_id'] for r in test).values()) == {16}
    assert all(r['split'] == 'test' for r in sources if r['generator'] == 'midjourney')
    assert read('outputs/extension/presentation/table_i_verification.json')['table_i_matches_independent_manifests']

    gate = rows('outputs/extension/hard_gate/test_predictions.csv')
    gate_results = read('outputs/extension/hard_gate/results.json')
    for name, prefix, gprefix in [('resnet', '', 'resnet_'), ('clip', 'clip_', 'clip_')]:
        for r in gate:
            enabled = float(r['active_score']) >= 13 / 16
            chosen = prefix + ('fusion' if enabled else 'probe')
            assert float(r[gprefix + 'gated']) == float(r[chosen])
            assert float(r[gprefix + 'gated_logit']) == float(r[chosen + '_logit'])
        auc = roc_auc_score([int(r['label']) for r in gate], [float(r[gprefix+'gated_logit']) for r in gate])
        assert math.isclose(auc, gate_results[name]['full']['gated']['auc'], abs_tol=1e-12)
        assert len(gate_results[name]['clean_marked_prevalence']) == 11

    fresh = read('outputs/extension/fresh/metrics.json')
    protocol = read('outputs/extension/fresh/protocol.json')
    selected = read('outputs/extension/fresh/selection_complete.json')
    assert protocol['frozen_at_utc'] < selected['completed_at_utc'] < fresh['inference_started_at_utc']
    assert not fresh['models_refitted'] and not fresh['thresholds_retuned']
    assert protocol['inference_passes_allowed'] == 1
    assert fresh['rows'] == fresh['sources'] == 192
    assert selected['manifest_sha256'] == fresh['manifest_sha256'] == digest(ROOT/'outputs/extension/fresh/manifest.csv')
    assert not read('outputs/extension/fresh/archive_identity_audit.json')['original_archive_id_overlap']
    for name, expected in protocol['models_sha256'].items():
        assert digest(ROOT/'outputs/conference_clean'/name) == expected, name
    fresh_rows = rows('outputs/extension/fresh/predictions.csv')
    assert len({r['source_id'] for r in fresh_rows}) == 192
    assert not kept.intersection(r['source_id'] for r in fresh_rows)
    for prefix in ['', 'clip_']:
        for r in fresh_rows:
            chosen = prefix + ('fusion' if float(r['active_score']) >= 13/16 else 'probe')
            assert float(r[prefix+'gated_logit']) == float(r[chosen+'_logit'])
            # Original probabilities are float32; CSV formats the selected
            # float64 column more precisely. Compare at original dtype.
            assert np.float32(r[prefix+'gated']) == np.float32(r[chosen])
    real = [r for r in fresh_rows if r['label'] == '0']
    policy = read('outputs/extension/hard_gate/policy.json')['selected']
    positive = [r for r in real if float(r['active_score']) >= 13/16 or float(r['gated']) >= .5+policy['margin']]
    assert len(real) == 96 and len(positive) == 9
    assert all(r['generator'] == 'openimages' for r in positive)

    covers = rows('outputs/extension/watermarks/base_covers.csv')
    measured = rows('outputs/extension/watermarks/measurements.csv')
    assert len(covers) == 192 and len({r['source_id'] for r in covers}) == 48
    assert len(measured) == 5568
    assert len({(r['cover_id'], r['condition']) for r in measured}) == 2784
    assert len({r['condition'] for r in measured}) == 16
    diffusion = read('outputs/extension/diffusion/complete.json')
    assert diffusion['outputs'] == 288
    assert diffusion['manifest_sha256'] == digest(ROOT/'outputs/extension/diffusion/manifest.csv')
    exact = read('outputs/extension/watermarks/source_verification_intervals.json')
    assert len(exact['measurements']) == 64
    assert exact['measurements_sha256'] == digest(ROOT/'outputs/extension/watermarks/measurements.csv')
    prevalence = read('outputs/extension/attack_prevalence/prevalence.json')
    assert prevalence['selected_counterfactual_rows'] == 1152
    assert prevalence['archived_inference_rows'] == 1776
    assert prevalence['source_count'] == 48 and len(prevalence['conditions']) == 16
    assert all(len(values) == 22 for values in prevalence['conditions'].values())

    tex = (ROOT/'manuscript/main.tex').read_text(encoding='utf-8')
    assert '{{' not in tex and r'\IEEEauthorblockN{Anonymous Authors}' in tex
    references = re.findall(r'\\bibitem\{([^}]+)\}', tex)
    first_cites = list(dict.fromkeys(k for group in re.findall(r'\\cite\{([^}]+)\}', tex) for k in group.split(',')))
    assert first_cites == references and len(references) == 22
    assert tex.count(r'\begin{table}') == 5
    assert r'\label{fig:pipeline}' in tex and r'\label{fig:prevalence}' in tex
    assert r'\label{fig:roc}' in tex
    assert tex.count(r'\begin{figure}')+tex.count(r'\begin{figure*}')==3
    assert 'OpenAI Codex assisted' in tex and '0.437 [0.401, 0.473]' in tex
    assert 'C:' not in tex and 'local_user' not in tex.lower()
    log = (ROOT/'manuscript/main.log').read_text(encoding='utf-8')
    assert 'Overfull' not in log
    assert not re.search(r'(Citation|Reference).*undefined', log)
    pdf = ROOT/'manuscript/main.pdf'
    reader = PdfReader(pdf)
    assert len(reader.pages) == 5 and not reader.is_encrypted
    fonts = set()
    for page in reader.pages:
        assert tuple(float(x) for x in page.mediabox) == (0., 0., 612., 792.)
        for font in resource_fonts(page['/Resources']):
            if '/DescendantFonts' in font:
                font = font['/DescendantFonts'][0].get_object()
            descriptor = font.get('/FontDescriptor')
            assert font.get('/Subtype') == '/Type3' or (descriptor is not None and any(
                key in descriptor.get_object() for key in ['/FontFile','/FontFile2','/FontFile3']))
            fonts.add(str(font.get('/BaseFont', '')))
    assert any('NimbusRom' in name or 'Times' in name for name in fonts)
    # New evidence is kept separate from the earlier 24-pair analysis.
    controls=read('outputs/phase3/controls/unmarked_frozen_results.json')
    assert controls['actual_unmarked_edit_rows']==19544 and controls['expanded_test_rows']==46424
    assert not controls['heads_refitted'] and not controls['thresholds_retuned']
    seeds=read('outputs/phase3/controls/split_seeds/complete.json')
    assert seeds['test_rows_per_seed']==49158 and len(seeds['seeds'])==5
    assert sorted(seeds['completed_backbones'])==['clip','resnet']
    repeated=rows('outputs/phase3/controls/split_seeds/per_seed_metrics.csv')
    means=rows('outputs/phase3/controls/split_seeds/mean_std_metrics.csv')
    assert len(repeated)==285 and len(means)==57
    grouped={}
    for r in repeated:
        key=tuple(r[k] for k in ['backbone','variant','system','real_source'])
        grouped.setdefault(key,[]).append(r)
    for r in means:
        key=tuple(r[k] for k in ['backbone','variant','system','real_source'])
        block=grouped[key]; assert len(block)==5
        for metric in ['auc','tpr_at_test_roc_fpr_5pct','false_positive_rate','true_positive_rate']:
            nums=np.array([float(v[metric]) for v in block])
            assert math.isclose(float(r[metric+'_mean']),nums.mean(),abs_tol=1e-12)
            assert math.isclose(float(r[metric+'_std']),nums.std(ddof=1),abs_tol=1e-12)
    expanded_covers=rows('outputs/phase3/watermarks/base_covers.csv')
    assert len(expanded_covers)==1632 and len({r['source_id'] for r in expanded_covers})==408
    local=read('outputs/phase3/watermarks/analysis_provenance.json')
    assert local['outputs']==21216 and local['rows']==42432
    expanded=read('outputs/phase3/watermarks/summary.json')
    for r in expanded['measurements']:
        if r['generator']=='all': assert r['n_sources']==204
    regen=read('outputs/phase3/regeneration/summary.json')
    assert regen['source_count']==204 and regen['outputs']==1224 and regen['generated_illustration_excluded']
    assert all(r['n_sources']==204 for r in regen['measurements'])
    assert regen['manifest_sha256']==digest(ROOT/'outputs/phase3/regeneration/manifest.csv')
    native=read('outputs/phase3/native_baseline/metrics.json')
    assert native['no_test_tuning'] and not native['cause_of_failures_confirmed']
    example=read('manuscript/assets/example_measurements.json')
    assert example['excluded_from_quantitative_benchmark'] and len(example['images'])==6
    for r in example['images']:
        assert len(r['bits'])==16
        assert r['agreement_count']==sum(a==b for a,b in zip(r['bits'],example['payload16']))
    abstract=tex.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
    assert len(re.findall(r'\d+(?:\.\d+)?',abstract))<=5
    phase4=check_phase4()
    report = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'venue': 'the conference submission', 'status': 'Anonymous five-page draft; not submitted',
        'pdf_pages': len(reader.pages), 'letter_size': True, 'fonts_embedded': True,
        'references': 22, 'tables': 5, 'figures': 3, 'no_overfull_or_unresolved_references': True,
        'table_i_verified': True, 'gate_exact_fallback': True,
        'fresh_sources': 192, 'frozen_model_hashes_match': len(protocol['models_sha256']),
        'fresh_real_errors': 9, 'fresh_real_sources': 96,
        'watermark_attack_outputs': 2784, 'distinct_watermark_sources': 48,
        'diffusion_outputs': 288, 'source_level_exact_intervals': 64,
        'attack_prevalence_selected_rows': 1152,
        'unmarked_AI_edits':19544,'fixed_expanded_test_rows':46424,'repeated_split_test_rows':49158,
        'split_seeds':5,'additional_Open_Images_sources':1000,'expanded_watermark_AI_sources':204,
        'expanded_local_watermark_outputs':21216,'expanded_regeneration_outputs':1224,
        'native_UFD_auc':native['overall']['native_auc'],
        **phase4,
        'abstract_numeric_values':len(re.findall(r'\d+(?:\.\d+)?',abstract)),
        'visual_review': 'All five final rendered pages inspected for clipping, overlap and legibility.' if args.visual_reviewed else 'Pending final rendered-page inspection.',
        'sha256': {name: digest(ROOT/name) for name in [
            'manuscript/main.tex','manuscript/main.pdf','manuscript/conference_template.tex',
            'manuscript/conference_bibliography.txt','outputs/extension/fresh/metrics.json',
            'outputs/extension/watermarks/summary.json','outputs/extension/attack_prevalence/prevalence.json',
            'outputs/phase3/controls/split_seeds/mean_std_metrics.csv','outputs/phase3/watermarks/summary.json',
            'outputs/phase3/regeneration/summary.json','outputs/phase3/native_baseline/metrics.json',
            'outputs/phase4/size_controlled/complete.json','outputs/phase4/size_controlled/mean_std_metrics.csv',
            'outputs/phase4/size_controlled/paired_source_bootstrap.csv','outputs/phase4/domain_generalization/summary.json',
            'outputs/phase4/watermark_reporting/summary.json','outputs/phase4/fresh_replication/summary.json',
            'manuscript/assets/pipeline_examples.pdf','manuscript/assets/resnet_roc.pdf']}}
    (ROOT/'outputs/extension/submission_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='sha256'},indent=2))

if __name__ == '__main__':
    main()
