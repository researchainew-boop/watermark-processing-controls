"""Paper tokens drawn only from completed, separately scoped Phase 3 evidence."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
def f(x,n=3): return f'{float(x):.{n}f}'
def ci(x,n=3): return ', '.join(f(v,n) for v in x)
def rows(name):
    with (ROOT/name).open(encoding='utf-8',newline='') as handle: return list(csv.DictReader(handle))

def values(old):
    controls=read('outputs/phase3/controls/unmarked_frozen_results.json')
    complete=read('outputs/phase3/controls/split_seeds/complete.json')
    assert complete['test_rows_per_seed']==49158 and sorted(complete['completed_backbones'])==['clip','resnet']
    repeated=rows('outputs/phase3/controls/split_seeds/mean_std_metrics.csv')
    index={(r['backbone'],r['variant'],r['system'],r['real_source']):r for r in repeated}
    def metric(model,variant='full',system='probe',real='overall'):
        return index[model,variant,system,real]
    def mean_sd(r,key='auc'): return '$'+f(r[key+'_mean'])+r'\pm'+f(r[key+'_std'])+'$'
    main=[]
    for model,label in [('active','Payload'),('resnet','ResNet'),('clip','CLIP')]:
        for system in (['payload'] if model=='active' else ['probe','fusion','gated']):
            m=metric(model,system=system)
            main.append(' & '.join([label+('' if model=='active' else ' '+{'gated':'gate'}.get(system,system)),
                mean_sd(m),mean_sd(m,'tpr_at_test_roc_fpr_5pct'),mean_sd(m,'false_positive_rate')])+r'\\')
    ablate=[]
    for variant,label in [('full','Full'),('no_attacks','No attacks'),('no_matched_reals','No matched real')]:
        rm=metric('resnet',variant); cm=metric('clip',variant)
        ablate.append(' & '.join([label,mean_sd(rm),mean_sd(rm,'tpr_at_test_roc_fpr_5pct'),mean_sd(cm)])+r'\\')
    u=controls['expanded_test_overall']; effects=controls['paired_AI_probe_effects']['resnet']
    unmarked=r'''With the original heads fixed, adding unmarked AI edits gives ResNet
probe/fusion/gate AUC '''+'/'.join(f(u['resnet_'+s]['auc']) for s in ['probe','fusion','gated'])+r''' and CLIP
'''+ '/'.join(f(u['clip_'+s]['auc']) for s in ['probe','fusion','gated'])+r'''.
The paired ResNet marked-minus-unmarked probability change averages
'''+f(effects['mean_probability_marked_minus_unmarked'],4)+r''' [
'''+ci(effects['mean_probability_delta_ci95'],4)+r''']; validation-threshold TPR change is
'''+f(effects['TPR_marked_minus_unmarked_at_original_validation_threshold'],4)+r''' [
'''+ci(effects['TPR_delta_ci95'],4)+r''']. The mark has a small measurable probability
effect, but no clear average TPR effect under these edits.
'''
    methods=r'''\subsection{Experiments 2 and 3}
Experiment~2 uses 300 approximately CLIP-matched COCO/AI pairs:
120 validation and 180 test, 60 test pairs per AI collection. Forty-nine
pairs below a training-derived similarity caliper remain included.
Both origins receive common PNG or OpenCV JPEG 95/75/50, with heads fixed.
The fresh sample contains 48 unused COCO, 48 Open Images, and 96 unused
DALL$\cdot$E~3 archive images. Selection follows saved SHA-256 ordering
and hash screening. All are unmarked, area-resized to $256\times256$,
and JPEG-95 encoded. The study's heads and policy are frozen before
selection and scored once. The unchanged published UniversalFakeDetect
ViT-L/14 head uses official center-crop 224, normalization, raw CLIP
features, and logit threshold zero. A paired follow-up also evaluates its
same fresh sources at native resolution, without the common JPEG round trip.

Experiment~3 uses 204 test AI sources (68 per collection), selected by
seeded ID-hash order, and 204 distinct COCO controls assigned by maximum
total CLIP cosine similarity. Three balanced 16-bit payloads are embedded
using DWT--DCT or official TrustMark 0.9.2 Q, strength 1, BCH5. TrustMark
uses a 61-bit message: the payload plus 45 fixed zeros; verification
requires the full expected message and valid schema/ECC. Ten shared
conditions plus three Gaussian-noise seeds at $\sigma=6$ are applied to
both marks and unmarked controls; unmarked controls are queried with
the original expected payload. Regeneration tests both marks on all
204 AI sources with the original payload: posterior-mean VAE round trip
and DDIM image-to-image strengths 0.25/0.50 using a cached Realistic Vision
V5.1 SD1.5-derived checkpoint, twenty configured steps (5/10 effective),
empty prompt, guidance 1, seed 20261009, and batch one. PSNR/SSIM accompany
recovery; no verification threshold is retuned.
'''
    enc=read('outputs/extension_controls/encoding/encoding_metrics.json')['overall']['resnet_full']
    enc_clip=read('outputs/extension_controls/encoding/encoding_metrics.json')['overall']['clip_full']
    published=read('outputs/extension/published_baseline/metrics.json')
    fresh=read('outputs/extension/fresh/metrics.json')
    native=read('outputs/phase3/native_baseline/metrics.json')['overall']
    results=r'''\subsection{Experiment 2: encoding and fresh sample}
ResNet AUC on common PNG and JPEG 95/75/50 is '''+'/'.join(f(enc[c]['roc_auc']) for c in ['common_png','jpeg_q95','jpeg_q75','jpeg_q50'])+r'''.
CLIP AUC is '''+'/'.join(f(enc_clip[c]['roc_auc']) for c in ['common_png','jpeg_q95','jpeg_q75','jpeg_q50'])+r''' on the same 180 pairs;
published UniversalFakeDetect ranges from '''+f(min(published['encoding_'+c]['univfd']['auc'] for c in ['common_png','jpeg_q95','jpeg_q75','jpeg_q50']))+' to '+f(max(published['encoding_'+c]['univfd']['auc'] for c in ['common_png','jpeg_q95','jpeg_q75','jpeg_q50']))+r'''.
ResNet's JPEG-95 change is '''+f(enc['jpeg_q95']['auc_change_from_common_png']['estimate'],4)+r''' [
'''+ci(enc['jpeg_q95']['auc_change_from_common_png']['ci95'],4)+r''']; at the validation threshold,
real FPR increases from '''+f(enc['common_png']['original_fixed']['fpr'])+' to '+f(enc['jpeg_q50']['original_fixed']['fpr'])+r''' under JPEG 50.
On the fresh 192 sources, ResNet probe/gate AUC is '''+f(fresh['overall']['resnet_probe']['auc'])+'/'+f(fresh['overall']['resnet_gated']['auc'])+r''' and CLIP probe AUC
is '''+f(fresh['overall']['clip_probe']['auc'])+r'''. The frozen policy covers '''+f(fresh['frozen_policy']['coverage'])+r''' and accepts
9/96 real inputs: all nine are Open Images (9/48), versus COCO 0/48.
Published UniversalFakeDetect AUC is '''+f(native['native_auc'])+r''' [
'''+ci(native['native_auc_ci95'])+r'''] on native inputs versus '''+f(native['common256_jpeg95_auc'])+r''' after common
resizing/JPEG; paired difference is '''+f(native['paired_auc_difference'])+r''' [
'''+ci(native['paired_auc_difference_ci95'])+r''']. We retain this baseline without reversing its scores.

\subsection{Experiment 3: learned mark}
'''
    marks=read('outputs/phase3/watermarks/summary.json')
    regen=read('outputs/phase3/regeneration/summary.json')
    assert regen['source_count']==204 and regen['outputs']==1224
    selected={(r['condition'],r['detector']):r for r in marks['measurements']
              if r['generator']=='all' and r['input_origin_label']==1}
    learned={(r['condition'],r['watermark_method']):r for r in regen['measurements']}
    covers=rows('outputs/phase3/watermarks/base_covers.csv')
    embedding={m:sum(float(r['psnr']) for r in covers if r['watermark_method']==m)/sum(r['watermark_method']==m for r in covers)
               for m in ['dwt_dct','trustmark']}
    markrows=[]
    for cond,label in [('identity','Identity'),('jpeg_s2','JPEG 50'),('crop_s1','Crop'),('lowres_s2','Low-res 32'),
                       ('regenerate_s2','Grid shift'),('spectral_s1','Spectral wipe'),('vae_only','VAE only'),
                       ('diffusion_strength0p25','Diffusion 0.25'),('diffusion_strength0p50','Diffusion 0.50')]:
        rate=[]; quality=[]
        for method in ['dwt_dct','trustmark']:
            if cond in ['vae_only','diffusion_strength0p25','diffusion_strength0p50']:
                r=learned[cond,method]; rate.append(f(r['verification_rate'])); quality.append(f(r['mean_psnr'],1))
            else:
                r=selected[cond,method]; assert r['n_sources']==204
                rate.append(f(r['expected_payload_match_rate'])); quality.append(f(embedding[method] if cond=='identity' else r['mean_psnr_vs_input_cover'],1))
        markrows.append(' & '.join([label,*rate,*quality])+r'\\')
    results+=r'''\begin{table}[t]
\caption{Verification and PSNR (dB) on 204 AI sources. D: $\ge13/16$
bits; T: exact 61-bit message and valid ECC. Identity/edits pool three
payloads; VAE/diffusion use one. Identity PSNR measures embedding;
other rows compare to the marked input.}
\label{tab:modern}\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}\toprule
Condition & D rate & T rate & D PSNR & T PSNR\\\midrule
'''+ '\n'.join(markrows)+r'''
\bottomrule\end{tabular}\end{table}
Table~\ref{tab:modern} separates compound grid shift from actual
generative reconstruction. At diffusion strength 0.50, DWT--DCT verifies
'''+str(learned['diffusion_strength0p50','dwt_dct']['verified'])+r'''/204 sources [
'''+ci(learned['diffusion_strength0p50','dwt_dct']['ci95'])+r'''] and TrustMark
'''+str(learned['diffusion_strength0p50','trustmark']['verified'])+r'''/204 [
'''+ci(learned['diffusion_strength0p50','trustmark']['ci95'])+r'''] (exact 95\% intervals).
Across three new noise seeds, DWT--DCT rates are '''+'/'.join(f(selected[f'noise_sigma6_seed{s}','dwt_dct']['expected_payload_match_rate']) for s in [20261009,20261010,20261011])+r''' and
TrustMark rates are '''+'/'.join(f(selected[f'noise_sigma6_seed{s}','trustmark']['expected_payload_match_rate']) for s in [20261009,20261010,20261011])+r'''.
Per-payload, collection, and real-control results are archived.
'''
    prevalence=old['PREVALENCE_RESULTS']
    figure=prevalence[prevalence.index(r'\begin{figure}'):prevalence.index(r'\end{figure}')+len(r'\end{figure}')]
    figure=figure.replace('height=4.2cm','height=3.4cm')
    gate=read('outputs/extension/hard_gate/results.json')['resnet']['clean_marked_prevalence']
    prevalence=r'''\subsection{Share of marked images}
For each original source, clean unmarked/marked outcomes receive weights
$1-\pi$/$\pi$, with real controls fixed. Figure~\ref{fig:prevalence}
reports eleven shares. ResNet gated AUC gain at 0/50/100\% marking is
'''+ '/'.join(f(gate[i]['gated_gain'],4) for i in [0,5,10])+r'''.
'''+figure
    return {'EXTENSION_METHODS':methods,'EXTENSION_RESULTS':results,'PREVALENCE_RESULTS':prevalence,
        'UNMARKED_EDIT_ROWS':f"{controls['actual_unmarked_edit_rows']:,}",'EXPANDED_ROWS':f"{controls['expanded_test_rows']:,}",
        'UNMARKED_RESULTS':unmarked,'REPEATED_MAIN_ROWS':'\n'.join(main),'REPEATED_ABLATION_ROWS':'\n'.join(ablate),
        'REPEATED_REAL_RESULTS':'ResNet probe AUC against COCO/Open Images is '+mean_sd(metric('resnet',real='coco'))+'/'+mean_sd(metric('resnet',real='openimages'))+'.',
        'REPEATED_ABLATION_RESULT':'ResNet TPR$_5$ falls from '+mean_sd(metric('resnet'),'tpr_at_test_roc_fpr_5pct')+' to '+mean_sd(metric('resnet','no_matched_reals'),'tpr_at_test_roc_fpr_5pct')+'.'}
