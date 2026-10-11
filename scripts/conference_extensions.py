"""Measured extension sections, kept separate from the original core numbers."""
import json
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
def f(x,n=3): return f'{x:.{n}f}'
def ci(x): return ', '.join(f(v) for v in x)

def values():
    gate=read('outputs/extension/hard_gate/results.json')
    fresh=read('outputs/extension/fresh/metrics.json')
    encoding=read('outputs/extension_controls/encoding/encoding_metrics.json')['overall']
    baseline=read('outputs/extension/published_baseline/metrics.json')
    marks=read('outputs/extension/watermarks/summary.json')
    diffusion=read('outputs/extension/diffusion/complete.json')
    attacked=read('outputs/extension/attack_prevalence/prevalence.json')
    assert diffusion['outputs']==288
    assert attacked['selected_counterfactual_rows']==1152
    with (ROOT/'outputs/extension/presentation/full_condition_table.csv').open() as handle:
        conditions={r['condition']:r for r in csv.DictReader(handle)}
    rg=gate['resnet']; cg=gate['clip']; dead=rg['reduced_recovery']
    with (ROOT/'outputs/extension/hard_gate/test_predictions.csv').open() as handle:
        gated_rows=list(csv.DictReader(handle))
    gate_main=[]
    for name,label in [('resnet','ResNet gated'),('clip','CLIP gated')]:
        ordered=sorted([(float(r[name+'_gated_logit']),int(r['label'])) for r in gated_rows],reverse=True)
        positives=sum(y for _,y in ordered); negatives=len(ordered)-positives
        import itertools
        tp=fp=0; tpr5=0.
        for _,group in itertools.groupby(ordered,key=lambda x:x[0]):
            for _,y in group: tp+=y; fp+=1-y
            if fp/negatives<=.05: tpr5=tp/positives
        metric=gate[name]['full']['gated']
        gate_main.append(' & '.join([label,f(metric['auc'])+' ['+ci(metric['ci95'])+']','---','---',f(tpr5)])+r'\\')
    labels=[('identity:0','Identity'),('jpeg:1','JPEG 70'),('jpeg:2','JPEG 50'),('blur:1','Blur, kernel 3'),('blur:2','Blur, kernel 5'),
        ('noise:1','Noise, $\\sigma=6$'),('resize:1','Resize, 80\\%'),('crop:1','Crop, 12 px/side'),('color:1','Color'),
        ('denoise:1','Denoise'),('inpaint:1','Inpaint'),('lowres:1','Low-res 64'),('lowres:2','Low-res 32$^\\star$'),
        ('regenerate:2','Grid shift$^\\star$'),('spectral:1','Spectral wipe$^\\star$')]
    weak_conditions={'crop:1','lowres:2','regenerate:2','spectral:1'}
    attacks=[]
    for key,label in labels:
        cells=[label]+[f(float(conditions[key][col])) for col in ['marked_mean_agreement','active_auc','probe_auc','fusion_auc','gated_auc']]
        if key in weak_conditions:
            cells=[r'\textbf{'+cell+'}' for cell in cells]
        attacks.append(' & '.join(cells)+r'\\')
    methods=r'''\subsection{Experiments 2 and 3}
Experiment~2 uses 300 approximately content-matched COCO/AI
pairs: 120 validation and 180 test pairs (60 test pairs per AI collection).
Pairs are selected by frozen CLIP cosine similarity; 49 fall below a
training-derived similarity caliper and remain included.
Both origins receive the same OpenCV JPEG round trip at quality 95, 75,
or 50; a common-PNG control preserves the prior pixels. All heads stay fixed.
An additional frozen sample contains 48 Open Images photographs
\cite{openimages}, 48 unused COCO photographs, and 96 unused DALL$\cdot$E~3
archive images. Candidates follow a saved SHA-256 order, with size and
cross-sample hash screening before inference. All receive area resizing
to $256\times256$ and identical JPEG quality 95. It is scored once after
the heads, gate, and decision policy are frozen; all images are unmarked.

The unchanged published UniversalFakeDetect ViT-L/14 head \cite{univfd}
uses its official center crop to 224, normalization, and unnormalized
CLIP features. Its Experiment~1 clean-original and Experiment~2 comparisons
use the same images as the study probes, with logit zero as the fixed
classification threshold.

Experiment~3 selects 24 test AI/real pairs, eight per AI
collection. Each AI receives three balanced 16-bit strings under DWT--DCT
or official TrustMark 0.9.2 Q, strength 1, BCH5. TrustMark carries the
16 bits plus 45 fixed zeros in its 61-bit data message; verification requires
the entire expected message and valid schema/ECC. Unmarked real controls
check the original expected string. Ten transformations and three extra
Gaussian-noise seeds ($\sigma=6$) are applied identically to paired covers.
The regeneration subset uses the original payload and unmarked controls:
96 covers, 48 input sources. A cached Realistic Vision V5.1 SD1.5-derived
checkpoint applies DDIM image-to-image denoising at strengths 0.25/0.50,
20 configured steps (5/10 effective denoising steps), seed 20261009, empty
prompt, guidance 1, and batch one. A separate posterior-mean VAE round trip
controls codec distortion. PSNR/SSIM accompany recovery.
Encoding intervals resample matched pairs; watermark intervals resample
sources without collection stratification, retaining payload repeats.
Fresh selection and head hashes precede its first inference.
'''
    plots=[]
    for name,color in [('resnet','blue'),('clip','red')]:
        for kind,style in [('gated','solid,mark=*'),('original_fusion','dashed,mark=triangle*')]:
            coords=' '.join(f"({row['marked_share']*100:g},{row[kind+'_auc']-row['probe_auc']:.7f})" for row in gate[name]['clean_marked_prevalence'])
            label=('ResNet' if name=='resnet' else 'CLIP')+(' gate' if kind=='gated' else ' fusion')
            plots.append(r'\addplot['+color+','+style+r'] coordinates {'+coords+r'};\addlegendentry{'+label+'}')
    original_gains='/'.join(f(rg['clean_marked_prevalence'][i]['original_fusion_auc']-rg['clean_marked_prevalence'][i]['probe_auc'],4) for i in [0,5,10])
    prevalence=r'''\subsection{Share of marked images}
We mix each source's clean unmarked and marked outcomes with weights
$1-\pi$ and $\pi$, leaving real controls fixed. This estimates clean
marking prevalence in Experiment~1.
Figure~\ref{fig:prevalence} reports all eleven shares from 0 to 100\%.
ResNet gated AUC gain is '''+f(rg['clean_marked_prevalence'][0]['gated_gain'],4)+'/'+f(rg['clean_marked_prevalence'][5]['gated_gain'],4)+'/'+f(rg['clean_marked_prevalence'][10]['gated_gain'],4)+r''' at 0/50/100\%.
Unconditional fusion gains are '''+original_gains+r''', respectively.
Rare accidental matches explain the small change at zero marking.
\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth,height=4.2cm,xmin=0,xmax=100,
xlabel={Marked AI share (\%)},ylabel={AUC minus probe AUC},
tick label style={font=\scriptsize},label style={font=\scriptsize},
legend columns=2,legend style={font=\scriptsize,at={(0.5,1.03)},anchor=south}]
'''+ '\n'.join(plots)+r'''
\end{axis}\end{tikzpicture}
\caption{Unconditional and gated clean-image fusion gain. Repeats are weighted
counterfactual outcomes of the same 1,396 synthetic sources.}
\label{fig:prevalence}
\end{figure}
'''
    crop50=next(r for r in attacked['conditions']['crop_s1'] if r['backbone']=='resnet' and r['marked_share']==.5)
    diff100=next(r for r in attacked['conditions']['diffusion_strength0p50'] if r['backbone']=='resnet' and r['marked_share']==1)
    prevalence+=r'''A separate exploratory analysis uses 24 AI/24 real sources, matched
marked/unmarked edits, and the same original expected payload. All eleven
shares under sixteen conditions are archived. At 50\% marking after crop,
ResNet unconditional fusion AUC change is '''+f(crop50['fusion_gain'],3)+' ['+ci(crop50['fusion_gain_ci95'])+r''']; the gate equals the probe.
At 100\% marking after strength-0.50 diffusion, input-provenance probe/gate
AUCs are '''+f(diff100['probe_auc'])+'/'+f(diff100['gated_auc'])+r'''.
'''
    encoding_order=['common_png','jpeg_q95','jpeg_q75','jpeg_q50']
    encoding_auc={model:'/'.join(f(encoding[model][condition]['roc_auc']) for condition in encoding_order) for model in ['resnet_full','clip_full']}
    ufd_auc=[baseline['encoding_'+condition]['univfd']['auc'] for condition in encoding_order]
    results=r'''\subsection{Experiment 2: encoding and fresh sample}
Across common PNG and JPEG qualities 95/75/50, ResNet AUC is '''+encoding_auc['resnet_full']+r''' and CLIP AUC is '''+encoding_auc['clip_full']+r''' on 180 matched test pairs.
ResNet's quality-95 change from common PNG is '''+f(encoding['resnet_full']['jpeg_q95']['auc_change_from_common_png']['estimate'],4)+r''' [
'''+', '.join(f(x,4) for x in encoding['resnet_full']['jpeg_q95']['auc_change_from_common_png']['ci95'])+r'''].
At the validation threshold, ResNet real FPR rises from '''+f(encoding['resnet_full']['common_png']['original_fixed']['fpr'])+' to '+f(encoding['resnet_full']['jpeg_q50']['original_fixed']['fpr'])+r''' under JPEG 50.
Published UniversalFakeDetect AUC ranges from '''+f(min(ufd_auc))+' to '+f(max(ufd_auc))+r''' on these controls and is '''+f(baseline['core_clean_unmarked']['univfd']['auc'])+r''' on Experiment~1 clean originals.

On the fresh 192-source sample, ResNet probe/gate AUC is '''+f(fresh['overall']['resnet_probe']['auc'])+'/'+f(fresh['overall']['resnet_gated']['auc'])+r''' and CLIP is '''+f(fresh['overall']['clip_probe']['auc'])+r''' [
'''+ci(fresh['overall']['clip_probe']['ci95'])+r''']; published UFD reaches '''+f(baseline['fresh']['univfd']['auc'])+r'''.
Against the same fresh AI, ResNet AUC is '''+f(fresh['per_real_collection']['coco']['resnet_probe']['auc'])+' with COCO and '+f(fresh['per_real_collection']['openimages']['resnet_probe']['auc'])+r''' with Open Images.
The frozen policy covers '''+f(fresh['frozen_policy']['coverage'])+r''' of the fresh sample and accepts '''+f(fresh['frozen_policy']['group_positive_rate']['R-C']['positive_rate'])+r''' of real inputs;
payload false matches account for '''+f(fresh['frozen_policy']['group_positive_rate']['R-C']['marked_rate'])+r'''.
All nine real-image errors are Open Images inputs (9/48); COCO has 0/48.

\subsection{Experiment 3: learned mark}
'''
    # Select complete all-generator/source summaries; never substitute missing results.
    selected={(r['condition'],r['detector'],r['input_origin_label']):r for r in marks['measurements'] if r['generator']=='all'}
    with (ROOT/'outputs/extension/watermarks/base_covers.csv').open() as handle:
        covers=list(csv.DictReader(handle))
    embedding={method:sum(float(r['psnr']) for r in covers if r['watermark_method']==method)/sum(r['watermark_method']==method for r in covers) for method in ['dwt_dct','trustmark']}
    markrows=[]
    for cond,label in [('identity','Identity'),('jpeg_s2','JPEG 50'),('crop_s1','Crop'),('lowres_s2','Low-res 32'),('regenerate_s2','Grid shift'),('spectral_s1','Spectral wipe'),('vae_only','VAE only'),('diffusion_strength0p25','Diffusion 0.25'),('diffusion_strength0p50','Diffusion 0.50')]:
        a=selected[cond,'dwt_dct',1]; b=selected[cond,'trustmark',1]
        markrows.append(' & '.join([label,f(a['expected_payload_match_rate']),f(b['expected_payload_match_rate']),f(embedding['dwt_dct'] if cond=='identity' else a['mean_psnr_vs_input_cover'],1),f(embedding['trustmark'] if cond=='identity' else b['mean_psnr_vs_input_cover'],1)])+r'\\')
    results+=r'''\begin{table}[t]
\caption{Verification rate and PSNR (dB) for 24 AI sources.
D: $\ge13/16$ bits; T: exact 61-bit message and valid ECC.
Identity/edits pool three payloads; VAE/diffusion use one.}
\label{tab:modern}\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}\toprule
Condition & D rate & T rate & D PSNR & T PSNR\\\midrule
'''+ '\n'.join(markrows)+r'''
\bottomrule\end{tabular}\end{table}
Table~\ref{tab:modern} distinguishes the compound grid shift from actual
generative denoising. Exact source-success intervals and per-payload/seed
measurements are archived.
At either diffusion strength, classical verification is 1/24 (exact 95\%
interval [0.001, 0.211]) and TrustMark verification is 0/24 [0, 0.142].
Both marks are tested at their fixed strengths.
Identity PSNR reports embedding distortion; other rows compare the attack
with its marked input cover. Three new noise seeds give classical recovery
1.000 throughout and TrustMark recovery 0.986/1.000/1.000 across 72
payload outputs from the same 24 sources.
'''
    return {'EXTENSION_METHODS':methods,'EXTENSION_RESULTS':results,'PREVALENCE_RESULTS':prevalence,'GATED_MAIN_ROWS':'\n'.join(gate_main),
        'ATTACK_ROWS':'\n'.join(attacks),'CROP_REAL_BITS':f(float(conditions['crop:1']['real_mean_agreement'])),
        'GATED_DEAD':f(dead['gated']['auc']),'GATED_DEAD_CI':ci(dead['gated']['ci95']),
        'GATED_DEAD_GAIN':f(dead['gated']['auc']-dead['probe']['auc'],4),'GATED_DEAD_GAIN_CI':', '.join(f(x,4) for x in dead['gated_gain_ci95']),
        'GATED_FULL':f(rg['full']['gated']['auc']),'GATED_CLIP_FULL':f(cg['full']['gated']['auc']),
        'GATED_CLIP_DEAD':f(cg['reduced_recovery']['gated']['auc']),
        'FRESH_RESNET':f(fresh['overall']['resnet_probe']['auc']),'FRESH_CLIP':f(fresh['overall']['clip_probe']['auc']),
        'FRESH_FPR':f(fresh['frozen_policy']['group_positive_rate']['R-C']['positive_rate'])}

def order_bibliography(bibliography,template):
    import re
    entries={key:'\\bibitem{'+key+'}\n'+'\n'.join(line for line in body.strip().splitlines() if not line.startswith('%')) for key,body in re.findall(r'\\bibitem\{([^}]+)\}([\s\S]*?)(?=\\bibitem|\Z)',bibliography)}
    ordered=[]
    for group in re.findall(r'\\cite\{([^}]+)\}',template):
        for key in group.split(','):
            if key not in ordered: ordered.append(key)
    if set(entries)!=set(ordered): raise ValueError({'uncited':set(entries)-set(ordered),'missing':set(ordered)-set(entries)})
    return '\n\n'.join(entries[key] for key in ordered)
