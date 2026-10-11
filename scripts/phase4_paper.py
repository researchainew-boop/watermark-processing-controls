"""Render manuscript additions only from completed controlled measurements."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def rows(path):
    with (ROOT/path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def f(x,n=3):return f'{float(x):.{n}f}'
def ci(x,n=3):return ', '.join(f(v,n) for v in x)
def values(old):
    size=read('outputs/phase4/size_controlled/complete.json')
    assert size['all_converged'] and set(size['variants'])=={'full','clean_repeat','jpeg_only','no_matched_reals'}
    metrics=rows('outputs/phase4/size_controlled/mean_std_metrics.csv')
    idx={(r['backbone'],r['variant'],r['real_source']):r for r in metrics}
    def m(b,v='full'):return idx[b,v,'overall']
    def ms(r,key):
        sd=f(r[key+'_std'])
        return '$'+f(r[key+'_mean'])+r'\pm'+(sd[1:] if sd.startswith('0.') else sd)+'$'
    paired=rows('outputs/phase4/size_controlled/paired_source_bootstrap.csv')
    pi={(r['seed'],r['backbone'],r['contrast'],r['metric']):r for r in paired}
    def contrast(b,v):
        r=pi['five_seed_mean',b,'full_minus_'+v,'auc']
        return f(r['estimate'],4)+' ['+f(r['ci95_low'],4)+', '+f(r['ci95_high'],4)+']'
    size_rows=[]
    for v,label in [('full','Full suite'),('clean_repeat','Clean repeat'),('jpeg_only','JPEG only'),('no_matched_reals','R-T removed')]:
        r=m('resnet',v);c=m('clip',v)
        size_rows.append(' & '.join([label,ms(r,'auc'),ms(r,'tpr_at_validation_fpr5'),ms(r,'test_fpr_at_validation_fpr5'),ms(c,'auc')])+r'\\')
    size_results=r'''Table~\ref{tab:ablate} controls training rows and source identities.
Full-minus-clean-repeat AUC gain is '''+contrast('resnet','clean_repeat')+r''' for
ResNet and '''+contrast('clip','clean_repeat')+r''' for CLIP. Against JPEG-only
augmentation, gains are '''+contrast('resnet','jpeg_only')+' and '+contrast('clip','jpeg_only')+r''', respectively.
These source-bootstrap intervals condition on fitted heads; repeated splits
share sources. Extra rows alone do not explain the benefit, while the
full suite also supplies greater processing diversity.
'''
    domain=read('outputs/phase4/domain_generalization/summary.json')
    attack=rows('outputs/phase4/domain_generalization/attack_scope_false_positives.csv')
    di={(r['backbone'],r['regime'],r['system'],r['attack_scope'],r['real_collection']):r for r in attack}
    def rate(b,v,c):return di[b,v,'probe','matched_13_conditions',c]['fpr']
    dr=domain['models']
    domain_results=r'''With the same matched thirteen conditions in calibration and test,
COCO-trained CLIP has FPR '''+f(rate('clip','coco_only','coco'))+'/'+f(rate('clip','coco_only','openimages'))+r''' on COCO/Open Images;
joint training gives '''+f(rate('clip','joint','coco'))+'/'+f(rate('clip','joint','openimages'))+r'''. Calibration FPR is 0.049
on separate sources. ResNet pooled AUC for COCO-only/Open-Images-only/joint
fits is '''+'/'.join(f(dr['resnet_'+v]['systems']['probe']['auc']) for v in ['coco_only','openimages_only','joint'])+r'''. Collection transfer remains
a calibration failure even after both real collections enter training.
'''
    methods=r'''Size controls replace each real source's eleven R-T slots with its clean
feature or six JPEG-70/five JPEG-50 features, retaining R-C/R-S. Full and
replacement arms have 53,326 rows; removal has 30,842. Sources, labels,
balanced weights, $C$, and optimization are identical.
A seed-20260920 collection diagnostic uses 700 COCO, 700 Open Images,
or 350+350 real training sources with the same synthetic sources.
Fusion/calibration use disjoint validation-source halves (75 real each).
All share 150+150 test photographs and 1,405 AI sources; Midjourney
enters neither fitting nor calibration.
'''
    threshold_methods=r'''New operating points select the lowest validation-real logit threshold
with empirical FPR $\le0.05$, retaining whole ties and classifying scores
$\ge$ threshold. Test thresholds stay fixed; observed test FPR can exceed
the calibration target. The collection diagnostic reserves calibration
sources from both probe and fusion fitting.
'''
    new=read('outputs/phase4/fresh_replication/summary.json')
    assert new['source_count']==400 and new['images']==2560 and not new['models_refitted'] and not new['thresholds_retuned']
    ni={(r['attack'],r['AI_mark'],r['system']):r for r in new['measurements']}
    def fresh(attack='identity',mark='unmarked',system='resnet_probe'):return ni[attack,mark,system]
    fresh_methods=r'''A second replication uses 400 previously unscored sources, 80 per
collection (COCO/Open Images/SD/DALL$\cdot$E~3/Midjourney), excluding
prior archive IDs and pixel/hash overlaps. Common resize/JPEG-95 precedes
identity, JPEG-50, crop, and low-res-32: 2,560 copies including both
AI marking states and unmarked real controls. Joint-collection
seed-20260920 heads and validation thresholds are locked before selection;
one frozen pass uses no refitting or test tuning.
'''
    pairs={(r['attack'],r['AI_mark'],r['backbone'],r['comparison']):r for r in new['paired_comparisons']}
    gain=pairs['overall','marked','resnet','gated-minus-probe']
    loss=pairs['overall','unmarked','resnet','gated-minus-probe']
    fresh_results=r'''On the new 400-source identity replication, unmarked ResNet/CLIP probe
AUC is '''+f(fresh(system='resnet_probe')['auc'])+'/'+f(fresh(system='clip_probe')['auc'])+r'''. Fixed-validation probe TPR/FPR is
'''+f(fresh(system='resnet_probe')['test_TPR'])+'/'+f(fresh(system='resnet_probe')['test_FPR'])+r''' for ResNet and
'''+f(fresh(system='clip_probe')['test_TPR'])+'/'+f(fresh(system='clip_probe')['test_FPR'])+r''' for CLIP.
Across all four conditions, marked ResNet gate-minus-probe AUC is
'''+f(gain['auc_difference'],4)+' ['+ci(gain['auc_difference_ci95'],4)+r''']; unmarked change is
'''+f(loss['auc_difference'],4)+' ['+ci(loss['auc_difference_ci95'],4)+r'''].
CLIP probe/gate FPR reaches '''+f(fresh('overall','unmarked','clip_probe')['test_FPR'])+r''' overall and
'''+f(fresh('lowres','unmarked','clip_probe')['test_FPR'])+r''' for low-res-32, despite validation targeting 0.05.
'''
    # Extend the measured Phase3 watermark table with photographic false matches.
    fp=rows('outputs/phase4/watermark_reporting/table_ready.csv')
    fi={(r['condition'],r['detector']):r for r in fp}
    ext=old['EXTENSION_RESULTS']
    ext=ext.replace(r'\setlength{\tabcolsep}{4pt}',r'\setlength{\tabcolsep}{2pt}')
    ext=ext.replace(r'\begin{tabular}{lrrrr}\toprule',r'\begin{tabular}{lrrrrrr}\toprule')
    ext=ext.replace('Condition & D rate & T rate & D PSNR & T PSNR', 'Condition & D rate & T rate & D FP & T FP & D dB & T dB')
    labels={'Identity':'identity','JPEG 50':'jpeg_s2','Crop':'crop_s1','Low-res 32':'lowres_s2','Grid shift':'regenerate_s2','Spectral wipe':'spectral_s1'}
    table_start=ext.index(r'\begin{tabular}')
    table_end=ext.index(r'\end{tabular}',table_start)
    table=ext[table_start:table_end]
    lines=[]
    for line in table.splitlines():
        label=line.split(' & ')[0]
        if label in labels:
            cells=line.split(' & ');cond=labels[label]
            cells=cells[:3]+[f(fi[cond,'dwt_dct']['photograph_default_rate']),f(fi[cond,'trustmark']['photograph_default_rate'])]+cells[3:]
            line=' & '.join(cells)
        elif label in ['VAE only','Diffusion 0.25','Diffusion 0.50']:
            cells=line.split(' & ');line=' & '.join(cells[:3]+['---','---']+cells[3:])
        lines.append(line)
    ext=ext[:table_start]+'\n'.join(lines)+'\n'+ext[table_end:]
    ext=ext.replace('bits; T: exact 61-bit message and valid ECC. Identity/edits pool three',
        'bits; T: exact 61-bit message and valid ECC. FP: default-payload photographic false matches on 204 controls. Criteria differ. Identity/edits pool three')
    ext=ext.replace('Per-payload, collection, and real-control results are archived.',
        r'''Across thirteen conditions, DWT photographic false matches rise from
0.0109 for one expected payload to 0.0317 when any of three is accepted.
TrustMark has no matches; each zero-of-204 condition has exact 95\%
upper bound 0.0179. These fixed criteria are not calibrated to equal FPR.''')
    # Keep encoding evidence, shorten an ancillary abstention-policy result.
    ext=ext.replace('The frozen policy covers 0.620 and accepts\n9/96 real inputs: all nine are Open Images (9/48), versus COCO 0/48.',
        'The earlier abstention policy accepts 9/96 real inputs, all Open Images; it violates its validation error constraint.')
    split=ext.index(r'\subsection{Experiment 3: learned mark}')
    ext=ext[:split]+fresh_results+'\n'+ext[split:]
    extension_methods=r'''\subsection{Experiments 2 and 3}
Experiment~2 fixes heads across common PNG/JPEG-95/75/50 on 300
approximately CLIP-matched COCO/AI pairs (120 validation/180 test).
The earlier frozen fresh sample has 48 COCO, 48 Open Images, and 96
DALL$\cdot$E~3 sources, unmarked and common-JPEG-95 processed.
Published UniversalFakeDetect ViT-L/14 uses its unchanged head and
official center-crop-224/raw-CLIP preprocessing, also tested on the
same native inputs without common resizing/JPEG.
'''+fresh_methods+r'''
Experiment~3 pairs 204 test AI sources (68 per collection) with 204
COCO controls by maximum total CLIP similarity. Seeded ID-hash ordering
selects AI sources. Three balanced payloads use DWT--DCT or official
TrustMark 0.9.2 Q, strength 1, BCH5: sixteen payload bits plus 45 zeros,
with full-message/schema/ECC verification. Ten conditions and three
noise seeds ($\sigma=6$) include unmarked controls.
On all 204 AI sources, each mark's default payload undergoes posterior-mean
VAE reconstruction or DDIM strengths 0.25/0.50 using Realistic Vision V5.1
(SD1.5-derived), twenty configured steps (5/10 effective), empty prompt,
guidance 1, seed 20261009, and batch one. Recovery and distortion use
fixed criteria; checkpoint fingerprints and settings are archived.
'''
    return {'SIZE_CLEAN_AUC':f(m('resnet','clean_repeat')['auc_mean']), 'SIZE_FULL_AUC':f(m('resnet')['auc_mean']),
        'SIZE_DOMAIN_METHODS':methods,'FIXED_THRESHOLD_METHODS':threshold_methods,'SIZE_RESULTS':size_results,
        'SIZE_ROWS':'\n'.join(size_rows),'DOMAIN_RESULTS':domain_results,'EXTENSION_RESULTS':ext,
        'EXTENSION_METHODS':extension_methods}
