"""Publication figures from saved predictions and the generated illustration."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'manuscript/assets'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                     'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
    ASSETS.mkdir(parents=True,exist_ok=True)
    fig.savefig(ASSETS/f'{name}.pdf',bbox_inches='tight',pad_inches=.03,
                metadata={'Author':'','Creator':'Study plotting script','CreationDate':None})
    fig.savefig(ASSETS/f'{name}.png',dpi=220,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)

def roc():
    frame=pd.read_csv(ROOT/'outputs/phase3/controls/expanded_test_predictions.csv',low_memory=False)
    fig,ax=plt.subplots(figsize=(3.45,2.12))
    metrics={}
    for label,col,color,style in [('Probe','resnet_probe_logit','#245487','-'),
            ('Fusion','resnet_fusion_logit','#b95720','--'),('Gate','resnet_gated_logit','#2a7b57','-.')]:
        # Legacy predictions use unprefixed ResNet column names.
        column=col if col in frame else col.removeprefix('resnet_')
        fpr,tpr,_=roc_curve(frame.label,frame[column],drop_intermediate=False)
        positive=fpr>0
        ax.step(fpr[positive],tpr[positive],where='post',label=label,color=color,
                linestyle=style,linewidth=1.25)
        metrics[label]=float(roc_auc_score(frame.label,frame[column]))
    ax.set_xscale('log'); ax.set_xlim(1e-3,1); ax.set_ylim(0,1.015)
    ax.set_xlabel('False-positive rate (log scale)'); ax.set_ylabel('True-positive rate')
    ax.axvline(.05,color='#777777',linestyle=':',linewidth=.8)
    ax.set_yticks([0,.25,.5,.75,1]); ax.grid(True,which='major',alpha=.18)
    ax.legend(loc='lower right',frameon=False,fontsize=7)
    fig.tight_layout(pad=.2)
    save(fig,'resnet_roc')
    (ASSETS/'roc_provenance.json').write_text(json.dumps({'rows':len(frame),
        'sources':int(frame.source_id.nunique()),'saved_scores_only':True,
        'refits':False,'auc':metrics,'input_sha256':hashlib.sha256(
            (ROOT/'outputs/phase3/controls/expanded_test_predictions.csv').read_bytes()).hexdigest()},indent=2))

def pipeline():
    measures=json.loads((ASSETS/'example_measurements.json').read_text())
    images=measures['images']
    assert len(images)==6 and all(len(r['bits'])==16 for r in images)
    fig=plt.figure(figsize=(7.05,2.28))
    diagram=fig.add_axes([.01,.67,.98,.32]); diagram.set_xlim(0,1); diagram.set_ylim(0,1); diagram.axis('off')
    def box(x,y,w,h,text,color):
        diagram.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.009,rounding_size=.022',
                          linewidth=.7,edgecolor='#404b58',facecolor=color))
        diagram.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7)
    def arrow(x,y,x2,y2):
        diagram.add_patch(FancyArrowPatch((x,y),(x2,y2),arrowstyle='-|>',mutation_scale=8,
                                         linewidth=.7,color='#404b58'))
    box(.014,.18,.172,.67,'Source split\nReal / AI\nmarked or unmarked','#eef1f5')
    box(.222,.29,.15,.45,'Matched\ntransformations','#eef1f5')
    box(.412,.55,.17,.36,'Frozen origin\nprobe score p','#eaf0f7')
    box(.412,.06,.17,.36,'Expected-payload\nagreement a','#f8efe6')
    box(.636,.25,.23,.5,'Agreement gate\na ≥ 13/16: fusion f(p,a)\na < 13/16: probe p','#edf4ef')
    box(.908,.29,.075,.45,'Origin\nscore','#eef1f5')
    arrow(.185,.515,.219,.515); arrow(.375,.515,.409,.73); arrow(.375,.515,.409,.24)
    arrow(.582,.73,.633,.58); arrow(.582,.24,.633,.40); arrow(.872,.5,.902,.5)
    for i,row in enumerate(images):
        ax=fig.add_axes([.013+i*.165,.15,.151,.48])
        ax.imshow(Image.open(ROOT/row['path']).convert('RGB')); ax.axis('off')
        ax.set_title(row['label'],fontsize=8,pad=3)
        count=row.get('agreement_count',sum(a==b for a,b in zip(row['bits'],measures['payload16'])))
        fig.text(.0885+i*.165,.099,row['bits'],ha='center',va='center',fontsize=6.5,fontfamily='DejaVu Sans Mono')
        fig.text(.0885+i*.165,.038,f'{count}/16 match',ha='center',va='center',fontsize=7)
    save(fig,'pipeline_examples')
    provenance_path=ASSETS/'generated-asset-provenance.json'
    provenance=json.loads(provenance_path.read_text())
    inventory={r['path']:r for r in provenance['assets']}
    for filename in ['pipeline_examples.pdf','pipeline_examples.png']:
        path=ASSETS/filename; relative=path.relative_to(ROOT).as_posix()
        inventory[relative]={'path':relative,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    provenance['assets']=list(inventory.values())
    provenance['pipeline_figure_driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance_path.write_text(json.dumps(provenance,indent=2),encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['roc','pipeline','both'],default='both')
    args=ap.parse_args()
    if args.stage in ['roc','both']: roc()
    if args.stage in ['pipeline','both']: pipeline()
