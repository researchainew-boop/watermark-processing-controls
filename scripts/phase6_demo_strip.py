"""Layout existing project-generated illustration pixels and measured bits."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'manuscript/assets'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous=json.loads((ASSETS/'generated-asset-provenance.json').read_text())
    measured=json.loads((ASSETS/'example_measurements.json').read_text())
    assert previous['project_generated'] and previous['dataset_images_used'] is False
    assert measured['figure_example_only'] and measured['excluded_from_quantitative_benchmark']
    fingerprints={r['path']:r['file_sha256'] for r in previous['figure_images']}
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,6,figsize=(7.2,1.64))
    for ax,row in zip(axes,measured['images']):
        source=ROOT/row['path']
        if sha(source)!=fingerprints[row['path']]:raise ValueError('Existing illustration changed')
        ax.imshow(mpimg.imread(source));ax.set_axis_off()
        ax.set_title(row['label'],fontsize=7,pad=4)
        ax.text(.5,-.075,row['bits'],transform=ax.transAxes,ha='center',va='top',fontsize=5.6,fontfamily='DejaVu Sans Mono')
        ax.text(.5,-.22,str(row['agreement_count'])+'/16',transform=ax.transAxes,ha='center',va='top',fontsize=6.5)
    fig.subplots_adjust(left=.01,right=.99,top=.85,bottom=.25,wspace=.05)
    outputs=[]
    for suffix in ['pdf','png']:
        path=ASSETS/('phase6_demo_strip.'+suffix)
        fig.savefig(path,dpi=300)
        outputs.append(path)
    plt.close(fig)
    # Separate provenance preserves the original illustration's sealed record.
    provenance={'project_generated':True,'dataset_images_used':False,'excluded_from_every_benchmark':True,
        'new_inference_performed':False,'illustration_only':True,'method':measured['method'],'strength':measured['strength'],
        'diffusion_parameters':measured['diffusion_parameters'],'payload':measured['payload16'],
        'not_the_primary_TrustMark_or_matched_40dB_setting':True,'no_pixel_reencoding_or_new_decoding':True,
        'layout_script_sha256':sha(Path(__file__)),
        'inputs':{'generated-asset-provenance.json':sha(ASSETS/'generated-asset-provenance.json'),
                  'example_measurements.json':sha(ASSETS/'example_measurements.json')},
        'assets':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in outputs]
            +[{'path':r['path'],'sha256':fingerprints[r['path']]} for r in measured['images']]
            +[{'path':previous['original_image'],'sha256':previous['original_file_sha256']}],
        'original_image_assets':[{'path':r['path'],'sha256':fingerprints[r['path']],'bits':r['bits'],
            'agreement_count':r['agreement_count']} for r in measured['images']]}
    (ASSETS/'phase6_demo_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('Prepared existing demonstration strip, no new measurement')


if __name__=='__main__':main()
