"""Build publication figures and LaTeX tables from recorded experiment evidence."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw

BASE=Path('report');ASSETS=BASE/'assets';GENERATED=BASE/'generated'
ASSETS.mkdir(parents=True,exist_ok=True);GENERATED.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})


def load(path):return json.loads(Path(path).read_text())


def box(ax,x,y,w,h,text):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02',facecolor='#edf5f2',edgecolor='#196650'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)


def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#263c35'))


fig,axes=plt.subplots(4,1,figsize=(8.5,6.2))
for ax in axes:ax.set(xlim=(0,10),ylim=(0,1.5));ax.axis('off')
rows=[['RGB 128x128','Encoder\n32/64/128 channels','Spatial latent\n64x16x16','Decoder\n(no skips)','Restored RGB'],
      ['RGB 128x128','CNN classifier\n4 probabilities','argmax route','Identity or\nnoise/blur/occlusion','Restored RGB'],
      ['RGB 128x128','Gate softmax\nT = 1.307','Identity +\n3 fine-tuned experts','Weighted sum\n(all branches)','Restored RGB'],
      ['Photo +\nstyle embedding','5-level U-Net\n32 base channels','4 skip links\nconditional decoding','Sigmoid sketch','Conditional PatchGAN\ntraining only']]
for index,(ax,labels) in enumerate(zip(axes,rows)):
    ax.text(0,1.35,f'Task {index+1}',weight='bold',fontsize=10)
    for j,label in enumerate(labels):
        box(ax,.05+2*j,.3,1.65,.7,label)
        if j<4:arrow(ax,(1.72+2*j,.65),(2.03+2*j,.65))
axes[-1].text(5,.05,'Task 4: discriminator observes photo, candidate/target sketch and the style embedding.',ha='center',fontsize=8)
fig.tight_layout();fig.savefig(ASSETS/'architectures.pdf',bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(8,5.5))
histories=[('Task 1 Run A','task1/final/history.json'),('Task 2 blur specialist','task2/gaussian_blur/final/history.json'),('Task 3 soft MoE','task3/final/history.json'),('Task 4 cGAN','task4/final/history.json')]
for ax,(title,relative) in zip(axes.flat,histories):
    h=load(BASE/'evidence'/relative);x=[r['epoch'] for r in h]
    ax.plot(x,[r['objective'] for r in h],label='Validation selection objective',color='#087f5b')
    key='g_total' if title.startswith('Task 4') else 'train_loss'
    other=ax.twinx();other.plot(x,[r[key] for r in h],color='#cc6d20',alpha=.65,label='Training loss')
    ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel('Validation objective');other.set_ylabel('Training loss')
    if 'Task 3' in title:ax.axvline(3.5,color='gray',linestyle='--')
fig.tight_layout();fig.savefig(ASSETS/'training_curves.pdf',bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(8,3))
h=load(BASE/'evidence/task2/classifier/final/history.json')
axes[0].plot([r['epoch'] for r in h],[r['accuracy'] for r in h],label='Validation accuracy');axes[0].plot([r['epoch'] for r in h],[r['macro_f1'] for r in h],label='Validation macro F1');axes[0].legend();axes[0].set(xlabel='Epoch',ylabel='Score',title='Corruption classifier')
h=load(BASE/'evidence/task4/final/history.json')
for key,label in [('d_real','D real'),('d_fake','D fake'),('g_adversarial','G adversarial')]:axes[1].plot([r['epoch'] for r in h],[r[key] for r in h],label=label)
axes[1].legend();axes[1].set(xlabel='Epoch',ylabel='BCE loss',title='GAN adversarial terms')
fig.tight_layout();fig.savefig(ASSETS/'classifier_gan_curves.pdf',bbox_inches='tight');plt.close(fig)

studies=[('T1','task1_original_search'),('T2 classifier','task2/classifier/search'),('T2 noise','task2/salt_and_pepper/search'),('T2 blur','task2/gaussian_blur/search'),('T2 occlusion','task2/rectangular_occlusion/search'),('T3','task3/search'),('T4','task4/search')]
fig,axes=plt.subplots(2,4,figsize=(9,4.4));summary=[]
for ax,(label,relative) in zip(axes.flat,studies):
    root=BASE/'evidence'/relative;s=load(root/'study_summary.json');rows=list(csv.DictReader((root/'trials.csv').open()))
    for row in rows:
        if row['value']:ax.scatter(int(row['number']),float(row['value']),marker='o' if row['state']=='COMPLETE' else 'x',color='#087f5b' if row['state']=='COMPLETE' else '#a55b19')
    ax.set_title(label);ax.set_xlabel('Trial');ax.set_ylabel('Objective')
    summary.append((label,s))
axes.flat[-1].axis('off');axes.flat[-1].text(0,.8,'o Complete\nx Pruned/other\nWithin-study comparison\nonly',va='top')
fig.tight_layout();fig.savefig(ASSETS/'optuna_trials.pdf',bbox_inches='tight');plt.close(fig)
lines=['\\begin{tabular}{lrrr}\\toprule Study & Total & Complete & Best trial\\\\\\midrule']
for label,s in summary:lines.append(f"{label} & {s['total_trials']} & {s['complete_trials']} & {s['best_trial']}\\\\")
lines.append('\\bottomrule\\end{tabular}');(GENERATED/'optuna_table.tex').write_text('\n'.join(lines))

for source,name in [('runs/integration/browser/workspace_3_desktop.png','app_desktop.png'),('runs/integration/browser/workspace_4_mobile.png','app_mobile.png')]:shutil.copy2(source,ASSETS/name)
styles=[BASE/f'evidence/task4/validation/example_6800_style_{s}.png' for s in [1,2,3]]
canvas=Image.new('RGB',(512,450),'white')
for i,path in enumerate(styles):canvas.paste(Image.open(path),(0,150*i))
canvas.save(ASSETS/'sketch_validation.png')
shutil.copy2(BASE/'evidence/task4/validation/failure_candidate_1.png',ASSETS/'pencil_failure.png')

locations=[]
for task in ['task1','task2','task3']:
    candidates=[p.parent for p in (BASE/'evidence'/task).rglob('results.json') if p.parent.name=='validation']
    locations.append(candidates[0])
shared=set(p.name for p in locations[0].glob('example_*.png'))
for location in locations[1:]:shared&={p.name for p in location.glob('example_*.png')}
chosen=[]
for condition in ['salt_and_pepper','gaussian_blur','rectangular_occlusion']:
    options=sorted(n for n in shared if condition in n)
    if options:chosen.append(options[0])
if chosen:
    canvas=Image.new('RGB',(640,150*len(chosen)),'white');draw=ImageDraw.Draw(canvas)
    for i,name in enumerate(chosen):
        panels=[Image.open(root/name).convert('RGB') for root in locations]
        for other in panels[1:]:assert np.array_equal(np.asarray(panels[0].crop((0,22,256,150))),np.asarray(other.crop((0,22,256,150))))
        crops=[panels[0].crop((0,22,128,150)),panels[0].crop((128,22,256,150)),*[p.crop((256,22,384,150)) for p in panels]]
        for j,(title,crop) in enumerate(zip(['Clean','Input','Universal','Hard','Soft'],crops)):
            canvas.paste(crop,(128*j,150*i+22));draw.text((128*j+3,150*i+3),title,fill='black')
    canvas.save(ASSETS/'restoration_validation.png')

test=Path('runs/final_test/restoration/results.json')
if test.exists():
    r=load(test);assert not r['limited'] and r['cases']==61490
    lines=['\\begin{tabular}{llrrrr}\\toprule Condition & Severity & Input & Universal & Hard & Soft\\\\\\midrule']
    for group,v in r['metrics'].items():
        condition,severity=group.split('/');condition={'clean':'Clean','salt_and_pepper':'Noise','gaussian_blur':'Blur','rectangular_occlusion':'Occlusion'}[condition]
        lines.append(condition+' & '+severity+' & '+' & '.join(f"{v[n]['psnr']:.2f}" for n in ['input','universal','hard','soft'])+'\\\\')
    lines.append('\\bottomrule\\end{tabular}');(GENERATED/'test_psnr.tex').write_text('\n'.join(lines))
    lines=['\\begin{tabular}{llrrrr}\\toprule Condition & Severity & Input & Universal & Hard & Soft\\\\\\midrule']
    for group,v in r['metrics'].items():
        condition,severity=group.split('/');condition={'clean':'Clean','salt_and_pepper':'Noise','gaussian_blur':'Blur','rectangular_occlusion':'Occlusion'}[condition]
        lines.append(condition+' & '+severity+' & '+' & '.join(f"{v[n]['ssim']:.4f}" for n in ['input','universal','hard','soft'])+'\\\\')
    lines.append('\\bottomrule\\end{tabular}');(GENERATED/'test_ssim.tex').write_text('\n'.join(lines))
    lines=['\\begin{tabular}{lrrrr}\\toprule Condition & Input & Universal & Hard & Soft\\\\\\midrule']
    for condition in ['clean','salt_and_pepper','gaussian_blur','rectangular_occlusion']:
        values=[v for group,v in r['metrics'].items() if group.startswith(condition+'/')]
        label={'clean':'Clean','salt_and_pepper':'Noise','gaussian_blur':'Blur','rectangular_occlusion':'Occlusion'}[condition]
        lines.append(label+' & '+' & '.join(f"{np.mean([v[name]['l1'] for v in values]):.5f}" for name in ['input','universal','hard','soft'])+'\\\\')
    lines.append('\\bottomrule\\end{tabular}');(GENERATED/'test_l1.tex').write_text('\n'.join(lines))
    low=r['metrics']['gaussian_blur/low'];medium=r['metrics']['gaussian_blur/medium'];high=r['metrics']['gaussian_blur/high']
    discussion=(f"The soft mixture improves mean PSNR and SSIM over hard routing in all nine corrupted test severity groups. "
      f"For mild blur, it still reduces PSNR from {low['input']['psnr']:.2f} to {low['soft']['psnr']:.2f} dB and improves both input metrics on only {low['soft']['both_metrics_better_than_input']}/6149 cases. "
      f"Medium and high blur improve both metrics on {medium['soft']['both_metrics_better_than_input']}/6149 and {high['soft']['both_metrics_better_than_input']}/6149. "
      f"Mean identity/blur weights shift from {low['mean_weights']['clean']:.3f}/{low['mean_weights']['gaussian_blur']:.3f} at low blur to {high['mean_weights']['clean']:.3f}/{high['mean_weights']['gaussian_blur']:.3f} at high blur. "
      "These test findings support the validation interpretation while preserving the mild-blur limitation.\n\n"
      f"Hard routing correctly identifies {r['classifier']['confusion_matrix'][0][0]} of 6149 clean views. Its largest confusion is blur classified as clean ({r['classifier']['confusion_matrix'][2][0]} cases across severities). "
      f"Only {sum(v['cascading_failures'] for v in r['metrics'].values())} wrong-route cases lose PSNR relative to oracle routing; bypassing a weak mild-blur expert can accidentally improve PSNR. "
      "Thus classification accuracy and restoration quality are related but not interchangeable. "
      "The soft mixture changes most clean images slightly, whereas hard routing usually returns them exactly. "
      "Soft clean mean SSIM exceeds hard clean SSIM here despite lower clean mean PSNR, because the hard router has occasional clean-image dispatch failures and an identity ceiling. "
      "These are descriptive single-seed measurements, not statistical significance tests.\n")
    (GENERATED/'test_discussion.tex').write_text(discussion)
    (GENERATED/'test_numbers.tex').write_text(f"\\newcommand{{\\TestAccuracy}}{{{r['classifier']['accuracy']:.4f}}}\n\\newcommand{{\\TestFOne}}{{{r['classifier']['macro_f1']:.4f}}}\n")
    for name in ['confusion_matrix.png','routing_heatmap.png']:shutil.copy2(test.parent/name,ASSETS/name)
    for model in ['universal','hard','soft']:shutil.copy2(test.parent/model/'failure_candidate_1.png',ASSETS/f'{model}_test_failure.png')
    canvas=Image.new('RGB',(512,450),'white')
    for i,model in enumerate(['universal','hard','soft']):canvas.paste(Image.open(ASSETS/f'{model}_test_failure.png'),(0,150*i))
    canvas.save(ASSETS/'test_failures.png')
sketch=load('runs/final_test/sketch/results.json')
lines=['\\begin{tabular}{lrrrr}\\toprule Style & L1 & SSIM & PSNR & Blank SSIM\\\\\\midrule']
for style,v in sketch['metrics'].items():lines.append(f"{style} & {v['l1']:.4f} & {v['ssim']:.4f} & {v['psnr']:.2f} & {v['blank_ssim']:.4f}\\\\")
lines.append('\\bottomrule\\end{tabular}');(GENERATED/'sketch_test.tex').write_text('\n'.join(lines))
shutil.copy2('runs/final_test/sketch/failure_candidate_1.png',ASSETS/'ink_test_failure.png')
print('Report figures and available result tables generated.')
