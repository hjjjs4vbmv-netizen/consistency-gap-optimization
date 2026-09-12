#!/usr/bin/env python3
"""Draw the exact training controls as editable vector artwork; no model data."""
from pathlib import Path
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

P = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
    'pdf.fonttype':42, 'svg.fonttype':'none', 'mathtext.fontset':'dejavusans'})
ink, blue, teal, orange = '#17324D', '#48647A', '#008681', '#C88730'
fig, ax = plt.subplots(figsize=(7.6, 4.2))
fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.005)
ax.set(xlim=(0,1), ylim=(0,1)); ax.axis('off')

def box(x,y,w,h,text,edge=blue,face='#F1F5F8',fs=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.015',
        edgecolor=edge,facecolor=face,linewidth=1.15))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=ink,linespacing=1.4)

def arrow(start,end,color=blue,rad=0):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,
        connectionstyle=f'arc3,rad={rad}',linewidth=1.05,color=color))

ax.text(.015,.966,'(a) One training example: target construction and loss scale',
        color=ink,weight='bold',va='top',fontsize=11)
box(.022,.66,.14,.145,'Data '+r'$x_0$'+'\nNoise '+r'$\epsilon$',fs=10)
box(.227,.818,.193,.105,r'$x_t=x_0+t\epsilon$',fs=11)
box(.227,.570,.193,.105,r'$x_r=x_0+r\epsilon$',edge=orange,face='#FFF8ED',fs=11)
arrow((.171,.762),(.217,.867))
arrow((.171,.705),(.217,.623))
box(.478,.824,.128,.095,r'$f_\theta(x_t,t)$',fs=11)
box(.478,.570,.128,.105,r'$\mathrm{sg}[f_\theta(x_r,r)]$',edge=orange,face='#FFF8ED',fs=9.4)
arrow((.429,.87),(.468,.87)); arrow((.429,.62),(.468,.62),orange)
ax.text(.542,.753,'Same '+r'$\theta$',ha='center',va='center',fontsize=9,color=blue)
ax.text(.542,.714,'Target stops gradient',ha='center',va='center',fontsize=8.3,color=orange)
box(.688,.703,.122,.107,'Residual\nnorm '+r'$R$',fs=10)
arrow((.616,.869),(.682,.8)); arrow((.616,.62),(.682,.716),orange)
box(.874,.702,.108,.108,r'$\ell=R/\delta$',edge=teal,face='#ECF7F5',fs=11)
arrow((.821,.756),(.864,.756),teal)
ax.text(.323,.515,r'Target control: $r$'+'\nC and B',ha='center',va='top',color=orange,fontsize=9.5)
arrow((.323,.535),(.323,.559),orange)
ax.text(.902,.605,r'Scale control: $\delta$'+'\nD and B',ha='center',va='top',color=teal,fontsize=9.5)
arrow((.916,.625),(.923,.691),teal)

ax.plot([.015,.985],[.417,.417],color='#D4DEE5',lw=.8)
ax.text(.015,.382,'(b) Component histories and startup interventions use different clocks',
        color=ink,weight='bold',va='top',fontsize=10.3)
ax.text(.015,.262,'Component\nhistory',ha='left',va='center',fontsize=10,color=ink)
box(.25,.206,.349,.09,r'Early arm $X\in\{A,B,C,D\}$',face='#F0EDF7',fs=9.7)
box(.631,.206,.349,.09,'Common A continuation',face='#ECF7F5',edge=teal,fs=9.7)
arrow((.608,.251),(.622,.251))
for x,t in [(.25,'0'),(.615,'512'),(.98,'1,024 kimg')]:
    ax.text(x,.322,t,ha='center' if x<.95 else 'right',va='center',fontsize=9,color=blue)
ax.text(.015,.097,'Startup\noperation',ha='left',va='center',fontsize=10,color=ink)
box(.25,.046,.73,.103,'Change LR on successful updates 1-5 or 6-10; then restore it.\nBoth windows occur near transfer start, before the history switch.',face='#F8FAFC',fs=9.4)

out=P/'figures/en';out.mkdir(parents=True,exist_ok=True)
for ext in ['pdf','svg','png']:
    buf=io.BytesIO()
    fig.savefig(buf,format=ext,dpi=200,bbox_inches='tight',pad_inches=.04,facecolor='white')
    (out/f'training_controls.{ext}').write_bytes(buf.getvalue())
plt.close(fig)
print('Built training_controls.pdf, .svg and .png')
