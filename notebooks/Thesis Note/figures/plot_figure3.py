from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path('results/baseline/final_linear_v3_validation_loss')
OUT = Path('results/figures/figure3_all_scenarios')
OUT.mkdir(parents=True, exist_ok=True)
SCENARIOS = [
 ('image-only', 'Photograph-only'),
 ('xray_only', 'Radiograph-only'),
 ('text_only', 'Clinical Text'),
 ('image-xray', 'Photograph + Radiograph'),
 ('image_text', 'Photograph + Text'),
 ('text_xray', 'Radiograph + Text'),
 ('full_multimodal', 'Full Multimodal'),
]
# A4 portrait with 2.5 cm margins: usable area 16.0 x 24.7 cm.
CM = 1 / 2.54
FIG_W, FIG_H = 16.0 * CM, 24.7 * CM
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.2,'axes.labelsize':6.2,
 'xtick.labelsize':5.4,'ytick.labelsize':5.4,'pdf.fonttype':42,'ps.fonttype':42,
 'axes.linewidth':0.55,'savefig.facecolor':'white'})
fig, axes = plt.subplots(7, 2, figsize=(FIG_W, FIG_H), squeeze=False)
fig.subplots_adjust(left=.145, right=.99, top=.975, bottom=.062, wspace=.34, hspace=.68)
TRAIN, VAL, BEST = '#4477AA', '#CC6677', '#333333'
for i, (folder, name) in enumerate(SCENARIOS):
    path = ROOT / folder / 'history.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing: {path}')
    hist = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(hist, list) or not hist:
        raise ValueError(f'Expected non-empty history list: {path}')
    required = ('epoch','train_macro_f1','validation_macro_f1','train_loss','validation_loss')
    for rec in hist:
        missing = [k for k in required if k not in rec or rec[k] is None]
        if missing: raise ValueError(f'{path}: epoch record missing {missing}')
    hist.sort(key=lambda x: int(x['epoch']))
    best = max(hist, key=lambda x: float(x['validation_macro_f1']))
    best_ep = int(best['epoch'])
    epochs = [int(x['epoch']) for x in hist]
    for col, metric in enumerate(('macro_f1', 'loss')):
        ax = axes[i, col]
        tr = [float(x[f'train_{metric}']) for x in hist]
        va = [float(x[f'validation_{metric}']) for x in hist]
        ax.plot(epochs, tr, color=TRAIN, lw=1.05, label='Training')
        ax.plot(epochs, va, color=VAL, lw=1.05, label='Validation')
        ax.axvline(best_ep, color=BEST, lw=.8, ls='--', alpha=.85, zorder=2)
        ax.scatter([best_ep], [float(best[f'validation_{metric}'])], s=21,
                   color=VAL, edgecolors=BEST, linewidths=.65, zorder=5)
        ax.annotate(f'E{best_ep}', xy=(best_ep, float(best[f'validation_{metric}'])),
                    xytext=(4, 5), textcoords='offset points', fontsize=5.1,
                    color=BEST, fontweight='bold', clip_on=True)
        ax.set_xlim(min(epochs), max(epochs) + 1)
        ax.set_xlabel('Epoch', labelpad=1)
        ax.set_ylabel('Macro-F1' if col == 0 else 'Loss', labelpad=2)
        if col == 0: ax.set_ylim(0, 1.04)
        ax.grid(alpha=.18, linewidth=.4)
        ax.tick_params(length=2, width=.5, pad=1.5)
        ax.spines[['top','right']].set_visible(False)
    # Scenario labels are outside axes, not subplot titles.
    pos = axes[i, 0].get_position()
    fig.text(.018, (pos.y0 + pos.y1)/2, name, rotation=90,
             va='center', ha='center', fontsize=6.0, fontweight='bold')
    print(f'{name:26s} best epoch {best_ep:2d}, val Macro-F1 {best["validation_macro_f1"]:.4f}, '
          f'val loss {best["validation_loss"]:.4f}')
legend = [Line2D([0],[0],color=TRAIN,lw=1.2,label='Training'),
          Line2D([0],[0],color=VAL,lw=1.2,label='Validation'),
          Line2D([0],[0],color=BEST,lw=.8,ls='--',label='Best checkpoint (validation Macro-F1)'),
          Line2D([0],[0],marker='o',color='none',markerfacecolor=VAL,
                 markeredgecolor=BEST,markersize=4.5,label='Validation value at best checkpoint')]
fig.legend(handles=legend, loc='lower center', ncol=2, bbox_to_anchor=(.5,.006),
           frameon=False, fontsize=5.7, columnspacing=1.4, handlelength=2.5)
for ext in ('pdf','png','svg'):
    dest = OUT / f'figure3_seven_scenarios.{ext}'
    fig.savefig(dest, dpi=400, facecolor='white')  # Do not use bbox_inches='tight': preserve exact page size.
    print('Saved:', dest)
plt.close(fig)
