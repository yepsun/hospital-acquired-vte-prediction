#!/usr/bin/env python3
"""99d: redraw Figure 2 (calibration) with data-filling per-panel axes.

Reviewer point M6: the published panel axes ran to 0.9% while the Padua
panel's data stopped at ~0.53%, leaving most of panel (a) empty. This script
re-renders the same decile-bin reliability curves from
revision_threshold_calib_v3.json (no recomputation) with per-panel limits:
(a) Padua 0-0.6%, (b, c) LR/XGBoost 0-1.0%. Output overwrites the ejim
deliverables Figure_2_calibration.png/.tif (600 dpi, LZW).
"""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from PIL import Image

RES = '/Users/Yepsun/Mywork/Vscodeprojects/mimic/results_vte'
EJIM = '/Users/Yepsun/Mywork/Vscodeprojects/mimic/results_vte/JAMIA/ejim'

calib = json.load(open(f'{RES}/ajm/new/output_era/revision_threshold_calib_v3.json'))['calibration']

C_PADUA, C_LR, C_XGB = '#4d4d4d', '#1f77b4', '#d62728'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10.5,
    'axes.linewidth': 0.9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#dcdcdc', 'grid.linewidth': 0.55,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'pdf.fonttype': 42})

panel_specs = [
    ('Padua (Platt-mapped)', 0.6,
     [('padua_platt', 'Padua (Platt)', C_PADUA, 's', 'solid', 'white')]),
    ('Logistic regression', 1.0,
     [('lr_pre', 'Uncalibrated', C_LR, 'o', (0, (3, 2)), 'white'),
      ('lr_post', 'Isotonic-recalibrated', C_LR, 'o', 'solid', C_LR)]),
    ('XGBoost', 1.0,
     [('xgb_pre', 'Uncalibrated', C_XGB, '^', (0, (3, 2)), 'white'),
      ('xgb_post', 'Isotonic-recalibrated', C_XGB, '^', 'solid', C_XGB)]),
]

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))
for j, (ttl, xmax, series) in enumerate(panel_specs):
    ax = axes[j]
    ax.plot([0, xmax], [0, xmax], color='#999999', ls='--', lw=1.0,
            label='Ideal', zorder=1)
    for key, lab, c, mkr, ls_, mfc in series:
        dd = calib[key]
        ax.plot(np.array(dd['bin_mean_predicted']) * 100,
                np.array(dd['bin_observed_fraction']) * 100,
                mkr + '-', color=c, ms=4.8, lw=1.6, ls=ls_, mec=c, mfc=mfc,
                mew=1.2, label=lab, zorder=3)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.04, xmax * 0.93)
    step = 0.2 if xmax <= 0.6 else 0.25
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.set_xlabel('Mean predicted VTE risk per bin (%)')
    if j == 0:
        ax.set_ylabel('Observed VTE fraction (%)')
    ax.legend(frameon=False, fontsize=7.8, loc='upper left',
              handlelength=2.2, borderaxespad=0.3)
    ax.set_title(f"({'abc'[j]}) {ttl}", fontsize=9.5, loc='left', pad=5)

fig.tight_layout(w_pad=1.6)
fig.savefig(f'{EJIM}/Figure_2_calibration.png', dpi=300)
fig.savefig(f'{EJIM}/Figure_2_calibration.svg')
plt.close(fig)

print('max bin means:', {k: round(float(np.max(v['bin_mean_predicted'])) * 100, 3)
                          for k, v in calib.items() if v.get('bin_mean_predicted')})
img = Image.open(f'{EJIM}/Figure_2_calibration.png').convert('RGB')
w, h = img.size
big = img.resize((w * 2, h * 2), Image.LANCZOS)
big.save(f'{EJIM}/Figure_2_calibration.tif', compression='tiff_lzw', dpi=(600, 600))
print('saved Figure_2_calibration.png/.svg/.tif (600 dpi)')
