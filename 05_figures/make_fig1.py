"""Fig 1 — pipeline schematic: diagnose -> attribute -> repair -> validate.
High-resolution PNG for the npj DM manuscript (development in MIMIC-IV,
external validation across MIMIC-III and eICU, de-overlapped repair pool).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13.2, 7.2), dpi=300)
ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")

BOX = dict(boxstyle="round,pad=0.55,rounding_size=0.25")
C = {
    "dev": "#DDEBF7", "fail": "#FCE4EC", "attr": "#FFF9C4",
    "repair": "#E8F5E9", "val": "#EDE7F6", "comp": "#ECEFF1",
    "edge": "#455A64", "hi": "#1A237E",
}

def box(x, y, w, h, text, fc, fs=11, tcol="#212121"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.18",
                                fc=fc, ec=C["edge"], lw=1.3, mutation_aspect=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tcol, linespacing=1.5)

def arrow(x1, y1, x2, y2, color=C["edge"], lw=1.8, label=None, lx=None, ly=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, lw=lw, color=color))
    if label:
        ax.text(lx if lx else (x1 + x2) / 2, ly if ly else (y1 + y2) / 2 + 0.25,
                label, ha="center", va="center", fontsize=9.5, color="#37474F",
                style="italic")

# --- top banner ---
ax.text(6.5, 7.55, "Diagnosing and repairing transportability failure of clinical ML\n"
        "— HA-VTE risk prediction across EHR systems —",
        ha="center", va="center", fontsize=14, fontweight="bold", color=C["hi"],
        linespacing=1.4)

# --- row 1: development + frozen failure ---
box(0.4, 5.4, 2.6, 1.4, "Development\nMIMIC-IV (v3.1)\n58 first-24h features\nXGB / LR (internal AUC 0.83)",
    C["dev"], fs=10)
box(5.0, 5.4, 2.7, 1.4, "Frozen transport\n→ MIMIC-III (cross-version)\n→ eICU (208 hospitals)",
    C["fail"], fs=10)
box(9.4, 5.4, 2.7, 1.4, "Collapse to Padua\nΔAUC −0.009 (p=0.69)\nvs +0.17 after local retraining",
    C["fail"], fs=10)

# --- row 2: attribution ---
box(0.4, 3.3, 2.6, 1.2, "Attribution\n(a) ICU case-mix saturation\n(b) outcome ascertainment\n(c) feature-coverage drift",
    C["attr"], fs=9.5)
box(5.0, 3.3, 2.7, 1.2, "Key logic\nfailure of the frozen\nparameterization, not\nthe prediction task",
    C["attr"], fs=9.5)

# --- row 3: repair ---
box(0.4, 1.0, 3.6, 1.5, "Repair — de-overlapped,\nversionally diverse pool\nMIMIC-III (2001–2012)\n+ MIMIC-IV (2014–2022)\nn=77,832 · 1,035 events",
    C["repair"], fs=10)
box(5.0, 1.0, 2.7, 1.5, "Independent validation\neICU · 10,000 cluster bootstrap\nAUC 0.693 (0.667–0.718)\nΔ vs Padua +0.043 (p=0.015)",
    C["val"], fs=9.5)
box(9.0, 1.0, 3.0, 1.5, "Ablation — diversity, not size\nIV-2014+ only 0.669\nIII only 0.603\npooled 0.693  (de-overlap got smaller)",
    C["val"], fs=9.5)

# --- comparator banner bottom ---
box(9.4, 3.3, 2.7, 1.2, "Comparator (same structured data)\nPadua · IMPROVE\npre-specified primary endpoint",
    C["comp"], fs=9.5)

# --- arrows ---
arrow(3.05, 6.1, 4.95, 6.1)
arrow(7.75, 6.1, 9.35, 6.1, label="failure", lx=8.55, ly=6.45)
arrow(2.7, 5.35, 2.2, 4.6)                     # dev -> attr
arrow(7.5, 5.35, 6.35, 4.6)                    # frozen -> key logic
arrow(1.7, 3.25, 2.7, 2.65, label="diagnose→repair", lx=2.1, ly=3.1)
arrow(6.35, 3.25, 6.35, 2.6)                   # key logic -> validation
arrow(2.1, 1.0, 5.6, 1.75, label="validate", lx=3.7, ly=1.5)   # repair -> val
arrow(8.0, 1.75, 8.95, 1.45)                   # val -> ablation

fig.savefig("ndm/figures/fig1_pipeline_v1.png", bbox_inches="tight", facecolor="white")
print("saved ndm/figures/fig1_pipeline_v1.png")
