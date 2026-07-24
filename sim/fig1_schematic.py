#!/usr/bin/env python3
"""
Figure 1 — conceptual schematic of the partial-observability incentive loop.
AI-generated diagram — marked as AI-generated in the figure caption.
Pure matplotlib (no external assets). Writes ../results/figs/fig1_schematic.pdf|png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(__file__)
FIGS = os.path.join(HERE, "..", "results", "figs")
os.makedirs(FIGS, exist_ok=True)

C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#8256A8", grey="#7A7A7A", ink="#222222", light="#F4F4F4",
         obs="#DDEBF7", hid="#FBE3D6")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")

BAND = 3.9  # vertical center of the main process band


def box(x, y, w, h, text, fc, ec=C["ink"], fs=8.5, bold=False, tcol=C["ink"]):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tcol)


def arrow(x1, y1, x2, y2, color=C["ink"], lw=1.3, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=13, lw=lw, color=color, ls=ls,
                 connectionstyle=f"arc3,rad={rad}"))


def route(pts, color, lw=1.5, ls="-"):
    """Orthogonal feedback path through waypoints, arrowhead on final segment."""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                 mutation_scale=13, lw=lw, color=color, ls=ls,
                 connectionstyle="arc3,rad=0"))


ax.text(0.1, 7.6, "The Goodhart trap in incentive allocation", fontsize=11.5,
        fontweight="bold", color=C["ink"])

# ---- main process band (compact, leaving whitespace above and below) ----
box(0.35, BAND - 0.6, 2.1, 1.2, "Incentive\nallocator\n(learner)", C["light"], bold=True, fs=8.4)
box(3.0, BAND - 0.5, 2.15, 1.0, "Allocate 1 of 5\nincentives / emp.\n(budget-limited)", "#FFFFFF", fs=7.6)
box(5.75, BAND - 0.75, 2.5, 1.5, "", C["light"])
ax.text(7.0, BAND + 0.42, "Workforce", ha="center", fontsize=9.2, fontweight="bold")
ax.text(7.0, BAND - 0.05, "4 archetypes", ha="center", fontsize=7.8)
ax.text(7.0, BAND - 0.45, "latent satisfaction $s$", ha="center", fontsize=7.6,
        style="italic", color=C["purple"])
box(9.75, BAND + 0.35, 1.95, 0.95, "Retention\nOBSERVED", C["obs"], ec=C["blue"],
    fs=8.0, bold=True, tcol=C["blue"])
box(9.75, BAND - 1.3, 1.95, 0.95, "Satisfaction\nHIDDEN", C["hid"], ec=C["red"],
    fs=8.0, bold=True, tcol=C["red"])

# forward flow
arrow(2.45, BAND, 3.0, BAND)
arrow(5.15, BAND, 5.75, BAND)
arrow(8.25, BAND + 0.30, 9.75, BAND + 0.8, color=C["blue"], lw=1.5)
arrow(8.25, BAND - 0.30, 9.75, BAND - 0.85, color=C["red"], lw=1.5)

# ---- TOP loop: retention -> learner (always). Clean up-over-down routing. ----
TOP = BAND + 2.15
route([(10.72, BAND + 0.83), (10.72, TOP), (1.4, TOP), (1.4, BAND + 0.6)],
      C["blue"], lw=1.7)
ax.text(6.0, TOP + 0.28, "reward = retention   (always observed)", ha="center",
        fontsize=8.2, color=C["blue"], fontweight="bold")

# ---- handcuff forward shortcut (short arc into retention) ----
arrow(6.6, BAND + 0.78, 9.75, BAND + 0.95, color=C["orange"], lw=1.1,
      ls=(0, (3, 2)), rad=-0.3)
ax.text(6.7, BAND + 1.45, "handcuff: monetary $\\rightarrow$ retention,\nbypassing satisfaction",
        ha="center", fontsize=6.9, color=C["orange"])

# ---- BOTTOM loop: satisfaction -> learner (only via sparse audit) ----
BOT = BAND - 2.15
route([(10.72, BAND - 0.83), (10.72, BOT), (1.4, BOT), (1.4, BAND - 0.6)],
      C["red"], lw=1.4, ls=(0, (4, 3)))
ax.text(6.0, BOT - 0.32, "satisfaction reaches the learner ONLY via a sparse audit "
        "(survey fraction $p$) — otherwise invisible",
        ha="center", fontsize=7.3, color=C["red"], style="italic")
ax.text(3.4, BOT + 0.26, "dynamics: crowding-out · habituation · handcuff",
        ha="center", fontsize=6.5, color=C["grey"])

fig.savefig(os.path.join(FIGS, "fig1_schematic.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIGS, "fig1_schematic.png"), bbox_inches="tight", dpi=300)
print("wrote fig1_schematic.pdf/png")
