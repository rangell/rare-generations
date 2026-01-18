import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# -----------------------------
# Data (stylized to match sketch)
# -----------------------------
x = np.linspace(0, 10, 800)

def gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)

# p_theta(x|c): add baseline support that persists past the harmful threshold
core = 0.8 * gauss(x, 2.0, 0.65) + 0.25 * gauss(x, 5.8, 1.0)
gate = 1 / (1 + np.exp(3.2 * (x - 6.0)))         # shuts off core on the right
# baseline = 0.015                                  # everywhere
# tail = 0.10 * (1 / (1 + np.exp(-2.2 * (x - 6.2))))  # turns on after threshold
p = core * gate #+ baseline #+ tail
p = np.clip(p, 0, None)

# q_phi(x|c): make it spikier (narrower sigmas + slightly higher peak)
q = (
    0.6 * gauss(x, 6.6, 0.65) +   # main spike near boundary
    0.03 * gauss(x, 5.10, 0.90) #+   # gentle shoulder
    # 0.010 * gauss(x, 0.25, 0.25)    # tiny left blip
) + 0.005
q = np.clip(q, 0, None)

# harmful indicator (step)
x0 = 6.25
h = (x >= x0).astype(float)

# -----------------------------
# Global styling (pretty mode)
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.linewidth": 1.1,
    "axes.edgecolor": "0.15",
    "xtick.color": "0.15",
    "ytick.color": "0.15",
    "text.color": "0.15",
    "mathtext.fontset": "stix",
    "font.family": "DejaVu Sans",
})

fig, ax = plt.subplots(figsize=(10.2, 5.9))
ax.set_facecolor("white")

# ax.grid(True, which="major", linewidth=0.9, alpha=0.14)
# ax.grid(True, which="minor", linewidth=0.6, alpha=0.08)
# ax.minorticks_on()

blue  = "#2B6CB0"
green = "#2F855A"
red   = "#C53030"
dark_red = "#991B1B"
light_red = "#EF4444"

outline = [pe.Stroke(linewidth=5.5, foreground="white", alpha=0.85), pe.Normal()]

# -----------------------------
# Light fills under p and q
# -----------------------------
ax.fill_between(x, 0, p, color=blue,  alpha=0.12, zorder=1)
ax.fill_between(x, 0, q, color=light_red, alpha=0.12, zorder=1)

# Curves
ax.plot(x, p, color=blue,  linewidth=3.2, solid_capstyle="round",
        path_effects=outline, zorder=3)
ax.plot(x, q, color=light_red, linewidth=3.2, solid_capstyle="round",
        path_effects=outline, zorder=3)
ax.plot(x, h, color=red,   linewidth=3.2, solid_capstyle="butt",
        path_effects=outline, zorder=3)

# -----------------------------
# Axes (no text/callouts; you'll do in Figma)
# -----------------------------
ax.set_xlim(0, 10)
ax.set_ylim(-0.02, 1.15)
ax.set_xticks([])
ax.set_yticks([])
# ax.set_yticks([0.0, 0.5, 1.0])
# ax.set_yticklabels(["0.0", "0.5", "1.0"])

for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(False)

# Arrowed axes (still present, but no labels)
# ax.annotate("", xy=(10.05, 0), xytext=(0, 0),
#             arrowprops=dict(arrowstyle="->", lw=1.6, color="0.15"),
#             annotation_clip=False)
# ax.annotate("", xy=(0, 1.17), xytext=(0, 0),
#             arrowprops=dict(arrowstyle="->", lw=1.6, color="0.15"),
#             annotation_clip=False)

plt.tight_layout()
plt.savefig("hero_graphic.png", transparent=False, bbox_inches="tight")
plt.close()
