import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

# ============================================================
# DATA extracted from Codabench Final Submission Phase PDF
# ============================================================

# Best score per approach (best single checkpoint from Final Test Phase PDF)
approaches = {
    'Appr. 1\n(RPPO+\nLSTM)': -1447.55,
    'Appr. 2\n(D3QN+\nGRU+PER)': -25715.40,
    'Appr. 3\n(Discrete\nSAC+GRU)': -1655.64,
    'Appr. 5\n(Simple\nDDQN)': -156899.23,
    'Appr. 6\n(D3QN v2\nNo Curr.)': -9435.41,
    'Appr. 7\n(D3QN+\nPBRS)': -1540.74,
}
# Note: Approach 4 (Hybrid Heuristic) was not submitted to the Final Test Phase

# Approach 1: Score vs Episode (from PDF)
a1_data = {
    200: -381539.57, 400: -381539.57, 600: -381539.57, 800: -381539.57,
    1000: -381539.57, 1200: -48826.36, 1400: -239800.01, 1600: -357919.69,
    1800: -357919.67, 2200: -381529.16, 2400: -172816.35, 2600: -381491.81,
    2800: -387621.19, 3000: -316428.62, 3200: -381539.57, 3400: -381539.57,
    3600: -379403.51, 3800: -381539.57, 4000: -381539.57, 4200: -381539.57,
    4400: -13484.71, 4600: -277367.18, 4800: -4081.87, 5000: -144075.02,
    5200: -1738.57, 5400: -239597.15, 5600: -188285.05, 5800: -381539.57,
    6000: -223797.41, 6200: -169388.94, 6400: -25353.53, 6600: -322567.71,
    6800: -192841.96, 7000: -191610.71, 7200: -1447.55, 7400: -1561.20,
    7600: -324368.13, 7800: -183062.90, 8000: -308839.25, 8200: -288317.90,
    8400: -198712.43, 8600: -237211.47, 8800: -338931.95, 9000: -242269.88,
    9200: -202665.70, 9400: -239299.07, 9600: -347896.84, 9800: -357795.31,
    10000: -40946.81, 10200: -322220.99, 10400: -36582.77, 10600: -43010.75,
    10800: -123648.19, 11000: -345736.08, 11200: -105992.32,
    11400: -381529.20, 11600: -319874.64, 11800: -87164.12,
    12000: -340864.29, 12200: -318425.46, 12400: -256250.89,
    12600: -99634.35, 12800: -54942.35, 13000: -243074.03,
    13200: -67342.44, 13400: -95204.97, 13600: -205040.31,
    13800: -373906.36, 14000: -175967.05, 14200: -194577.74,
    14400: -107900.64, 14600: -95478.15, 14800: -93476.90, 15000: -111481.64,
}

# Approach 3: Score vs Episode
a3_data = {
    200: -381539.57, 400: -381539.57, 600: -204873.70, 800: -381539.57,
    1000: -190343.53, 1200: -381381.56, 1400: -381539.57, 1600: -381419.57,
    1800: -365838.67, 2000: -1655.64, 2200: -1828.00, 2400: -1828.41,
    2600: -1828.51, 2800: -1828.51, 3000: -1828.51, 3200: -1828.51,
    3400: -1828.59, 3600: -1828.38, 3800: -1828.38, 4000: -1828.53,
    4200: -1828.57, 4400: -1828.57, 4600: -1828.57, 4800: -1828.53,
    5000: -1828.48, 5200: -1828.38, 5400: -1828.38, 5600: -1828.38,
    5800: -1828.12, 6000: -1828.48, 6200: -1828.38, 6400: -1827.73,
    6600: -1827.76, 6800: -1827.76, 7000: -1828.13, 7200: -1828.37,
    7400: -1828.25, 7600: -1827.81, 7800: -1828.37, 8000: -1828.47,
}

# Approach 7: Score vs Episode (sorted by episode)
a7_data = {
    50: -205219.42, 100: -159385.08, 150: -265222.47, 200: -156668.03,
    250: -177725.31, 300: -222556.39, 350: -140103.66, 400: -143111.71,
    450: -87184.01, 500: -136865.98, 550: -13933.26, 600: -139109.01,
    650: -109850.47, 700: -109627.05, 750: -9790.77, 800: -50655.66,
    850: -144023.19, 900: -18901.11, 950: -13906.03, 1000: -18653.47,
    1050: -16285.33, 1100: -23688.23, 1150: -10904.75, 1200: -16123.34,
    1250: -6379.57, 1300: -15655.36, 1350: -51790.35, 1400: -12564.12,
    1450: -154128.51, 1500: -14869.18, 1550: -117626.16, 1600: -12678.29,
    1650: -190982.95, 1700: -145590.61, 1750: -34029.41, 1800: -21998.73,
    1850: -141539.31, 1900: -130302.27, 1950: -14688.40, 2000: -135337.95,
    2050: -144887.73, 2100: -31583.45, 2150: -27387.73, 2200: -26943.85,
    2250: -27722.61, 2300: -9352.89, 2350: -14458.99, 2400: -30077.74,
    2450: -6833.29, 2500: -13952.55, 2550: -2408.52, 2600: -123426.75,
    2650: -122125.73, 2700: -12082.59, 2750: -2601.08, 2800: -1904.76,
    2850: -3193.37, 2900: -4463.67, 2950: -13243.03, 3000: -2299.49,
    3050: -5101.18, 3100: -5528.28, 3150: -8615.37, 3200: -10617.07,
    3250: -5782.32, 3300: -8031.89, 3350: -15279.09, 3400: -9023.21,
    3450: -7015.75, 3500: -16518.81, 3550: -6610.13, 3600: -116307.41,
    3650: -19239.29, 3700: -3102.00, 3750: -6111.92, 3800: -10497.87,
    3850: -10401.99, 3900: -6532.52, 3950: -11809.37, 4000: -6472.88,
    4050: -2033.55, 4100: -139730.90, 4150: -18820.60, 4200: -77326.59,
    4250: -13759.97, 4300: -11355.99, 4350: -128405.91, 4400: -13685.05,
    4450: -4156.82, 4500: -41542.31, 4550: -19902.61, 4600: -22498.03,
    4650: -10572.55, 4700: -17344.83, 4750: -11086.68, 4800: -29737.69,
    4850: -4018.17, 4900: -5121.81, 4950: -6886.51, 5000: -145859.91,
    5050: -4734.00, 5100: -2527.16, 5150: -14697.32, 5200: -1540.74,
    5250: -5427.31, 5300: -4167.59, 5350: -6610.13, 5400: -106647.86,
    5450: -10029.52, 5500: -7573.76, 5550: -10669.04, 5600: -47736.95,
    5650: -14052.95, 5700: -3776.81, 5750: -13092.05, 5800: -26135.05,
    5850: -8745.51, 5900: -48625.43, 5950: -8914.11,
}

# ============================================================
# PLOT 1: Best Score Comparison Across Approaches (Bar Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)

names = list(approaches.keys())
scores = list(approaches.values())

# Color the bars: top 3 get highlighted
colors = []
sorted_scores = sorted(scores, reverse=True)
for s in scores:
    if s == sorted_scores[0]:
        colors.append('#2ecc71')  # best = green
    elif s == sorted_scores[1]:
        colors.append('#3498db')  # 2nd = blue
    elif s == sorted_scores[2]:
        colors.append('#e67e22')  # 3rd = orange
    else:
        colors.append('#95a5a6')  # rest = gray

bars = ax.bar(names, scores, color=colors, edgecolor='black', linewidth=0.5)

# Add score labels on top of bars
for bar, score in zip(bars, scores):
    y_off = max(abs(score) * 0.015, 4500)
    label = f'{score:.0f}' if score > -10000 else f'{score/1000:.1f}K'
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        score + y_off,
        label,
        ha='center',
        va='bottom',
        fontsize=7.3,
        fontweight='bold' if score > -10000 else None,
        color='#222' if score > -10000 else '#555',
        clip_on=False,
    )

ax.set_ylabel('Codabench Score (higher is better)', fontsize=10)
ax.set_title('Best Codabench Score per Approach (Final Test Phase)', fontsize=11, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.3)
ax.set_ylim(bottom=min(scores)*1.18, top=12000)
ax.tick_params(axis='x', labelsize=7.5)
ax.tick_params(axis='y', labelsize=8)
ax.grid(axis='y', linestyle=':', alpha=0.35)

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', edgecolor='black', label='1st'),
                   Patch(facecolor='#3498db', edgecolor='black', label='2nd'),
                   Patch(facecolor='#e67e22', edgecolor='black', label='3rd')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8, title='Rank', title_fontsize=8)

plt.savefig('c:/Users/neeld/Downloads/CS780_PROJECT/Report Template/16_best_score_per_approach.png', dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.close()
print("Plot 1 saved: 16_best_score_per_approach.png")


# ============================================================
# PLOT 2: Score vs Episode for Top 3 Approaches (1, 3, 7)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), constrained_layout=True)

# --- Approach 1 ---
ax = axes[0]
eps1 = sorted(a1_data.keys())
sc1 = [a1_data[e] for e in eps1]
ax.plot(eps1, sc1, 'o-', color='#2ecc71', markersize=1.5, linewidth=0.7, alpha=0.8)
best_ep1 = min(a1_data, key=lambda k: -a1_data[k])  # max score
ax.plot(best_ep1, a1_data[best_ep1], '*', color='red', markersize=10, zorder=5)
ax.annotate(f'Best: {a1_data[best_ep1]:.0f}\n(ep{best_ep1})',
            xy=(best_ep1, a1_data[best_ep1]), xytext=(14, -18), textcoords='offset points',
            fontsize=6.4, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='red', linewidth=0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
ax.set_title('Approach 1 (RPPO+LSTM)', fontsize=8, fontweight='bold')
ax.set_xlabel('Training Episode', fontsize=7)
ax.set_ylabel('Codabench Score', fontsize=7)
ax.tick_params(labelsize=6)
ax.set_ylim(bottom=-400000, top=5000)
ax.grid(linestyle=':', alpha=0.3)

# --- Approach 3 ---
ax = axes[1]
eps3 = sorted(a3_data.keys())
sc3 = [a3_data[e] for e in eps3]
ax.plot(eps3, sc3, 'o-', color='#e67e22', markersize=2, linewidth=0.8)
best_ep3 = min(a3_data, key=lambda k: -a3_data[k])
ax.plot(best_ep3, a3_data[best_ep3], '*', color='red', markersize=10, zorder=5)
ax.annotate(f'Best: {a3_data[best_ep3]:.0f}\n(ep{best_ep3})',
        xy=(best_ep3, a3_data[best_ep3]), xytext=(12, -18), textcoords='offset points',
        fontsize=6.4, color='red', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='red', linewidth=0.5),
        arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
ax.set_title('Approach 3 (Discrete SAC)', fontsize=8, fontweight='bold')
ax.set_xlabel('Training Episode', fontsize=7)
ax.tick_params(labelsize=6)
ax.set_ylim(bottom=-400000, top=5000)
# Highlight the stable plateau region
ax.axhspan(-1830, -1650, alpha=0.15, color='green')
ax.text(5500, -1738, 'Stable\nplateau', fontsize=5.5, color='green', ha='center', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='green', linewidth=0.4))
ax.grid(linestyle=':', alpha=0.3)

# --- Approach 7 ---
ax = axes[2]
eps7 = sorted(a7_data.keys())
sc7 = [a7_data[e] for e in eps7]
ax.plot(eps7, sc7, 'o-', color='#3498db', markersize=1.2, linewidth=0.6, alpha=0.8)
best_ep7 = min(a7_data, key=lambda k: -a7_data[k])
ax.plot(best_ep7, a7_data[best_ep7], '*', color='red', markersize=10, zorder=5)
ax.annotate(f'Best: {a7_data[best_ep7]:.0f}\n(ep{best_ep7})',
            xy=(best_ep7, a7_data[best_ep7]), xytext=(-72, -18), textcoords='offset points',
            fontsize=6.4, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='red', linewidth=0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
ax.set_title('Approach 7 (D3QN+PBRS)', fontsize=8, fontweight='bold')
ax.set_xlabel('Training Episode', fontsize=7)
ax.tick_params(labelsize=6)
ax.set_ylim(bottom=min(sc7) * 1.05, top=5000)
ax.grid(linestyle=':', alpha=0.3)

fig.suptitle('Codabench Score vs Training Episode (Final Test Phase)', fontsize=10, fontweight='bold', y=1.02)
plt.savefig('c:/Users/neeld/Downloads/CS780_PROJECT/Report Template/17_score_vs_episode.png', dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.close()
print("Plot 2 saved: 17_score_vs_episode.png")


# ============================================================
# PLOT 3: Zoomed-in view of Approach 3 plateau + Approach 7 good region
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)

# Approach 3: zoom into the converged plateau (ep2200+, excluding ep2000 outlier at -1655)
a3_plateau = {k: v for k, v in a3_data.items() if k >= 2200}
eps3p = sorted(a3_plateau.keys())
sc3p = [a3_plateau[e] for e in eps3p]
ax1.plot(eps3p, sc3p, 'o-', color='#e67e22', markersize=3, linewidth=1)
ax1.fill_between(eps3p, min(sc3p)-1, max(sc3p)+1, alpha=0.1, color='#e67e22')
ax1.set_title('Approach 3: Converged Policy Plateau', fontsize=8.5, fontweight='bold')
ax1.set_xlabel('Training Episode', fontsize=7.5)
ax1.set_ylabel('Codabench Score', fontsize=7.5)
ax1.tick_params(labelsize=7)
ax1.set_ylim(min(sc3p)-2, max(sc3p)+2)
ax1.grid(linestyle=':', alpha=0.3)
# Mark best and worst
best3 = max(sc3p)
worst3 = min(sc3p)
ax1.axhline(y=best3, color='green', linewidth=0.5, linestyle='--', alpha=0.7)
ax1.axhline(y=worst3, color='red', linewidth=0.5, linestyle='--', alpha=0.7)
ax1.text(7500, best3+0.3, f'Best: {best3:.2f}', fontsize=6, color='green')
ax1.text(7500, worst3-0.8, f'Worst: {worst3:.2f}', fontsize=6, color='red')
range_val = best3 - worst3
ax1.text(4000, (best3+worst3)/2, f'Range: {range_val:.2f}\n(remarkably stable)',
         fontsize=6.5, ha='center', style='italic', color='#555',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

# Approach 7: zoom into episodes 2300-5200 (the good region)
a7_good = {k: v for k, v in a7_data.items() if 2300 <= k <= 5200}
eps7g = sorted(a7_good.keys())
sc7g = [a7_good[e] for e in eps7g]
ax2.plot(eps7g, sc7g, 'o-', color='#3498db', markersize=2.5, linewidth=0.8)
ax2.set_title('Approach 7: Best Checkpoint Region (ep2300-5200)', fontsize=8.5, fontweight='bold')
ax2.set_xlabel('Training Episode', fontsize=7.5)
ax2.set_ylabel('Codabench Score', fontsize=7.5)
ax2.tick_params(labelsize=7)
ax2.grid(linestyle=':', alpha=0.3)
# Mark best
best_ep = min(a7_good, key=lambda k: -a7_good[k])
ax2.plot(best_ep, a7_good[best_ep], '*', color='red', markersize=12, zorder=5)
ax2.annotate(f'Best: {a7_good[best_ep]:.0f} (ep{best_ep})',
             xy=(best_ep, a7_good[best_ep]), xytext=(-88, -18), textcoords='offset points',
             fontsize=7, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='red', linewidth=0.5),
             arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
# Show the variance
ax2.axhline(y=-5000, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
ax2.text(2400, -4500, 'High variance across checkpoints', fontsize=6, color='gray', style='italic')

plt.savefig('c:/Users/neeld/Downloads/CS780_PROJECT/Report Template/18_zoomed_analysis.png', dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.close()
print("Plot 3 saved: 18_zoomed_analysis.png")

print("\nAll 3 plots generated successfully!")
