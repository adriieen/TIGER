import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

# Vocal mixing frequency splits & perceptual regions
# Based on speech acoustics and vocal characterization:
# - Sub-80 Hz: Mechanical rumble, HVAC hum, plosive air blasts ("P", "B")
# - 80 – 200 Hz: Low voice fundamental (F0), chest resonance
# - 200 – 400 Hz: High voice fundamental (F0), proximity effect, sinus congestion
# - 400 – 800 Hz: First formant (F1), pharyngeal & box-like room reflections
# - 800 – 1500 Hz: Mid formants (F2), nasal/oral tract resonances, core vowel timbre
# - 1500 – 4000 Hz: Upper formants (F3, F4), consonant attack, presence & intelligibility
# - 4000 – 8000 Hz: Air friction from sibilant consonants ("S", "T", "CH")
vocal_bands = [
    (10, 80, "Sub-80 Hz", "Rumble &\nplosives"),
    (80, 200, "80 – 200 Hz", "Low voice $F_0$\n& chest"),
    (200, 400, "200 – 400 Hz", "High voice $F_0$\n& proximity"),
    (400, 800, "400 – 800 Hz", "Formant $F_1$\n& boxiness"),
    (800, 1500, "0.8 – 1.5 kHz", "Mid formants $F_2$\n& nasality"),
    (1500, 4000, "1.5 – 4 kHz", "Upper formants $F_{3,4}$\n& presence"),
    (4000, 8000, "4 – 8 kHz", "Sibilance &\nair friction"),
]
vocal_boundaries = [80, 200, 400, 800, 1500, 4000]
all_ticks = [10, 80, 200, 400, 800, 1500, 4000, 8000]

def plot_subband_split(f_edges, n_sb, title, filename, max_y=52):
    widths = [f_edges[i+1] - f_edges[i] for i in range(len(f_edges) - 1)]
    fig, ax = plt.subplots(figsize=(10.5, 6))

    # Plot sub-bands
    ax.bar(
        f_edges[:-1],
        n_sb,
        width=widths,
        align='edge',
        edgecolor='black',
        facecolor='#cbe3f3',
        alpha=0.6,
        linewidth=1.2,
        label='Model sub-bands',
    )

    # Add number etiquette on top of each bin
    for i, count in enumerate(n_sb):
        geo_mean = np.sqrt(f_edges[i] * f_edges[i+1])
        ax.text(
            geo_mean,
            count + 0.8,
            str(count),
            ha='center',
            va='bottom',
            fontsize=9.5,
            fontweight='bold',
            color='#1c3144',
            zorder=5,
        )

    # Add vertical dashed lines for vocal characterization boundaries
    for i, b in enumerate(vocal_boundaries):
        ax.axvline(
            b,
            color='#c0392b',
            linestyle='--',
            linewidth=1.4,
            alpha=0.85,
            zorder=3,
            label='Vocal region boundary' if i == 0 else '',
        )

    # Staggered vertical positions for the 7 text annotations to avoid collisions
    stagger_y = [44.0, 37.5, 44.0, 37.5, 44.0, 37.5, 36.5]
    for idx, (low, high, name, desc) in enumerate(vocal_bands):
        # Slightly offset rightmost box toward interior for generous margin
        if idx == len(vocal_bands) - 1:
            geo_mean = 5300
        else:
            geo_mean = np.sqrt(low * high)
        y_pos = stagger_y[idx]
        ax.text(
            geo_mean,
            y_pos,
            f"{name}\n{desc}",
            ha='center',
            va='center',
            fontsize=7.8,
            fontweight='bold',
            color='#7B241C',
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='#ffffff',
                edgecolor='#c0392b',
                alpha=0.88,
                linewidth=0.8,
            ),
            zorder=4,
        )

    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Frequency $f$ (Hz)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of sub-bands', fontsize=11, fontweight='bold')
    ax.set_xticks(all_ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(f_edges[0], f_edges[-1])
    ax.set_ylim(0, max_y)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved: {filename}")


def plot_actual_tiger_subbands(filename):
    # Construct individual sub-bands of original TIGER (66 total):
    # 40 bands of 25 Hz in 0-1000 Hz
    # 10 bands of 100 Hz in 1000-2000 Hz
    # 8 bands of 250 Hz in 2000-4000 Hz
    # 8 bands of 500 Hz in 4000-8000 Hz
    sb_edges = [0]
    widths = []
    for _ in range(40):
        widths.append(25)
        sb_edges.append(sb_edges[-1] + 25)
    for _ in range(10):
        widths.append(100)
        sb_edges.append(sb_edges[-1] + 100)
    for _ in range(8):
        widths.append(250)
        sb_edges.append(sb_edges[-1] + 250)
    for _ in range(8):
        widths.append(500)
        sb_edges.append(sb_edges[-1] + 500)

    sb_lefts = np.array(sb_edges[:-1], dtype=float)
    sb_lefts_plot = np.where(sb_lefts == 0, 10, sb_lefts)
    sb_rights = np.array(sb_edges[1:], dtype=float)
    plot_widths = sb_rights - sb_lefts_plot

    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot individual sub-bands showing their bandwidth
    ax.bar(
        sb_lefts_plot,
        widths,
        width=plot_widths,
        align='edge',
        edgecolor='black',
        facecolor='#cbe3f3',
        alpha=0.7,
        linewidth=0.7,
        label='Individual sub-bands (TIGER, 66 total)',
    )

    # Vocal characterization boundary lines
    for i, b in enumerate(vocal_boundaries):
        ax.axvline(
            b,
            color='#c0392b',
            linestyle='--',
            linewidth=1.4,
            alpha=0.85,
            zorder=3,
            label='Vocal region boundary' if i == 0 else '',
        )

    # Staggered annotations comfortably above max bar height (500 Hz)
    stagger_y = [590, 540, 590, 540, 590, 540, 540]
    for idx, (low, high, name, desc) in enumerate(vocal_bands):
        if idx == len(vocal_bands) - 1:
            geo_mean = 5300
        else:
            geo_mean = np.sqrt(low * high)
        ax.text(
            geo_mean,
            stagger_y[idx],
            f"{name}\n{desc}",
            ha='center',
            va='center',
            fontsize=7.8,
            fontweight='bold',
            color='#7B241C',
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='#ffffff',
                edgecolor='#c0392b',
                alpha=0.88,
                linewidth=0.8,
            ),
            zorder=4,
        )

    ax.set_xscale('log')
    ax.set_xlabel('Frequency $f$ (Hz)', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'Sub-band bandwidth $\Delta f$ (Hz)', fontsize=11, fontweight='bold')
    ax.set_xticks(all_ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(10, 8000)
    ax.set_ylim(0, 640)
    ax.set_title(
        'Actual Sub-band Distribution in Original TIGER (40×25Hz, 10×100Hz, 8×250Hz, 8×500Hz)',
        fontsize=12,
        fontweight='bold',
        pad=12,
    )
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved: {filename}")


# Base output paths
figs_dir = "/Users/adriendubois/Desktop/4A/4A_scolaire/Projects/MA1/TIGER/tests_adrien/figs"
existing_dir = os.path.join(figs_dir, "existing_split")
new_dir = os.path.join(figs_dir, "vocal_focus_splits")

# Migrate legacy 'new' folder to 'vocal_focus_splits'
legacy_new_dir = os.path.join(figs_dir, "new")
if os.path.exists(legacy_new_dir) and not os.path.exists(new_dir):
    os.rename(legacy_new_dir, new_dir)
    print(f"Renamed {legacy_new_dir} -> {new_dir}")
elif not os.path.exists(new_dir):
    os.makedirs(new_dir, exist_ok=True)

os.makedirs(existing_dir, exist_ok=True)

# ==============================================================================
# 1. EXISTING SPLITS (The 3 reference splits from TIGER paper / README.md)
# ==============================================================================

# A. Low-Frequency Narrow split (TIGER Baseline, 66 sub-bands)
# Uniform 25 Hz in 0-1000 Hz, 100 Hz in 1000-2000 Hz, 250 Hz in 2000-4000 Hz, 500 Hz in 4000-8000 Hz:
# - [10, 80 Hz]: 3 bands (80/25 = 3.2 -> 3)
# - [80, 200 Hz]: 5 bands ((200-80)/25 = 120/25 = 4.8 -> 5)
# - [200, 400 Hz]: 8 bands (200/25 = 8)
# - [400, 800 Hz]: 16 bands (400/25 = 16)
# - [800, 1500 Hz]: 13 bands (8 of 25 Hz in 800-1000 + 5 of 100 Hz in 1000-1500)
# - [1500, 4000 Hz]: 13 bands (5 of 100 Hz in 1500-2000 + 8 of 250 Hz in 2000-4000)
# - [4000, 8000 Hz]: 8 bands (8 of 500 Hz in 4000-8000)
# Total = 3 + 5 + 8 + 16 + 13 + 13 + 8 = 66 sub-bands
plot_subband_split(
    f_edges=[10, 80, 200, 400, 800, 1500, 4000, 8000],
    n_sb=[3, 5, 8, 16, 13, 13, 8],
    title="Low Frequency Narrow Split (TIGER Baseline, 66 sub-bands)",
    filename=os.path.join(existing_dir, "low_freq_narrow_split.png"),
)
shutil.copyfile(
    os.path.join(existing_dir, "low_freq_narrow_split.png"),
    os.path.join(existing_dir, "[OG]low_freq_narrow_split.png"),
)

# B. Normal Split (47 sub-bands)
# 20 bands of 50 Hz in 0-1000 Hz, 10 of 100 Hz in 1000-2000 Hz, 8 of 250 Hz in 2000-4000 Hz, 8 of 500 Hz in 4000-8000 Hz + 1 residual:
# - [10, 80 Hz]: 2 bands
# - [80, 200 Hz]: 2 bands
# - [200, 400 Hz]: 4 bands
# - [400, 800 Hz]: 8 bands
# - [800, 1500 Hz]: 9 bands (4 of 50 Hz in 800-1000 + 5 of 100 Hz in 1000-1500)
# - [1500, 4000 Hz]: 13 bands (5 of 100 Hz in 1500-2000 + 8 of 250 Hz in 2000-4000)
# - [4000, 8000 Hz]: 9 bands (8 of 500 Hz + 1 residual)
# Total = 2 + 2 + 4 + 8 + 9 + 13 + 9 = 47 sub-bands
plot_subband_split(
    f_edges=[10, 80, 200, 400, 800, 1500, 4000, 8000],
    n_sb=[2, 2, 4, 8, 9, 13, 9],
    title="Normal Split (47 sub-bands)",
    filename=os.path.join(existing_dir, "normal_split.png"),
)

# C. Even Split (67 sub-bands)
# Uniform 100 Hz bands in 0-6600 Hz (66 bands) + 1 residual in 6600-8000 Hz:
# - [10, 80 Hz]: 1 band
# - [80, 200 Hz]: 1 band
# - [200, 400 Hz]: 2 bands
# - [400, 800 Hz]: 4 bands
# - [800, 1500 Hz]: 7 bands
# - [1500, 4000 Hz]: 25 bands
# - [4000, 8000 Hz]: 27 bands (26 bands in 4-6.6 kHz + 1 residual)
# Total = 1 + 1 + 2 + 4 + 7 + 25 + 27 = 67 sub-bands
plot_subband_split(
    f_edges=[10, 80, 200, 400, 800, 1500, 4000, 8000],
    n_sb=[1, 1, 2, 4, 7, 25, 27],
    title="Even Split (67 sub-bands, uniform 100 Hz up to 6.6 kHz)",
    filename=os.path.join(existing_dir, "even_split.png"),
)

# D. Raw individual sub-bands of TIGER mapped to new vocal boundaries
plot_actual_tiger_subbands(
    filename=os.path.join(existing_dir, "original_actual_subbands.png")
)

# ==============================================================================
# 2. NEW VOCAL FOCUS SPLITS (Adapted to the refined vocal characterization grid)
# Keeping 66 sub-bands total to match TIGER baseline computational budget
# ==============================================================================

# A. Low Voice F0 Focus (80 – 200 Hz: Low voice fundamental & chest resonance)
# Boost resolution in 80-200 Hz for male voice pitch tracking and chest warmth
plot_subband_split(
    f_edges=[10, 80, 200, 400, 800, 1500, 4000, 8000],
    n_sb=[3, 15, 8, 8, 11, 13, 8],
    title="Low Voice F0 Focus Split (80–200 Hz Chest Resonance Focus, 66 sub-bands)",
    filename=os.path.join(new_dir, "low_voice_f0_focus_split.png"),
)

# B. High Voice F0 Focus (200 – 400 Hz: High voice fundamental & warmth/proximity)
# Boost resolution in 200-400 Hz for female/child pitch tracking and fullness
plot_subband_split(
    f_edges=[10, 80, 200, 400, 800, 1500, 4000, 8000],
    n_sb=[3, 5, 18, 8, 11, 13, 8],
    title="High Voice F0 Focus Split (200–400 Hz Proximity & Fullness Focus, 66 sub-bands)",
    filename=os.path.join(new_dir, "high_voice_f0_focus_split.png"),
)

# C. First Formant (F1) Focus (400 – 800 Hz: Pharyngeal reflections & vocal body)
# Boost resolution in 400-800 Hz for vowel height distinction and boxiness resolution
plot_subband_split(
    f_edges=[10, 80, 200, 400, 800, 1500, 4000, 8000],
    n_sb=[3, 5, 6, 24, 7, 13, 8],
    title="First Formant (F1) Focus Split (400–800 Hz Vocal Body Focus, 66 sub-bands)",
    filename=os.path.join(new_dir, "first_formant_f1_focus_split.png"),
)

# D. Mid Formants (F2) Focus (800 – 1500 Hz: Nasal/oral tract & core timbre)
# Boost resolution in 800-1500 Hz for speaker identity and vowel frontness/backness
plot_subband_split(
    f_edges=[10, 80, 200, 400, 800, 1500, 4000, 8000],
    n_sb=[3, 5, 6, 9, 22, 13, 8],
    title="Mid Formants (F2) Focus Split (0.8–1.5 kHz Vowel Timbre Focus, 66 sub-bands)",
    filename=os.path.join(new_dir, "mid_formants_f2_focus_split.png"),
)

# E. Upper Formants & Presence Focus (1.5 – 4 kHz: Consonant attack & speech intelligibility)
# Boost resolution in 1.5-4 kHz for speech projection and clarity
plot_subband_split(
    f_edges=[10, 80, 200, 400, 800, 1500, 4000, 8000],
    n_sb=[3, 5, 6, 11, 11, 22, 8],
    title="Upper Formants Focus Split (1.5–4 kHz Intelligibility Focus, 66 sub-bands)",
    filename=os.path.join(new_dir, "upper_formants_presence_focus_split.png"),
)

print("\nAll plots generated successfully!")