import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

# Music production frequency splits & perceptual regions
music_bands = [
    (10, 80, "Sub-80 Hz", "Rumble & plosives"),
    (80, 300, "80 – 300 Hz", "Body &\nfundamental"),
    (300, 1000, "300 Hz – 1 kHz", "Boxiness &\nhollow tones"),
    (1000, 4000, "1 – 4 kHz", "Presence &\nintelligibility"),
    (4000, 8000, "4 – 8 kHz", "Sibilance"),
]
music_boundaries = [80, 300, 1000, 4000]
all_ticks = [10, 80, 300, 1000, 2000, 4000, 8000]

def plot_subband_split(f_edges, n_sb, title, filename):
    widths = [f_edges[i+1] - f_edges[i] for i in range(len(f_edges) - 1)]
    fig, ax = plt.subplots(figsize=(9, 5.5))

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

    # Add vertical dashed lines for music production boundaries
    for i, b in enumerate(music_boundaries):
        ax.axvline(
            b,
            color='#c0392b',
            linestyle='--',
            linewidth=1.4,
            alpha=0.85,
            zorder=3,
            label='Music production boundary' if i == 0 else '',
        )

    # Add annotations for each music production band centered in log space
    for low, high, name, desc in music_bands:
        geo_mean = np.sqrt(low * high)
        ax.text(
            geo_mean,
            38,
            f"{name}\n{desc}",
            ha='center',
            va='center',
            fontsize=8,
            fontweight='bold',
            color='#7B241C',
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='#ffffff',
                edgecolor='#c0392b',
                alpha=0.85,
                linewidth=0.8,
            ),
            zorder=4,
        )

    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('f (Hz)', fontsize=11)
    ax.set_ylabel('number of sub-bands', fontsize=11)
    ax.set_xticks(all_ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(f_edges[0], f_edges[-1])
    ax.set_ylim(0, 48)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved: {filename}")


def plot_actual_tiger_subbands(filename):
    # Construct individual sub-bands of original TIGER:
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

    fig, ax = plt.subplots(figsize=(10, 5.5))

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

    # Music production boundary lines
    for i, b in enumerate(music_boundaries):
        ax.axvline(
            b,
            color='#c0392b',
            linestyle='--',
            linewidth=1.4,
            alpha=0.85,
            zorder=3,
            label='Music production boundary' if i == 0 else '',
        )

    # Annotations
    for low, high, name, desc in music_bands:
        geo_mean = np.sqrt(low * high)
        ax.text(
            geo_mean,
            350,
            f"{name}\n{desc}",
            ha='center',
            va='center',
            fontsize=8,
            fontweight='bold',
            color='#7B241C',
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='#ffffff',
                edgecolor='#c0392b',
                alpha=0.85,
                linewidth=0.8,
            ),
            zorder=4,
        )

    ax.set_xscale('log')
    ax.set_xlabel('f (Hz)', fontsize=11)
    ax.set_ylabel(r'Sub-band bandwidth $\Delta f$ (Hz)', fontsize=11)
    ax.set_xticks(all_ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(10, 8000)
    ax.set_ylim(0, 550)
    ax.set_title(
        'Actual Sub-band Distribution in Original TIGER (40×25Hz, 10×100Hz, 8×250Hz, 8×500Hz)',
        fontsize=12,
        fontweight='bold',
    )
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved: {filename}")


out_dir = "/Users/adriendubois/Desktop/4A/4A_scolaire/Projects/MA1/TIGER/tests_adrien/figs/rough_split"

# 1. Original TIGER split with actual sub-band distribution in 0-1000 Hz
# (40 sub-bands of 25 Hz: ~3 in Sub-80 Hz, ~9 in 80-300 Hz, and 28 in 300-1000 Hz)
plot_subband_split(
    f_edges=[10, 80, 300, 1000, 2000, 4000, 8000],
    n_sb=[3, 9, 28, 10, 8, 8],
    title="Low Frequency Narrow split (TIGER actual sub-band distribution)",
    filename=os.path.join(out_dir, "low_freq_narrow_split.png"),
)

# 2. Actual sub-bands plot showing all individual 25Hz/100Hz/250Hz/500Hz bands
plot_actual_tiger_subbands(
    filename=os.path.join(out_dir, "original_actual_subbands.png")
)



# 4. Fundamental focus: 10/20/10 in [10, 80, 300, 1000]
plot_subband_split(
    f_edges=[10, 80, 300, 1000, 2000, 4000, 8000],
    n_sb=[5, 25, 10, 10, 8, 8],
    title="Fundamental Focus (5/25/10 split in 0-1000 Hz)",
    filename=os.path.join(out_dir, "fundamental_focus_split.png"),
)

# 5. Boxiness focus: 10/10/20 in [10, 80, 300, 1000]
plot_subband_split(
    f_edges=[10, 80, 300, 1000, 2000, 4000, 8000],
    n_sb=[5, 10, 25, 10, 8, 8],
    title="Boxiness Focus (5/10/25 split in 0-1000 Hz) ~ tiger ",
    filename=os.path.join(out_dir, "boxiness_focus_split.png"),
)

# 6. Even split
plot_subband_split(
    f_edges=[10, 80, 300, 1000, 2000, 4000, 6600,8000],
    n_sb=[1, 2, 7, 10, 20, 26, 1],
    title="Even split, focused on the interval [0, 6600]Hz ",
    filename=os.path.join(out_dir, "even_split.png"),
)

# 7. Normal split 
plot_subband_split(
    f_edges=[10, 80, 300, 1000, 2000, 4000,8000],
    n_sb=[1, 5, 14, 10, 8, 8],
    title="Normal split (47 subbands) ",
    filename=os.path.join(out_dir, "normal_split.png"),
)

# 8. Interpolated normal split (same nb of subbands)