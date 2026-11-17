"""Schematic timelines of one pointcloud: motion samples, point stamps, and the pose accumulator.

1D motion, arbitrary units. Each corrector's accumulator is simulated with its own matching and
integration rules; poses relative to another reference are shifted into the header-stamp frame.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SAMPLE_STAMPS = np.array([10.0, 30.0, 50.0, 70.0, 90.0, 110.0])  # ms after header.stamp
SAMPLE_VELOCITIES = np.array([8.0, 8.0, 3.0, 3.0, 8.0, 8.0])
MAX_SAMPLE_DISTANCE = 100.0

NOTE_CUDA = "older than point 0, the reference:\ndt clamped to 0, not corrected"
NOTE_CPU_BACK = "every step back at the velocity of s\u2085,\nzero once s\u2085 is more than 100 ms away"
NOTE_CPU_GAP = "the next sample's velocity\nspans the whole gap"
NOTE_CPU_JUMP = "time jumps back, but s\u2085 is 108 ms away:\nzero velocity, no step back"

THEMES = {
    "light": dict(
        surface="#ffffff", text="#0b0b0b", text2="#52514e", grid="#e4e3df", fill="#e9e8e4",
        table="#2a78d6", cpu="#eb6834", cuda="#1baf7a",
    ),
    "dark": dict(
        surface="#0d1117", text="#f0efec", text2="#c3c2b7", grid="#30302e", fill="#2b2b29",
        table="#3987e5", cpu="#d95926", cuda="#199e70",
    ),
}


def velocity_at(t):
    """The velocity of the first sample stamped at or after t (the last one past the end)."""
    i = min(np.searchsorted(SAMPLE_STAMPS, t, side="left"), len(SAMPLE_STAMPS) - 1)
    return SAMPLE_VELOCITIES[i]


def exact_position(t):
    """Exact integral of the sample velocities from the header stamp to t."""
    edges = np.concatenate([[0.0], SAMPLE_STAMPS])
    x = 0.0
    for lo, hi, v in zip(edges[:-1], edges[1:], SAMPLE_VELOCITIES):
        if t <= lo:
            break
        x += v * (min(t, hi) - lo)
    return x


def main_cpu(points):
    """main's CPU corrector: one pose carried along the array, forward-only sample iterator."""
    i = min(np.searchsorted(SAMPLE_STAMPS, points[0], side="left"), len(SAMPLE_STAMPS) - 1)
    x, previous, poses = 0.0, points[0], []
    for t in points:
        while i < len(SAMPLE_STAMPS) - 1 and t > SAMPLE_STAMPS[i]:
            i += 1
        valid = abs(t - SAMPLE_STAMPS[i]) <= MAX_SAMPLE_DISTANCE
        x += (SAMPLE_VELOCITIES[i] if valid else 0.0) * (t - previous)
        previous = t
        poses.append(x)
    return np.array(poses) + exact_position(points[0])


def main_cuda(points):
    """main's CUDA kernels: reference is the first point, older samples dropped, dt clamped to 0."""
    reference = points[0]
    kept = [(s, v) for s, v in zip(SAMPLE_STAMPS, SAMPLE_VELOCITIES) if s >= reference]
    structs, cumulative, last = [], 0.0, reference
    for stamp, v in kept:
        structs.append((stamp, v, cumulative, last))
        cumulative += v * (stamp - last)
        last = stamp
    poses = []
    for t in points:
        stamp, v, cumulative, last = next((s for s in structs if s[0] >= t), structs[-1])
        poses.append(cumulative + v * max(0.0, t - last))
    return np.array(poses) + exact_position(reference)


def plot(path, theme, rings, title, series, notes=(), legend_loc="upper left"):
    """`rings`: list of (label, stamps in array order) with consecutive array indices.
    `series`: subset of {"cpu", "cuda"} to compare against the motion table."""
    c = THEMES[theme]
    plt.rcParams.update({
        "font.size": 9, "text.color": c["text"], "axes.labelcolor": c["text2"],
        "xtick.color": c["text2"], "ytick.color": c["text2"], "axes.edgecolor": c["grid"],
        "svg.fonttype": "path",
    })
    points = np.concatenate([stamps for _, stamps in rings])
    fig, (ax_v, ax_p, ax_x) = plt.subplots(
        3, 1, figsize=(8.4, 5.6), sharex=True, facecolor=c["surface"],
        gridspec_kw=dict(height_ratios=[1.0, 0.35 + 0.3 * len(rings), 3.2], hspace=0.12))
    for ax in (ax_v, ax_p, ax_x):
        ax.set_facecolor(c["surface"])
        for stamp in SAMPLE_STAMPS:
            ax.axvline(stamp, color=c["grid"], lw=1, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.set_yticks([])

    # Motion samples: the velocity each time takes from its sample
    edges = np.concatenate([[0.0], SAMPLE_STAMPS])
    ax_v.fill_between(edges, np.concatenate([[SAMPLE_VELOCITIES[0]], SAMPLE_VELOCITIES]),
                      step="pre", color=c["fill"], lw=0)
    ax_v.step(edges, np.concatenate([[SAMPLE_VELOCITIES[0]], SAMPLE_VELOCITIES]), where="pre",
              color=c["text2"], lw=1.2)
    ax_v.plot(SAMPLE_STAMPS, SAMPLE_VELOCITIES, "v", color=c["text"], ms=6, clip_on=False)
    for k, (s, v) in enumerate(zip(SAMPLE_STAMPS, SAMPLE_VELOCITIES)):
        ax_v.annotate(f"s{chr(0x2080 + k)}", (s, v), xytext=(0, 6), textcoords="offset points",
                      ha="center", va="bottom", color=c["text2"], fontsize=8)
    ax_v.set_ylim(0, SAMPLE_VELOCITIES.max() * 1.55)
    ax_v.set_ylabel("motion\nsamples", rotation=0, ha="right", va="center")
    ax_v.set_title(title, loc="left", color=c["text"], fontsize=10.5, pad=8)

    # Point stamps, one row per ring, labelled with their index in the array
    index = 0
    for row, (label, stamps) in enumerate(rings):
        y = len(rings) - 1 - row
        ax_p.plot(stamps, np.full(len(stamps), y), "o", color=c["text"], ms=5)
        for t in stamps:
            ax_p.annotate(str(index), (t, y), xytext=(0, 5), textcoords="offset points",
                          ha="center", va="bottom", color=c["text2"], fontsize=7.5)
            index += 1
        ax_p.text(-1.5, y, label, ha="right", va="center", color=c["text2"], fontsize=8)
    ax_p.set_ylim(-0.6, len(rings) - 0.2)
    ax_p.set_ylabel("points\n(array index)", rotation=0, ha="right", va="center", labelpad=34)

    # Accumulated pose
    t_fine = np.linspace(0.0, SAMPLE_STAMPS[-1], 400)
    x_table = np.array([exact_position(t) for t in t_fine])
    ax_x.plot(t_fine, x_table, color=c["table"], lw=2, label="This PR: motion table", zorder=3)
    ax_x.plot(points, [exact_position(t) for t in points], "o", color=c["table"], ms=5,
              mec=c["surface"], mew=1.5, zorder=4)
    styles = {"cpu": ("main, CPU corrector", main_cpu, "s"),
              "cuda": ("main, CUDA kernels", main_cuda, "D")}
    for key in series:
        label, simulate, marker = styles[key]
        poses = simulate(points)
        ax_x.plot(points, poses, "-", color=c[key], lw=1.2, alpha=0.55, zorder=2)
        ax_x.plot(points, poses, marker, color=c[key], ms=5.5, mec=c["surface"], mew=1.5,
                  label=label, zorder=5)
        for t, x in zip(points, poses):
            if abs(x - exact_position(t)) > 1.0:
                ax_x.plot([t, t], [x, exact_position(t)], color=c["text2"], lw=0.8, ls=":",
                          zorder=1)
    for text, xy, xytext in notes:
        ax_x.annotate(text, xy, xytext=xytext, textcoords="data", fontsize=8, color=c["text"],
                      ha="left", va="center",
                      arrowprops=dict(arrowstyle="-", color=c["text2"], lw=0.8))
    ax_x.set_ylabel("ego displacement\nsince header.stamp", rotation=0, ha="right", va="center")
    ax_x.set_xlabel("time since header.stamp [ms]")
    ax_x.set_xlim(-2, SAMPLE_STAMPS[-1] + 2)
    ax_x.set_xticks(np.arange(0, 111, 10))
    legend = ax_x.legend(loc=legend_loc, frameon=False, fontsize=8.5, handlelength=1.6)
    for text in legend.get_texts():
        text.set_color(c["text"])

    fig.savefig(path, facecolor=c["surface"], bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1]
    ring_a = np.array([0.0, 24.0, 48.0, 72.0, 96.0])
    ring_b = ring_a + 2.0
    newest_first = np.arange(96.0, -1.0, -12.0)
    figures = {
        "table": dict(
            rings=[("cloud", newest_first)],
            title="One cloud, stored newest point first",
            series=("cpu", "cuda"), legend_loc="lower right",
            notes=[(NOTE_CUDA, (30, 568), (14, 640)), (NOTE_CPU_BACK, (36, 88), (44, 20))]),
        "cpu": dict(
            rings=[("ring 0", ring_a), ("ring 1", ring_b)],
            title="One cloud, stored ring by ring",
            series=("cpu",),
            notes=[(NOTE_CPU_GAP, (72, 456), (78, 230)), (NOTE_CPU_JUMP, (40, 648), (12, 530))]),
        "cuda": dict(
            rings=[("cloud", newest_first)],
            title="One cloud, stored newest point first",
            series=("cuda",), legend_loc="lower right",
            notes=[(NOTE_CUDA, (30, 568), (14, 640))]),
    }
    for name, kwargs in figures.items():
        for theme in THEMES:
            for ext in ("svg", "png"):
                plot(f"{out}/{name}_{theme}.{ext}", theme, **kwargs)
