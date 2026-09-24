"""

E. Wes Bethel, Copyright (C) 2022
Adapted for CSC 746 HW2 -- matrix multiply study (basic / blocked+copy / CBLAS).

Description: loads .csv files of results, derives MFLOP/s where needed, and
             creates N-variable charts, LaTeX tables, and a printed analysis
             summary for the report's discussion sections.

Usage:   python plot_mmul_savefig.py                 # builds all three charts
         python plot_mmul_savefig.py basic_vs_blas   # builds just one
         (valid names: basic_vs_blas, blocked_vs_blas, basic_vs_blocked)

Inputs:  basic_vs_blas.csv, blocked_vs_blas.csv, basic_vs_blocked.csv
Outputs: <name>.png    300 dpi chart
         <name>.tex    LaTeX tabular of the same numbers
         plus a printed summary (% of peak, ratio vs CBLAS) on stdout

CSV format: first column = problem size N. Every remaining column is one
            series. Column headers become legend labels. Any number of
            series is supported.

            Values may be EITHER MFLOP/s (default) or elapsed seconds --
            set "units" to "seconds" in the chart's config and the script
            derives MFLOP/s itself as 2*N^3 / t / 1e6.

Dependencies: matplotlib, pandas

"""

import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # write files without needing a display
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ---------------------------------------------------------------- platform

# EDIT THIS with your own Perlmutter details before submitting.
PLATFORM = ("Perlmutter CPU node: AMD EPYC 7763 (Milan), 1 core, "
            "GCC 14.3.0, -O3 -march=native, median of 3 runs")

# Theoretical per-core peak, MFLOP/s. Drawn as a dashed reference line and
# used for the "% of peak" column in the summary.
#   2.45 GHz base clock x 16 dp FLOPS/clock = 39.2 GFLOP/s, where
#   16 = (AVX2 256-bit = 4 doubles) x (FMA = 2 FLOPS) x (2 FMA units)
# Set PEAK_MFLOPS to None to omit the line and the % column.
PEAK_MFLOPS = 39200.0
PEAK_LABEL = "AMD EPYC 7763 peak = 39.2 GFLOP/s"

# Cache sizes in KiB, for the annotation printed with the summary.
# Perlmutter CPU node (AMD EPYC 7763), per Lecture 09 slide 7.
CACHES = [("L1", 32), ("L2", 512), ("L3 (per CCD)", 32 * 1024)]

# Show the x-axis footprint as A+B+C (True, matches the lecture's table)
# or as a single N x N matrix (False).
FOOTPRINT_ABC = True

# ---------------------------------------------------------------- charts

CHARTS = {
    "basic_vs_blas": {
        "fname":      "basic_vs_blas.csv",
        "plot_fname": "basic_vs_blas.png",
        "tex_fname":  "basic_vs_blas.tex",
        "title":      "MMUL CBLAS vs Basic MFLOP/s, AMD Milan 7763",
        "caption":    "Basic three-loop MM and CBLAS, MFLOP/s vs problem size.",
        "units":      "mflops",      # "mflops" or "seconds"
        "logy":       True,
        "yticks":     None,          # None = chosen automatically
        "legend_title": "Implementation",
    },
    "blocked_vs_blas": {
        "fname":      "blocked_vs_blas.csv",
        "plot_fname": "blocked_vs_blas.png",
        "tex_fname":  "blocked_vs_blas.tex",
        "title":      "MMUL CBLAS vs Blocked MFLOP/s, AMD Milan 7763",
        "caption":    ("Blocked MM with copy optimization at four block sizes, "
                       "and CBLAS, MFLOP/s vs problem size."),
        "units":      "mflops",
        "logy":       True,
        "yticks":     None,
        "legend_title": "Implementation",
    },
    "basic_vs_blocked": {
        "fname":      "basic_vs_blocked.csv",
        "plot_fname": "basic_vs_blocked.png",
        "tex_fname":  "basic_vs_blocked.tex",
        "title":      "Basic vs Blocked MFLOP/s, AMD Milan 7763",
        "caption":    ("Basic three-loop MM and blocked MM with copy optimization "
                       "at four block sizes, MFLOP/s vs problem size."),
        "units":      "mflops",
        "logy":       True,
        "yticks":     None,
        "legend_title": "Implementation",
    },
}

# enough distinct styles for 5+ series
STYLES = ["r-o", "b-x", "g-^", "m-s", "c-v", "y-D", "k-*"]

# candidate log-scale ticks; filtered to the data range at plot time
TICK_CANDIDATES = [50, 100, 200, 500, 1000, 2000, 5000,
                   10000, 20000, 50000, 100000, 200000]


# ---------------------------------------------------------------- helpers

def flops(n):
    """FLOPs performed by an N x N matrix multiply: N^2 outputs, each needing
    N multiplies and N adds -> 2*N^3. Float arithmetic throughout: 2*n*n*n in
    32-bit integer arithmetic overflows to 0 at N=2048."""
    return 2.0 * n * n * n


def to_mflops(n, seconds):
    return flops(n) / seconds / 1.0e6


def footprint_kib(n):
    """Memory footprint in KiB. A+B+C by default, else one matrix."""
    matrices = 3 if FOOTPRINT_ABC else 1
    return matrices * n * n * 8 / 1024.0


def size_label(n):
    """'N' over its memory footprint, e.g. '2048\n(96 MB)'."""
    kib = footprint_kib(n)
    if kib >= 1024 * 1024:
        foot = "%g GB" % (kib / 1024 / 1024)
    elif kib >= 1024:
        foot = "%g MB" % (kib / 1024)
    else:
        foot = "%g KB" % kib
    return "%d\n(%s)" % (n, foot)


def pick_ticks(lo, hi):
    """Keep the candidate ticks that bracket the data."""
    return [t for t in TICK_CANDIDATES if t >= lo / 1.6 and t <= hi * 1.6]


def load_mflops(cfg):
    """Read the CSV and return (problem_sizes, {series_name: [mflops...]}).
    Converts from elapsed seconds if the config says the file holds seconds."""
    df = pd.read_csv(cfg["fname"], comment="#")
    print("--- " + cfg["fname"])
    print(df)

    var_names = list(df.columns)
    problem_sizes = df[var_names[0]].values.tolist()

    series = {}
    for name in var_names[1:]:
        values = df[name].values.tolist()
        if cfg.get("units", "mflops") == "seconds":
            values = [to_mflops(n, t) for n, t in zip(problem_sizes, values)]
        series[name] = values

    if cfg.get("units", "mflops") == "seconds":
        print("(converted elapsed seconds -> MFLOP/s using 2*N^3 / t / 1e6)")

    return problem_sizes, series


# ---------------------------------------------------------------- outputs

def write_latex_table(cfg, problem_sizes, series):
    """Emit a LaTeX tabular of the same numbers the chart shows."""
    names = list(series.keys())

    lines = []
    lines.append("% generated by plot_mmul_savefig.py")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{r" + "r" * len(names) + "}")
    lines.append("\\hline")
    safe = [n.replace("_", "\\_") for n in names]
    lines.append("Problem size $N$ & " + " & ".join(safe) + " \\\\")
    lines.append("\\hline")
    for i, n in enumerate(problem_sizes):
        row = ["%d" % n] + ["%.1f" % series[name][i] for name in names]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{%s All values in MFLOP/s.}" % cfg["caption"])
    lines.append("\\label{tab:%s}" % cfg["plot_fname"].replace(".png", ""))
    lines.append("\\end{table}")

    with open(cfg["tex_fname"], "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote " + cfg["tex_fname"])


def print_summary(problem_sizes, series):
    """Numbers the discussion sections actually ask for: range across problem
    sizes, % of theoretical peak, and ratio against CBLAS."""
    names = list(series.keys())

    print("")
    print("  %-18s %10s %10s %8s  %s" %
          ("series", "min", "max", "max/min", "range of problem sizes"))
    for name in names:
        v = series[name]
        lo, hi = min(v), max(v)
        print("  %-18s %10.1f %10.1f %8.1fx  N=%d..%d" %
              (name, lo, hi, hi / lo if lo else float("nan"),
               problem_sizes[0], problem_sizes[-1]))

    if PEAK_MFLOPS:
        print("")
        print("  %% of theoretical peak (%.1f GFLOP/s):" % (PEAK_MFLOPS / 1000))
        hdr = "  %-18s" % "series" + "".join("%9d" % n for n in problem_sizes)
        print(hdr)
        for name in names:
            row = "  %-18s" % name
            row += "".join("%8.2f%%" % (100.0 * v / PEAK_MFLOPS)
                           for v in series[name])
            print(row)

    ref = next((n for n in names if "cblas" in n.lower()), None)
    if ref:
        print("")
        print("  speedup of %s over each series (x times faster):" % ref)
        hdr = "  %-18s" % "series" + "".join("%9d" % n for n in problem_sizes)
        print(hdr)
        for name in names:
            if name == ref:
                continue
            row = "  %-18s" % name
            row += "".join("%9.1f" % (r / v if v else float("nan"))
                           for r, v in zip(series[ref], series[name]))
            print(row)

    print("")
    print("  memory footprint of A+B+C vs cache (Lecture 09, slide 7):")
    for n in problem_sizes:
        kib = 3 * n * n * 8 / 1024.0
        fits = [lbl for lbl, size in CACHES if kib <= size]
        where = ("fits " + fits[0]) if fits else "exceeds L3"
        print("    N=%-5d  A+B+C = %9.1f KiB   %s" % (n, kib, where))


def make_chart(key):
    cfg = CHARTS[key]
    problem_sizes, series = load_mflops(cfg)
    series_names = list(series.keys())

    if len(series_names) > len(STYLES):
        print("warning: %d series but only %d styles defined" %
              (len(series_names), len(STYLES)))

    plt.figure(figsize=(11.5, 6.5))
    plt.title(cfg["title"], fontsize=13)

    xlocs = [i for i in range(len(problem_sizes))]
    plt.xticks(xlocs, [size_label(n) for n in problem_sizes], fontsize=9)

    lo, hi = float("inf"), 0.0
    for i, name in enumerate(series_names):
        values = series[name]
        plt.plot(values, STYLES[i % len(STYLES)])
        lo = min(lo, min(values))
        hi = max(hi, max(values))

    legend_labels = list(series_names)
    if PEAK_MFLOPS:
        plt.axhline(y=PEAK_MFLOPS, color="k", linestyle="--", linewidth=1.8)
        legend_labels.append(PEAK_LABEL)
        hi = max(hi, PEAK_MFLOPS)

    if cfg["logy"]:
        plt.yscale("log")
        ax = plt.gca()
        ticks = cfg["yticks"] if cfg["yticks"] else pick_ticks(lo, hi)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()

    what = "A+B+C" if FOOTPRINT_ABC else "one N×N matrix"
    plt.xlabel("Problem Size (N; %s memory footprint in parentheses)" % what)
    plt.ylabel("Rate (MFLOP/s%s)" % (", log scale" if cfg["logy"] else ""))

    plt.legend(legend_labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
               borderaxespad=0.0, fontsize=10, title=cfg["legend_title"],
               title_fontsize=10)

    plt.grid(axis='both')
    plt.figtext(0.42, 0.045, PLATFORM, ha="center", fontsize=8, style="italic")
    plt.subplots_adjust(left=0.08, right=0.76, top=0.87, bottom=0.17)

    plt.savefig(cfg["plot_fname"], dpi=300, bbox_inches="tight")
    print("wrote " + cfg["plot_fname"])

    write_latex_table(cfg, problem_sizes, series)
    print_summary(problem_sizes, series)
    print("")


if len(sys.argv) > 1:
    requested = sys.argv[1:]
else:
    requested = list(CHARTS.keys())

for key in requested:
    if key not in CHARTS:
        print("unknown chart '%s'; valid names: %s" % (key, ", ".join(CHARTS)))
        sys.exit(1)
    make_chart(key)

# EOF