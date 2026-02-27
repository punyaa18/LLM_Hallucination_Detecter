"""Generate feature architecture diagram for the research paper."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, xy, w, h, text, color="#e8eef8"):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02",
        linewidth=1.4,
        edgecolor="#2b4c7e",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=12, linewidth=1.2, color="#1f2d3d"))


def main():
    base = Path(__file__).resolve().parent
    figures = base / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.03, 0.62), 0.16, 0.2, "Prompt Set\n(8 categories)", "#dceaf9")
    box(ax, (0.24, 0.62), 0.2, 0.2, "Ollama Models\nllama3 / llama3.2 / custom", "#e8f5e9")
    box(ax, (0.49, 0.62), 0.2, 0.2, "Tri-LLM Evaluator\nconsistency + divergence", "#fff8e1")
    box(ax, (0.74, 0.62), 0.22, 0.2, "Scoring Engine\nrisk + overconfidence", "#fdecea")

    box(ax, (0.16, 0.2), 0.24, 0.2, "nomic-embed-text\nretrieval diagnostics", "#ede7f6")
    box(ax, (0.45, 0.2), 0.23, 0.2, "Research Assets\nCSV + JSON + tests", "#e0f7fa")
    box(ax, (0.73, 0.2), 0.23, 0.2, "Paper Outputs\nIEEE TEX + PDF + DOCX", "#fce4ec")

    arrow(ax, (0.19, 0.72), (0.24, 0.72))
    arrow(ax, (0.44, 0.72), (0.49, 0.72))
    arrow(ax, (0.69, 0.72), (0.74, 0.72))

    arrow(ax, (0.34, 0.62), (0.28, 0.40))
    arrow(ax, (0.59, 0.62), (0.56, 0.40))
    arrow(ax, (0.85, 0.62), (0.84, 0.40))

    arrow(ax, (0.40, 0.30), (0.45, 0.30))
    arrow(ax, (0.68, 0.30), (0.73, 0.30))

    ax.text(0.5, 0.94, "Feature Architecture for Local Hallucination Benchmark", ha="center", va="center", fontsize=12, fontweight="bold")

    out = figures / "feature_architecture.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Created {out}")


if __name__ == "__main__":
    main()
