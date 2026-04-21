2"""Generate a minimal methodology feature module diagram for the paper."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


TITLE_COLOR = "#18324d"
TEXT_COLOR = "#24384d"
LINE_COLOR = "#254f7d"
BOX_FILL = "#f7f9fc"
def add_box(ax, x, y, w, h, title, detail):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            linewidth=1.6,
            edgecolor=LINE_COLOR,
            facecolor=BOX_FILL,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    ax.text(
        x + w / 2,
        y + h * 0.33,
        detail,
        ha="center",
        va="center",
        fontsize=8.5,
        color=TEXT_COLOR,
    )


def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=13,
            linewidth=1.4,
            color=LINE_COLOR,
        )
    )
def main():
    base = Path(__file__).resolve().parent
    figures = base / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")

    ax.text(
        0.5,
        0.95,
        "Methodology Feature Module Diagram",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=TITLE_COLOR,
    )
    ax.text(
        0.5,
        0.915,
        "Minimal pipeline for hallucination detection in LLM outputs",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#4a647f",
    )

    box_y = 0.34
    box_h = 0.22
    box_w = 0.12
    gap = 0.016
    start_x = 0.04

    boxes = [
        ("Input\nLayer", "Query / text"),
        ("Claim\nExtraction", "Atomic claims"),
        ("Evidence\nRetrieval", "Trusted sources"),
        ("Verification", "NLI / similarity"),
        ("Hallucination\nDetection", "Supported / contradicted / unverif."),
        ("Risk\nScoring", "Confidence score"),
        ("Annotated\nResponse", "Flags + reliability"),
    ]

    positions = []
    for index, (title, detail) in enumerate(boxes):
        x = start_x + index * (box_w + gap)
        positions.append(x)
        add_box(ax, x, box_y, box_w, box_h, title, detail)

    mid_y = box_y + box_h / 2
    for x in positions[:-1]:
        add_arrow(ax, (x + box_w, mid_y), (x + box_w + gap, mid_y))

    add_arrow(ax, (0.13, 0.68), (0.13, box_y + box_h))
    add_arrow(ax, (0.88, box_y), (0.88, 0.29))

    out = figures / "feature_architecture.png"
    plt.tight_layout()
    plt.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"Created {out}")


if __name__ == "__main__":
    main()
