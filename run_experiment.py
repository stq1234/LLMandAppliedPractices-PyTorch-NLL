"""Reproducible CPU experiment for PyTorch negative log likelihood."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from html import escape

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SEED = 20261025
EPOCHS = 120
LEARNING_RATE = 0.2
NUM_CLASSES = 3
PAD_ID = 0


def manual_nll(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log probability of each integer class target."""
    return -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1).mean()


def loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, method: str) -> torch.Tensor:
    if method == "manual_gather":
        return manual_nll(F.log_softmax(logits, dim=1), targets)
    if method == "nll_loss":
        return F.nll_loss(F.log_softmax(logits, dim=1), targets)
    if method == "cross_entropy":
        return F.cross_entropy(logits, targets)
    raise ValueError(f"Unknown loss method: {method}")


def compare_fixed_logits() -> dict[str, object]:
    logits = torch.tensor(
        [[1.2, -0.1, 0.3], [-0.7, 1.1, 0.2], [0.0, 0.5, -0.2]],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 2, 1], dtype=torch.long)
    values: dict[str, float] = {}
    gradients: dict[str, torch.Tensor] = {}

    for method in ("manual_gather", "nll_loss", "cross_entropy"):
        input_logits = logits.clone().requires_grad_(True)
        loss = loss_from_logits(input_logits, targets, method)
        loss.backward()
        values[method] = float(loss.detach())
        gradients[method] = input_logits.grad.detach().clone()

    for method in ("nll_loss", "cross_entropy"):
        torch.testing.assert_close(
            torch.tensor(values["manual_gather"], dtype=torch.float64),
            torch.tensor(values[method], dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            gradients["manual_gather"], gradients[method], rtol=1e-12, atol=1e-12
        )

    return {
        "loss_by_method": values,
        "max_abs_loss_difference": max(values.values()) - min(values.values()),
        "max_abs_gradient_difference": max(
            float((gradients["manual_gather"] - gradients[method]).abs().max())
            for method in ("nll_loss", "cross_entropy")
        ),
    }


def check_sequence_padding() -> dict[str, float | int]:
    # Token 0 is padding. The first token in each full sequence is the context
    # start marker; targets are shifted by one position for next-token prediction.
    full_sequences = torch.tensor([[1, 3, 4, 2], [1, 4, 2, PAD_ID]], dtype=torch.long)
    input_ids = full_sequences[:, :-1]
    targets = full_sequences[:, 1:].clone()

    logits = torch.arange(2 * 3 * 5, dtype=torch.float64).reshape(2, 3, 5) / 9.0
    log_probs = F.log_softmax(logits, dim=-1)
    flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
    flat_targets = targets.reshape(-1)
    valid = flat_targets.ne(PAD_ID)

    picked = flat_log_probs.gather(
        dim=1, index=flat_targets.clamp_min(0).unsqueeze(1)
    ).squeeze(1)
    manual_mean = -picked.masked_select(valid).mean()
    nll_mean = F.nll_loss(flat_log_probs, flat_targets, ignore_index=PAD_ID)
    ce_mean = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), flat_targets, ignore_index=PAD_ID
    )
    torch.testing.assert_close(manual_mean, nll_mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(manual_mean, ce_mean, rtol=1e-12, atol=1e-12)

    return {
        "batch_size": int(input_ids.shape[0]),
        "sequence_length": int(input_ids.shape[1]),
        "valid_target_tokens": int(valid.sum()),
        "mean_nll_manual": float(manual_mean),
        "mean_nll_nll_loss": float(nll_mean),
        "mean_nll_cross_entropy": float(ce_mean),
        "max_abs_difference": max(
            float((manual_mean - nll_mean).abs()), float((manual_mean - ce_mean).abs())
        ),
    }


def check_numerical_stability() -> dict[str, int | bool]:
    extreme_logits = torch.tensor([[1000.0, 0.0, -1000.0]], dtype=torch.float32)
    naive_log_probs = torch.softmax(extreme_logits, dim=1).log()
    stable_log_probs = F.log_softmax(extreme_logits, dim=1)
    naive_nonfinite_count = int((~torch.isfinite(naive_log_probs)).sum())
    stable_is_finite = bool(torch.isfinite(stable_log_probs).all())
    assert naive_nonfinite_count > 0
    assert stable_is_finite
    return {
        "naive_softmax_then_log_nonfinite_values": naive_nonfinite_count,
        "log_softmax_all_finite": stable_is_finite,
    }


def make_synthetic_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    centers = torch.tensor([[-2.0, -0.5], [2.0, -0.5], [0.0, 2.4]], dtype=torch.float32)
    samples_per_class = 120
    train_per_class = 96
    standard_deviation = 0.8
    train_features: list[torch.Tensor] = []
    train_labels: list[torch.Tensor] = []
    test_features: list[torch.Tensor] = []
    test_labels: list[torch.Tensor] = []

    for class_id, center in enumerate(centers):
        samples = center + standard_deviation * torch.randn(
            samples_per_class, 2, generator=generator
        )
        order = torch.randperm(samples_per_class, generator=generator)
        train_index = order[:train_per_class]
        test_index = order[train_per_class:]
        train_features.append(samples[train_index])
        train_labels.append(torch.full((train_per_class,), class_id, dtype=torch.long))
        test_features.append(samples[test_index])
        test_labels.append(torch.full((samples_per_class - train_per_class,), class_id, dtype=torch.long))

    return (
        torch.cat(train_features),
        torch.cat(train_labels),
        torch.cat(test_features),
        torch.cat(test_labels),
    )


def save_synthetic_data(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> None:
    with (DATA_DIR / "synthetic_3class.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["split", "feature_1", "feature_2", "label"])
        for split, features, labels in (
            ("train", train_x, train_y),
            ("test", test_x, test_y),
        ):
            for row, label in zip(features.tolist(), labels.tolist()):
                writer.writerow([split, f"{row[0]:.8f}", f"{row[1]:.8f}", label])


def train_equivalent_objectives(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> tuple[dict[str, dict[str, float]], list[dict[str, float]], float, float]:
    torch.manual_seed(SEED + 1)
    initial_model = nn.Linear(2, NUM_CLASSES)
    initial_state = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
    methods = ("manual_gather", "nll_loss", "cross_entropy")
    histories: dict[str, list[float]] = {method: [] for method in methods}
    metrics: dict[str, dict[str, float]] = {}
    final_parameters: dict[str, list[torch.Tensor]] = {}

    with torch.no_grad():
        initial_loss = float(loss_from_logits(initial_model(train_x), train_y, "cross_entropy"))

    for method in methods:
        model = nn.Linear(2, NUM_CLASSES)
        model.load_state_dict(initial_state)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

        for _ in range(EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_from_logits(model(train_x), train_y, method)
            loss.backward()
            optimizer.step()
            histories[method].append(float(loss.detach()))

        with torch.no_grad():
            final_train_loss = float(loss_from_logits(model(train_x), train_y, method))
            test_logits = model(test_x)
            test_loss = float(F.cross_entropy(test_logits, test_y))
            test_accuracy = float((test_logits.argmax(dim=1) == test_y).float().mean())

        metrics[method] = {
            "initial_train_nll": initial_loss,
            "final_train_nll": final_train_loss,
            "final_test_nll": test_loss,
            "test_accuracy": test_accuracy,
        }
        final_parameters[method] = [parameter.detach().clone() for parameter in model.parameters()]

    history_differences = [
        abs(histories["manual_gather"][epoch] - histories[method][epoch])
        for method in ("nll_loss", "cross_entropy")
        for epoch in range(EPOCHS)
    ]
    max_history_difference = max(history_differences, default=0.0)
    max_parameter_difference = max(
        float((left - right).abs().max())
        for method in ("nll_loss", "cross_entropy")
        for left, right in zip(final_parameters["manual_gather"], final_parameters[method])
    )
    assert max_history_difference < 1e-6
    assert max_parameter_difference < 1e-6

    history_rows = [
        {"epoch": epoch + 1, **{method: histories[method][epoch] for method in methods}}
        for epoch in range(EPOCHS)
    ]
    return metrics, history_rows, max_history_difference, max_parameter_difference


def save_training_curve(history_rows: list[dict[str, float]]) -> None:
    """Write a dependency-free vector plot so it can be edited or re-rendered."""
    epochs = [int(row["epoch"]) for row in history_rows]
    styles = {
        "manual_gather": ("Manual gather", "#1f77b4", ""),
        "nll_loss": ("F.nll_loss", "#d95f02", "10 6"),
        "cross_entropy": ("F.cross_entropy", "#2ca02c", "2 5"),
    }
    width, height = 1000, 600
    left, right, top, bottom = 95, 35, 80, 85
    plot_width, plot_height = width - left - right, height - top - bottom
    all_values = [float(row[method]) for row in history_rows for method in styles]
    y_max = max(all_values) * 1.05
    x_max = max(epochs)

    def x_position(epoch: int) -> float:
        return left + (epoch - 1) * plot_width / max(1, x_max - 1)

    def y_position(value: float) -> float:
        return top + (y_max - value) * plot_height / y_max

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#333;stroke-width:1.5}</style>',
        f'<text x="{width/2}" y="35" text-anchor="middle" font-size="21">Equivalent PyTorch NLL objectives on synthetic data</text>',
    ]
    for tick in range(0, 6):
        value = y_max * tick / 5
        y = y_position(value)
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        elements.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-size="14">{value:.2f}</text>')
    for epoch in (1, 30, 60, 90, 120):
        x = x_position(epoch)
        elements.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        elements.append(f'<text x="{x:.2f}" y="{height-bottom+27}" text-anchor="middle" font-size="14">{epoch}</text>')
    elements.extend([
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
        f'<text x="{left + plot_width/2}" y="{height-20}" text-anchor="middle" font-size="16">Epoch</text>',
        f'<text x="25" y="{top + plot_height/2}" text-anchor="middle" font-size="16" transform="rotate(-90 25 {top + plot_height/2})">Mean training NLL</text>',
    ])
    for index, (method, (label, color, dash)) in enumerate(styles.items()):
        points = " ".join(
            f'{x_position(int(row["epoch"])):.2f},{y_position(float(row[method])):.2f}'
            for row in history_rows
        )
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"{dash_attribute}/>'
        )
        legend_x = left + 30 + index * 205
        elements.append(f'<line x1="{legend_x}" y1="58" x2="{legend_x+35}" y2="58" stroke="{color}" stroke-width="3"{dash_attribute}/>' )
        elements.append(f'<text x="{legend_x+43}" y="63" font-size="14">{escape(label)}</text>')
    elements.append("</svg>")
    (FIGURES_DIR / "training_loss.svg").write_text("\n".join(elements) + "\n", encoding="utf-8")

    # Raster copy for platforms that require a PNG image in the report editor.
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 21)
        label_font = ImageFont.truetype("arial.ttf", 15)
        tick_font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        title_font = label_font = tick_font = ImageFont.load_default()
    draw.text((width / 2, 18), "Equivalent PyTorch NLL objectives on synthetic data", fill="#222222", font=title_font, anchor="mt")
    for tick in range(0, 6):
        value = y_max * tick / 5
        y = y_position(value)
        draw.line((left, y, width - right, y), fill="#dddddd", width=1)
        draw.text((left - 12, y), f"{value:.2f}", fill="#222222", font=tick_font, anchor="rm")
    for epoch in (1, 30, 60, 90, 120):
        x = x_position(epoch)
        draw.line((x, top, x, height - bottom), fill="#eeeeee", width=1)
        draw.text((x, height - bottom + 7), str(epoch), fill="#222222", font=tick_font, anchor="mt")
    draw.line((left, top, left, height - bottom), fill="#333333", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#333333", width=2)
    draw.text((left + plot_width / 2, height - 35), "Epoch", fill="#222222", font=label_font, anchor="mt")
    # Vertical labels are drawn on a separate layer and rotated for readability.
    ylabel = Image.new("RGBA", (260, 32), (255, 255, 255, 0))
    ImageDraw.Draw(ylabel).text((130, 0), "Mean training NLL", fill="#222222", font=label_font, anchor="mt")
    ylabel = ylabel.rotate(90, expand=True)
    image.paste(ylabel, (3, int(top + plot_height / 2 - ylabel.height / 2)), ylabel)
    draw = ImageDraw.Draw(image)
    import math

    dash_patterns = {"": None, "10 6": (10, 6), "2 5": (2, 5)}
    marker_shapes = {"manual_gather": "circle", "nll_loss": "square", "cross_entropy": "triangle"}

    def draw_styled_line(
        points: list[tuple[float, float]],
        color: str,
        pattern: tuple[int, ...] | None,
        line_width: int,
    ) -> None:
        if pattern is None:
            draw.line(points, fill=color, width=line_width, joint="curve")
            return
        pattern_index = 0
        draw_phase = True
        remaining = pattern[pattern_index]
        for start, end in zip(points, points[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            segment_length = math.hypot(dx, dy)
            if segment_length == 0:
                continue
            ux, uy = dx / segment_length, dy / segment_length
            offset = 0.0
            while offset < segment_length:
                step = min(remaining, segment_length - offset)
                if draw_phase:
                    draw.line(
                        (
                            start[0] + ux * offset,
                            start[1] + uy * offset,
                            start[0] + ux * (offset + step),
                            start[1] + uy * (offset + step),
                        ),
                        fill=color,
                        width=line_width,
                    )
                offset += step
                remaining -= step
                if remaining <= 1e-9:
                    pattern_index = (pattern_index + 1) % len(pattern)
                    draw_phase = pattern_index % 2 == 0
                    remaining = pattern[pattern_index]

    def draw_marker(point: tuple[float, float], color: str, shape: str) -> None:
        x, y = point
        radius = 5
        if shape == "circle":
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        elif shape == "square":
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        else:
            draw.polygon(
                ((x, y - radius - 1), (x - radius - 1, y + radius), (x + radius + 1, y + radius)),
                outline=color,
                width=2,
            )

    marker_epochs = {1, 30, 60, 90, 120}
    for index, (method, (label, color, dash)) in enumerate(styles.items()):
        points = [
            (x_position(int(row["epoch"])), y_position(float(row[method])))
            for row in history_rows
        ]
        marker = marker_shapes[method]
        pattern = dash_patterns[dash]
        draw_styled_line(points, color, pattern, line_width=3)
        for row, point in zip(history_rows, points):
            if int(row["epoch"]) in marker_epochs:
                draw_marker(point, color, marker)
        legend_x = left + 30 + index * 205
        draw_styled_line([(legend_x, 58), (legend_x + 35, 58)], color, pattern, line_width=3)
        draw_marker((legend_x + 18, 58), color, marker)
        draw.text((legend_x + 43, 48), label, fill="#222222", font=label_font)
    image.save(FIGURES_DIR / "training_loss.png", optimize=True)


def main() -> None:
    for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    fixed = compare_fixed_logits()
    sequence = check_sequence_padding()
    stability = check_numerical_stability()
    train_x, train_y, test_x, test_y = make_synthetic_data()
    save_synthetic_data(train_x, train_y, test_x, test_y)
    metrics, history_rows, max_history_difference, max_parameter_difference = (
        train_equivalent_objectives(train_x, train_y, test_x, test_y)
    )
    save_training_curve(history_rows)

    with (RESULTS_DIR / "training_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "manual_gather", "nll_loss", "cross_entropy"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(history_rows)

    summary = {
        "environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "device": "cpu",
            "dtype": "float32 for training; float64 for equivalence checks",
            "seed": SEED,
        },
        "fixed_logits_equivalence": fixed,
        "sequence_padding_check": sequence,
        "numerical_stability_check": stability,
        "synthetic_classification": {
            "classes": NUM_CLASSES,
            "features": 2,
            "train_examples": int(train_y.numel()),
            "test_examples": int(test_y.numel()),
            "samples_per_class": 120,
            "train_fraction_per_class": 0.8,
            "model": "torch.nn.Linear(2, 3)",
            "optimizer": "torch.optim.SGD",
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batching": "full batch",
            "loss_by_method": metrics,
            "max_abs_training_curve_difference": max_history_difference,
            "max_abs_final_parameter_difference": max_parameter_difference,
        },
    }
    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
