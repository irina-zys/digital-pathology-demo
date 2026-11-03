"""Read measured Lightning CSV epochs and plot learning curves without smoothing."""

import csv
import math
from pathlib import Path


FIELDS = ('train/loss', 'val/loss', 'train/accuracy', 'val/accuracy')


def read_epoch_history(path):
    """Join separate train/validation logger rows by epoch; omit test-only rows.

    Returned epochs are one-based. Missing train/validation loss, nonfinite values,
    repeated metric values within an epoch and gaps are rejected, not filled or
    interpolated. Accuracy is optional for nonclassification workflows.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 10_000_000:
        raise ValueError('Expected an existing epoch CSV no larger than 10 MB')
    epochs = {}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'epoch', 'train/loss', 'val/loss'} <= set(reader.fieldnames or []):
            raise ValueError('CSV must contain epoch and training/validation losses')
        for row in reader:
            values = {name: float(row[name]) for name in FIELDS if row.get(name, '') != ''}
            if not values:
                continue
            epoch = float(row['epoch'])
            if not math.isfinite(epoch) or epoch < 0 or not epoch.is_integer():
                raise ValueError('Epoch must be a nonnegative integer')
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError('Nonfinite epoch metric')
            target = epochs.setdefault(int(epoch) + 1, {})
            if set(target) & set(values):
                raise ValueError('Repeated metric within an epoch; use one isolated training run')
            target.update(values)
    if not epochs:
        raise ValueError('No measured training epochs found')
    ordered = sorted(epochs)
    if ordered != list(range(ordered[0], ordered[-1] + 1)):
        raise ValueError('Missing epoch; learning curves must not conceal gaps')
    if any(not {'train/loss', 'val/loss'} <= set(row) for row in epochs.values()):
        raise ValueError('Every plotted epoch needs measured training and validation loss')
    return [{'epoch': epoch, **epochs[epoch]} for epoch in ordered]


def plot_learning_curves(history, *, selected_epoch=None):
    """Plot recorded losses and optional accuracies; leave checkpoint selection to callers."""
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError('No measured epochs to plot')
    epochs = [row['epoch'] for row in history]
    if selected_epoch is not None and selected_epoch not in epochs:
        raise ValueError('Selected checkpoint epoch is absent from the history')
    accuracy = all({'train/accuracy', 'val/accuracy'} <= set(row) for row in history)
    fig, axes = plt.subplots(1, 2 if accuracy else 1, figsize=(11 if accuracy else 6, 4),
                             layout='constrained', squeeze=False)
    panels = [('loss', 'Loss')]
    if accuracy:
        panels.append(('accuracy', 'Accuracy'))
    for ax, (metric, ylabel) in zip(axes.flat, panels):
        for stage, label, color in [('train', 'Training', '#0072B2'), ('val', 'Validation', '#D55E00')]:
            ax.plot(epochs, [row[f'{stage}/{metric}'] for row in history], label=label,
                    color=color, marker='o', markersize=3)
        if selected_epoch is not None:
            ax.axvline(selected_epoch, color='0.4', linestyle='--', label='Selected checkpoint')
        ax.set(xlabel='Completed epoch', ylabel=ylabel)
        if metric == 'accuracy':
            ax.set_ylim(-.03, 1.03)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Measured learning curves')
    return fig
