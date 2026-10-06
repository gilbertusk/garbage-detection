"""Split per kelas dengan grup duplikat dijaga utuh di satu split."""

import random
from collections import Counter, defaultdict
from collections.abc import Hashable, Mapping, Sequence

RATIO_TOLERANCE = 1e-6


def _primary_labels(labels: Sequence[str], groups: Sequence[Hashable]) -> dict:
    """Kelas mayoritas tiap grup (seri diputus menurut urutan abjad)."""
    label_counts: dict = defaultdict(Counter)
    for label, group in zip(labels, groups):
        label_counts[group][label] += 1
    return {
        group: min(counts, key=lambda label: (-counts[label], label))
        for group, counts in label_counts.items()
    }


def assign_splits(
    labels: Sequence[str],
    groups: Sequence[Hashable],
    ratios: Mapping[str, float],
    seed: int,
) -> list[str]:
    """Tentukan split tiap gambar.

    Tiap grup masuk utuh ke satu split. Di dalam tiap kelas, grup diisi secara
    greedy ke split yang paling jauh di bawah targetnya.
    """
    if abs(sum(ratios.values()) - 1.0) > RATIO_TOLERANCE:
        raise ValueError(f"Rasio split harus berjumlah 1, dapat {sum(ratios.values())}")

    group_sizes = Counter(groups)
    groups_by_label = defaultdict(list)
    for group, label in _primary_labels(labels, groups).items():
        groups_by_label[label].append(group)

    rng = random.Random(seed)
    split_of_group = {}
    for label in sorted(groups_by_label):
        label_groups = sorted(groups_by_label[label], key=str)
        rng.shuffle(label_groups)
        label_groups.sort(key=lambda group: -group_sizes[group])  # stabil: besar dulu

        total = sum(group_sizes[group] for group in label_groups)
        filled = dict.fromkeys(ratios, 0)
        for group in label_groups:
            target = max(ratios, key=lambda name: ratios[name] * total - filled[name])
            split_of_group[group] = target
            filled[target] += group_sizes[group]

    return [split_of_group[group] for group in groups]
