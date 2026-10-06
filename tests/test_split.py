from collections import Counter, defaultdict

import pytest

from wastesorter.split import assign_splits

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _dataset(per_class=200, group_size=1):
    labels, groups = [], []
    for label in ("glass", "metal", "trash"):
        for index in range(per_class):
            labels.append(label)
            groups.append(f"{label}-{index // group_size}")
    return labels, groups


def test_every_group_lands_in_exactly_one_split():
    labels, groups = _dataset(group_size=3)

    splits = assign_splits(labels, groups, RATIOS, seed=0)

    splits_per_group = defaultdict(set)
    for group, split in zip(groups, splits):
        splits_per_group[group].add(split)
    assert all(len(found) == 1 for found in splits_per_group.values())


def test_each_class_follows_the_requested_ratios():
    labels, groups = _dataset(per_class=200)

    splits = assign_splits(labels, groups, RATIOS, seed=0)

    for label in ("glass", "metal", "trash"):
        counts = Counter(split for split, found in zip(splits, labels) if found == label)
        assert counts["train"] == pytest.approx(140, abs=2)
        assert counts["val"] == pytest.approx(30, abs=2)
        assert counts["test"] == pytest.approx(30, abs=2)


def test_same_seed_gives_same_split():
    labels, groups = _dataset()

    assert assign_splits(labels, groups, RATIOS, seed=7) == assign_splits(
        labels, groups, RATIOS, seed=7
    )


def test_group_spanning_two_classes_stays_together():
    labels = ["glass"] * 50 + ["metal"] * 50
    groups = [f"g{i}" for i in range(50)] + [f"m{i}" for i in range(50)]
    groups[0] = groups[50] = "shared"

    splits = assign_splits(labels, groups, RATIOS, seed=0)

    assert splits[0] == splits[50]


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        assign_splits(["glass"], ["g0"], {"train": 0.5, "val": 0.1}, seed=0)
