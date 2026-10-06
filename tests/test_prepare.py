from wastesorter.data import scan_images
from wastesorter.manifest import read_manifest, write_manifest
from wastesorter.prepare import build_manifest, drop_exact_duplicates


def test_identical_files_in_one_class_keep_a_single_copy(tmp_path, make_image):
    source = make_image(tmp_path / "glass" / "a.jpg", seed=1)
    (tmp_path / "glass" / "b.jpg").write_bytes(source.read_bytes())
    records, _ = scan_images(tmp_path)

    result = drop_exact_duplicates(records)

    assert [r.path for r in result.kept] == ["glass/a.jpg"]
    assert result.dropped_copies == ["glass/b.jpg"]
    assert result.label_conflicts == []


def test_identical_files_in_different_classes_are_all_dropped(tmp_path, make_image):
    source = make_image(tmp_path / "glass" / "a.jpg", seed=1)
    (tmp_path / "metal").mkdir()
    (tmp_path / "metal" / "a.jpg").write_bytes(source.read_bytes())
    make_image(tmp_path / "metal" / "other.jpg", seed=2)
    records, _ = scan_images(tmp_path)

    result = drop_exact_duplicates(records)

    assert [r.path for r in result.kept] == ["metal/other.jpg"]
    assert result.label_conflicts == ["glass/a.jpg", "metal/a.jpg"]


def test_manifest_keeps_near_duplicates_in_the_same_split(tmp_path, make_image):
    from PIL import Image

    for seed in range(20):
        path = make_image(tmp_path / "glass" / f"{seed}.jpg", seed=seed, size=(200, 200))
        Image.open(path).resize((120, 120)).save(tmp_path / "glass" / f"{seed}_small.jpg", quality=60)
    records, _ = scan_images(tmp_path)

    rows = build_manifest(records)

    split_of = {row["path"]: row["split"] for row in rows}
    assert all(split_of[f"glass/{s}.jpg"] == split_of[f"glass/{s}_small.jpg"] for s in range(20))
    assert {row["split"] for row in rows} == {"train", "val", "test"}


def test_manifest_round_trips_through_csv(tmp_path, make_image):
    make_image(tmp_path / "glass" / "a.jpg", seed=1)
    records, _ = scan_images(tmp_path)
    rows = build_manifest(records)

    write_manifest(rows, tmp_path / "out" / "split.csv")

    loaded = read_manifest(tmp_path / "out" / "split.csv")
    assert loaded[0]["path"] == "glass/a.jpg"
    assert loaded[0]["label"] == "glass"
    assert loaded[0]["group"] == "g00000"
