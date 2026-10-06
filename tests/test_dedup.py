from PIL import Image

from wastesorter.dedup import correlation, group_near_duplicates, thumbnail_signature


def _signature(path):
    with Image.open(path) as image:
        return thumbnail_signature(image)


def _resized_copy(source, target, size=(120, 120)):
    with Image.open(source) as image:
        image.resize(size).save(target, quality=60)
    return target


def test_resized_and_recompressed_copy_correlates_strongly(tmp_path, make_image):
    original = make_image(tmp_path / "a.jpg", seed=1, size=(200, 200))
    copy = _resized_copy(original, tmp_path / "small.jpg")

    assert correlation(_signature(original), _signature(copy)) > 0.97


def test_unrelated_images_correlate_weakly(tmp_path, make_image):
    first = make_image(tmp_path / "a.jpg", seed=1)
    second = make_image(tmp_path / "b.jpg", seed=2)

    assert abs(correlation(_signature(first), _signature(second))) < 0.6


def test_grouping_joins_copies_and_separates_unrelated_images(tmp_path, make_image):
    original = make_image(tmp_path / "a.jpg", seed=1, size=(200, 200))
    copy = _resized_copy(original, tmp_path / "small.jpg")
    other = make_image(tmp_path / "b.jpg", seed=2)

    groups = group_near_duplicates(
        [_signature(original), _signature(other), _signature(copy)], min_correlation=0.97
    )

    assert groups[0] == groups[2]
    assert groups[1] != groups[0]


def test_grouping_works_across_chunk_boundaries(tmp_path, make_image):
    same = _signature(make_image(tmp_path / "a.jpg", seed=1))
    others = [_signature(make_image(tmp_path / f"{seed}.jpg", seed=seed)) for seed in (2, 3)]

    groups = group_near_duplicates(
        [same, others[0], same, others[1], same], min_correlation=0.97, chunk_size=2
    )

    assert groups[0] == groups[2] == groups[4]
    assert len(set(groups.tolist())) == 3


def test_flat_image_is_never_grouped():
    flat = bytes([128]) * (32 * 32)

    groups = group_near_duplicates([flat, flat], min_correlation=0.97)

    assert groups[0] != groups[1]
