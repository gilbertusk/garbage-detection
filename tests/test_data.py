from wastesorter.data import scan_images


def test_scan_reads_class_from_parent_folder(tmp_path, make_image):
    make_image(tmp_path / "glass" / "a.jpg", seed=1)
    make_image(tmp_path / "metal" / "b.jpg", seed=2)

    records, invalid = scan_images(tmp_path)

    assert invalid == []
    assert [(r.path, r.label) for r in records] == [
        ("glass/a.jpg", "glass"),
        ("metal/b.jpg", "metal"),
    ]


def test_scan_records_image_size(tmp_path, make_image):
    make_image(tmp_path / "glass" / "a.jpg", seed=1, size=(120, 80))

    records, _ = scan_images(tmp_path)

    assert (records[0].width, records[0].height) == (120, 80)


def test_scan_reports_corrupt_file_as_invalid(tmp_path, make_image):
    make_image(tmp_path / "glass" / "ok.jpg", seed=1)
    (tmp_path / "glass" / "broken.jpg").write_bytes(b"bukan gambar")

    records, invalid = scan_images(tmp_path)

    assert [r.path for r in records] == ["glass/ok.jpg"]
    assert invalid == ["glass/broken.jpg"]


def test_scan_gives_identical_files_the_same_sha1(tmp_path, make_image):
    source = make_image(tmp_path / "glass" / "a.jpg", seed=1)
    (tmp_path / "glass" / "copy.jpg").write_bytes(source.read_bytes())

    records, _ = scan_images(tmp_path)

    assert records[0].sha1 == records[1].sha1
