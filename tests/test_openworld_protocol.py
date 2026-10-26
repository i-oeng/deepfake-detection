from deepfake_detection.data import openworld
from deepfake_detection.data.openworld import (
    Ownership,
    _external_rows,
    df40_member,
    method_roles,
    source_entities,
)


def test_every_generator_has_exactly_one_role():
    roles = method_roles()
    assert set(roles) == {"train", "validation", "test"}
    names = [method for methods in roles.values() for method in methods]
    assert len(names) == len(set(names))
    assert len(roles["train"]) == len(openworld.TRAIN_METHODS)


def test_round2_extends_only_training_and_validation_roles():
    v1 = method_roles()
    round2 = method_roles(openworld.PROTOCOL_ROUND2)
    assert round2["train"] - v1["train"] == openworld.ROUND2_TRAIN_METHODS
    assert round2["validation"] - v1["validation"] == openworld.ROUND2_VALIDATION_METHODS
    assert round2["test"] == v1["test"]
    names = [method for methods in round2.values() for method in methods]
    assert len(names) == len(set(names))


def test_df40_member_layouts():
    assert df40_member("DiT/001/10037.png", default_domain="ff") == ("ff", "001", "10037.png")
    assert df40_member("cdf/frames/id9_id0_0006/349.png", default_domain=None) == (
        "cdf", "id9_id0_0006", "349.png")
    assert df40_member(
        "pixart/cdf/Fake_from_Celeb-real/id0_0000/Celeb-real_id1_0004_171.png",
        default_domain=None,
    ) == ("cdf", "id0_0000", "Celeb-real_id1_0004_171.png")
    # Opaque YouTube-real names carry no identity evidence.
    assert df40_member("SiT/cdf/Fake_from_Youtube-real/00000/1.png", default_domain=None) is None
    assert df40_member("FaceForensics++/original_sequences/youtube/c23/frames/000/000.png",
                       default_domain="ff") == ("ff", "000", "000.png")


def test_source_entities_use_folder_and_conditioning_frame():
    assert source_entities("ff", "306_278", "328.png") == ("ff:278", "ff:306")
    assert source_entities("ff", "227_BMOjVYgYaG8_50007", "125.png") == ("ff:227",)
    # Diffusion archives name the conditioning frame; frame numbers are not clips.
    assert source_entities("ff", "000", "006_199.png") == ("ff:000", "ff:006")
    assert source_entities("ff", "001", "10037.png") == ("ff:001",)
    assert source_entities("cdf", "id0_0000", "Celeb-real_id1_0004_171.png") == (
        "cdf:id0", "cdf:id1")
    assert source_entities("cdf", "00000", "1.png") == ()


def test_ownership_keeps_evaluation_identities_out_of_training():
    ownership = Ownership("seed", {"ff:001", "ff:002"}, {"ff:002", "cdf:id1", "cdf:id2"})
    assert ownership.owner["ff:001"] == "train"
    assert ownership.owner["ff:002"] in {"validation", "test"}
    assert not ownership.accepts("train", ("ff:001", "ff:002"))
    assert not ownership.accepts("train", ())
    split = ownership.owner["cdf:id1"]
    assert ownership.accepts(split, ("cdf:id1",))


def test_external_rows_label_by_folder_and_group_clips():
    members = [
        "heygen_new/fake/frames/0/000.png",
        "heygen_new/fake/frames/0/021.png",
        "heygen_new/real/Clip+abc+P0/00000000.png",
        "stargan/fake/fake/0.jpg",
    ]
    rows = _external_rows("heygen", "test", "face_reenactment", "test/heygen.zip",
                          members[:3], "seed")
    labels = {(row["label"], row["video_id"]) for row in rows}
    assert labels == {("FAKE", "heygen:fake-0"), ("REAL", "heygen:real-Clip+abc+P0")}
    assert all(row["preprocess"] == "canonical:heygen" for row in rows)
    assert all(row["relative_path"].endswith(".jpg") for row in rows)
    assert {row["fake_method"] for row in rows} == {"heygen", ""}


def test_synthesis_folders_are_not_identity_evidence():
    ownership = Ownership("seed", {"ff:001"}, {"cdf:id0", "cdf:id1"})
    groups = {
        ("cdf", "id0_0000"): [("11287.png", "DiT/cdf/Fake_from_Celeb-real/id0_0000/11287.png")],
        ("cdf", "id0_0001"): [("Celeb-real_id1_0004_171.png", "pixart/cdf/x/id0_0001/a.png")],
    }
    split = ownership.owner["cdf:id1"]
    other = "test" if split == "validation" else "validation"
    rows, _ = openworld._sample_groups("seed", other, "dit", openworld.EFS, "a.zip", groups,
                                       ownership, 10, 4)
    # The noise-generated image is accepted in any split; the conditioned one is not.
    assert [row["source_path"] for row in rows] == [
        "DiT/cdf/Fake_from_Celeb-real/id0_0000/11287.png"]
    assert "id0" not in rows[0]["video_id"]
    assert rows[0]["lineage_video_ids"].startswith("synthetic:dit:cdf:")
    rows, _ = openworld._sample_groups("seed", split, "dit", openworld.EFS, "a.zip", groups,
                                       ownership, 10, 4)
    conditioned = [row for row in rows if row["source_path"].startswith("pixart")]
    assert conditioned[0]["identity_id"] == "id1"
