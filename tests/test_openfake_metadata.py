from collections import Counter

from scripts.openfake_metadata import (
    EXCLUDED,
    PERSON,
    family,
    lfw_name,
    plan_download,
    public_figures,
)


def test_fine_tunes_join_their_base_family() -> None:
    assert family("flux.1-dev") == family("flux-mvc5000") == "flux1"
    assert family("flux.2-klein-4b") == "flux2"
    assert family("sdxl-juggernaut") == family("cyberrealistic-pony-v14.1") == "sdxl"
    assert family("playground-v2.5-1024px-aesthetic") == "playground"
    assert family("sd-turbo") == "sd2" and family("stable-diffusion-v1-5") == "sd1"
    assert family("amateur-photography-v6") == "other:amateur-photography-v6"


def test_public_figures_match_full_lfw_names_only() -> None:
    names = {"Kelly Brook", "George W Bush", "Sean Paul"}
    assert public_figures("A photograph shows Kelly Brook walking outside", names) == [
        "Kelly Brook"]
    assert public_figures("George W. Bush waves; a poster for Sean Paul's song", names) == [
        "George W Bush", "Sean Paul"]
    assert public_figures("A poster for The Dark Knight with Paul", names) == []


def test_lfw_names_come_from_image_filenames() -> None:
    assert lfw_name("Aaron_Eckhart_0001.jpg") == "Aaron Eckhart"
    assert lfw_name("lfw/George_W_Bush/George_W_Bush_0042.jpg") == "George W Bush"


def test_person_prompts_and_excluded_generators() -> None:
    assert PERSON.search("A photograph of a serene couple practicing yoga")
    assert PERSON.search("She stands in the snow wearing a green scarf")
    assert not PERSON.search("A bowl of fruit on a wooden table, studio lighting")
    assert all(EXCLUDED.search(m) for m in ("wan-video-2.2", "animagine-xl-3.1", "pony", "veo-3"))
    assert not any(EXCLUDED.search(m) for m in ("flux.1-dev", "sdxl", "cyberrealistic-pony-v14.1"))


def test_plan_download_meets_quotas_with_few_row_groups() -> None:
    fake, real = ("fake", "m"), ("real", "r")
    units = {
        ("a", 0): Counter({fake: 2}),
        ("a", 1): Counter({fake: 1, real: 1}),
        ("b", 0): Counter({real: 2, fake: 2}),
    }
    costs = {("a", 0): 10, ("a", 1): 10, ("b", 0): 10}
    plan = plan_download(units, costs, {fake: 2, real: 2})
    assert plan["picked"] == [("b", 0)]
    assert plan["bytes"] == 10 and plan["shortfall"] == {}
    short = plan_download(units, costs, {fake: 9})
    assert short["shortfall"] == {fake: 5} and len(short["picked"]) == 3
