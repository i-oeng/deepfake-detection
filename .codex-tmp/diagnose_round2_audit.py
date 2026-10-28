import csv
from collections import Counter
from pathlib import Path

from deepfake_detection.data.unseen import row_links

round2 = Path("data/manifests/df40_openworld_round2/c7d430eae6ea915427d0/manifest.csv")
fresh = Path("data/manifests/openworld_fresh_v2/b209f9bcef05d95af5c1/manifest.csv")

with round2.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
missing = Counter(
    (row["split"], row["source_domain"], row["fake_method"] or "real", row["label"])
    for row in rows
    if not row_links(row)["identity"]
)
print("MISSING_IDENTITY", sum(missing.values()))
for key, count in missing.most_common():
    print(count, *key)

value = "334c8e8e969686cc"
fields = (
    "sample_id", "source_id", "split", "label", "source_domain", "fake_method",
    "video_id", "identity_id", "lineage_video_ids", "relative_path", "dhash64",
)
for path in (round2, fresh):
    print("DHASH", path)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("dhash64") == value:
                print({field: row.get(field, "") for field in fields})
