# CS231 Final M0 Dataset — FIX2

Task: closed-set image classification of one Vietnamese polymer banknote.
Classes: 10000, 20000, 50000, 100000, 200000, 500000

Source candidate pool before visual review: 9984
Classification-Centric Visual Audit V2 removals: 23
Final images: 9961

Balancing:
- No class undersampling/balancing was performed.
- All samples that passed Visual Audit V2 were retained.

Split:
- Approximate target: 80% train / 10% dev / 10% test.
- Split unit is a leakage group, not an individual image.
- money_v1: same capture_date is kept in one split.
- banknotes_v2: historical capture-sequence grouping is preserved.
- exact source/crop SHA and UUID-like lineage are unioned when available.
- group integrity takes priority over exact ratio.

Naming:
- Canonical filename format: <denomination>_<6-digit-index>.png
- Index is global within each denomination and does not reset by split.
- Source identity is removed from filename but preserved in manifest.csv.

Representation:
- one-banknote source-stage logic;
- bbox short side >= 120 px on available export;
- crop = banknote bbox + 20% natural context;
- aspect ratio preserved;
- no permanent 224x224 resize.

Limitation:
- Public sources do not provide reliable physical-banknote IDs.
- Therefore the dataset does not claim physical-note-disjoint splits.
