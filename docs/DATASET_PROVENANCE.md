# PhishXAI - Dataset v1 Provenance (B2)

Built: 2026-11-17 (IST) / 2026-11-16T21:44 UTC (builder `built_at_utc`)
Builder: `scripts/build_dataset_v1.py`, SEED=42, deterministic.
Raw files in `data/raw/` are preserved **unchanged**; all cleaning and
transformations wrote to `data/processed/dataset_v1/` only.
Raw + processed data are excluded from Git (.gitignore).

## Sources

| File (data/raw/) | Source URL | Retrieved | Size | Format | Labels | Licensing |
|---|---|---|---|---|---|---|
| openphish_feed_2026-11-17.txt | https://openphish.com/feed.txt | 2026-11-17 03:12 IST | 18,086 B | one phishing URL per line | phishing (observed, no per-URL confirmation) | OpenPhish free feed; URLs are reported phishing pages - inspect text only, never fetch or execute |
| tranco_JZWZY_top10k_2026-11-17.txt | https://tranco-list.eu/download/JZWZY/10000 | 2026-11-17 03:14 IST | 187,074 B | `rank,domain`, 10,000 rows | **legitimate-domain source, not verified URLs** | CC-BY (Tranco); daily-updated ranking (CUA + Cloudflare Radar) |
| public_suffix_list.dat | https://publicsuffix.org/list/public_suffix_list.dat | 2026-11-17 03:13 IST | 341,648 B | PSL | - | CC0 |

Tranco note: the legacy `/download/top-10k.csv` URL 404s; current lists are
served per list-ID (latest = JZWZY at collection time). List JZWZY is a
*domain* ranking - we use it as a legitimate-domain source, and disclose that
below.

## Dataset v1 numbers (from manifest.json)

- Phishing: 300 raw feed lines -> 300 kept (0 exact dupes, 0 normalized dupes,
  0 discarded malformed). 261 unique registered domains.
- Legitimate: 300 constructed URLs (seeded sample of Tranco top-1000 domains,
  40% bare-domain `https://d/` + 60% path-constructed from a 13-path pool).
  **Observed/crawled legitimate URLs: 0.** Every constructed URL is flagged
  `url_kind=constructed` and `path_constructed` in the CSV.
- One collision: `weebly.com` appears in the Tranco top-1000 and a phishing
  feed URL uses a weebly.com subdomain; the legit sample excluded it.
- Total: 600 rows, 561 registered-domain groups.

## Split (domain-grouped, ratios 80/10/10)

| split | rows | phishing | legitimate | unique registered domains |
|---|---|---|---|---|
| train | 479 | 238 | 241 | 448 |
| val   | 61  | 33  | 28  | 56 |
| test  | 60  | 29  | 31  | 57 |

Leakage check: registered-domain sets across the three splits are verified
disjoint (manifest: "NONE - verified disjoint"). Split is by *group count*
(449/56/56 groups), not row count, so per-class rows are uneven by design.
80/10/10 was applied at the group level; the 80/10/20 plan shape was
adjusted because 20% test would leave only ~50 rows - too thin for a
meaningful test set at this data size.

## Known limitations (must be in the report)

1. 300 phishing URLs is small; PR-AUC/F1 have wide uncertainty, and 29-33
   phishing test/val rows give coarse per-class recall estimates.
2. Legitimate URLs are synthetically constructed paths on real popular
   domains: the model can learn "popular TLD+1 domain => legitimate" in a
   way that does not transfer to real, arbitrary legit URLs. Path-construction
   is a documented, disclosed method, not crawled evidence.
3. OpenPhish feed lines are reported sightings; labels are not per-URL
   verified.
4. PSL: 8 exception (`!`) rules ignored - negligible effect on top-10k.
5. Single-point-in-time feed; phishing URLs may be dead links now.
