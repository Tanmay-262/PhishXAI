"""Build PhishXAI dataset v1 (B2).

Sources (data/raw/, never modified):
  - openphish_feed_2026-11-17.txt   : real phishing URLs (observed)
  - tranco_JZWZY_top10k_2026-11-17.txt : legitimate-domain source (Tranco list ID JZWZY)
  - public_suffix_list.dat           : PSL for registered-domain extraction

Outputs (data/processed/dataset_v1/):
  train.csv / val.csv / test.csv  (columns: url,label,source,url_kind,
                                   path_constructed,registered_domain,extra)
  manifest.json                   (provenance, counts, cleaning decisions,
                                   split verification)

Deterministic: SEED=42 for all sampling/shuffling. Raw files untouched.
No model training happens here.
"""
import csv, hashlib, json, random
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "dataset_v1"
SEED = 42

OPENPHISH = RAW / "openphish_feed_2026-11-17.txt"
TRANCO = RAW / "tranco_JZWZY_top10k_2026-11-17.txt"
PSL = RAW / "public_suffix_list.dat"

# Path pool for synthetically constructed legitimate URLs.
# Mix: bare-domain constructions (path "/") and real-looking page paths,
# so the model cannot use "has a non-trivial path" as a phishing signal.
PATH_POOL = [
    "/", "/en", "/search", "/about", "/products", "/support",
    "/login", "/images/hero.png", "/blog/2026/announcements",
    "/checkout/cart", "/contact", "/help/faq", "/downloads",
]
BARE_FRACTION = 0.4  # 40% bare-domain, 60% path-constructed
SAMPLE_FROM_TOP_N = 1000  # sample legit domains from Tranco top-1000 ranks


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------- Public Suffix List ----------------

def load_psl(path: Path):
    """Load PSL into exact + wildcard rule sets (MVP: exception rules ignored, documented)."""
    exact, wildcard = set(), set()
    ignored_exceptions = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        if s.startswith("!"):
            ignored_exceptions += 1
            continue
        if s.startswith("*."):
            wildcard.add(s[2:])
        else:
            exact.add(s)
    return exact, wildcard, ignored_exceptions


def registered_domain(host: str, exact: set, wildcard: set) -> str:
    """Registered domain (eTLD+1) using PSL rules. Falls back to host for
    IPs and private domains not matched by any PSL rule."""
    host = host.lower().rstrip(".")
    labels = host.split(".")
    for n in (3, 2, 1):  # longest candidate public suffix first
        if n > len(labels):
            continue
        suffix = ".".join(labels[-n:])
        is_rule = suffix in exact or (n >= 2 and ".".join(suffix.split(".")[1:]) in wildcard)
        if is_rule:
            return ".".join(labels[-(n + 1):]) if len(labels) > n else host
    return host


# ---------------- Phishing side (observed, real URLs) ----------------

def load_phishing(exact, wildcard):
    rows, discarded, dup_exact, dup_norm = [], [], 0, 0
    seen_exact, seen_norm = set(), set()
    for lineno, line in enumerate(OPENPHISH.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        url = line.strip()
        if not url:
            continue
        p = urlparse(url)
        host = p.netloc.split("@")[-1].split(":")[0]
        if p.scheme not in ("http", "https") or not host:
            discarded.append((lineno, url, "unparseable/scheme"))
            continue
        if url in seen_exact:
            dup_exact += 1
            continue
        norm = p.netloc.lower() + "|" + p.path.lower().rstrip("/")
        if norm in seen_norm:
            dup_norm += 1
            continue
        seen_exact.add(url)
        seen_norm.add(norm)
        rd = registered_domain(host, exact, wildcard)
        rows.append({
            "url": url, "label": "phishing", "source": "openphish_feed_2026-11-17",
            "url_kind": "observed", "path_constructed": "false",
            "registered_domain": rd, "extra": f"feed_line={lineno}",
        })
    return rows, discarded, dup_exact, dup_norm


# ---------------- Legitimate side (Tranco domains, constructed URLs) ----------------

def load_legit(phishing, n_target, exact, wildcard, rng):
    """Sample n_target domains from Tranco top-1000 (seeded), construct URLs.

    Every constructed URL is flagged path_constructed=true and
    url_kind=constructed, so downstream docs never claim these are
    verified/crawled URLs.
    """
    tranco = []
    for lineno, line in enumerate(TRANCO.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if "," not in line:
            continue
        rank_s, domain = line.split(",", 1)
        if not domain.strip():
            continue
        tranco.append((int(rank_s), domain.strip()))
    tranco.sort(key=lambda t: t[0])
    top = tranco[:SAMPLE_FROM_TOP_N]

    phish_rds = {r["registered_domain"] for r in phishing}
    collision = sorted({d for _, d in top if d in phish_rds})
    pool = [(r, d) for r, d in top if d not in phish_rds]
    chosen = rng.sample(pool, min(n_target, len(pool)))

    rows = []
    for rank, domain in sorted(chosen, key=lambda t: t[0]):
        bare = rng.random() < BARE_FRACTION
        path = "/" if bare else rng.choice([p for p in PATH_POOL if p != "/"])
        rows.append({
            "url": f"https://{domain}{path}", "label": "legitimate",
            "source": "tranco_JZWZY_top10k_2026-11-17",
            "url_kind": "constructed", "path_constructed": "true" if not bare else "false(bare-domain)",
            "registered_domain": domain, "extra": f"tranco_rank={rank}",
        })
    return rows, collision, len(tranco)


# ---------------- Split: registered-domain-grouped, no overlap ----------------

def grouped_split(phishing, legit, ratios, rng):
    groups = {}
    for r in phishing + legit:
        groups.setdefault(r["registered_domain"], []).append(r)
    keys = list(groups.keys())
    rng.shuffle(keys)
    n = len(keys)
    cuts = [int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))]
    splits = {"train": keys[:cuts[0]], "val": keys[cuts[0]:cuts[1]], "test": keys[cuts[1]:]}
    out = {}
    for name, ks in splits.items():
        rows = [r for k in ks for r in groups[k]]
        rng.shuffle(rows)
        out[name] = rows
    # leakage verification
    sets = {name: set(ks) for name, ks in splits.items()}
    overlap = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
    return out, overlap


def main():
    rng = random.Random(SEED)
    exact, wildcard, ignored_exc = load_psl(PSL)

    phish, discarded, dup_exact, dup_norm = load_phishing(exact, wildcard)
    n_phish = len(phish)
    legit, collision, tranco_total = load_legit(phish, n_phish, exact, wildcard, rng)

    splits, overlap = grouped_split(phish, legit, (0.8, 0.1, 0.1), rng)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["url", "label", "source", "url_kind", "path_constructed", "registered_domain", "extra"]
    for name, rows in splits.items():
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    def class_counts(rows):
        return {"phishing": sum(1 for r in rows if r["label"] == "phishing"),
                "legitimate": sum(1 for r in rows if r["label"] == "legitimate")}

    manifest = {
        "dataset_version": "v1", "seed": SEED,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size,
                               "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}
                      for p in (OPENPHISH, TRANCO, PSL)},
        "psl": {"ignored_exception_rules": ignored_exc, "note": "MVP ignores PSL exception ('!') rules"},
        "phishing": {"raw_lines": n_phish + dup_exact + dup_norm + len(discarded),
                      "clean": n_phish, "discarded": discarded,
                      "exact_dupes_removed": dup_exact, "normalized_dupes_removed": dup_norm,
                      "unique_registered_domains": len({r['registered_domain'] for r in phish})},
        "legitimate": {"constructed": len(legit), "observed_real_urls": 0,
                       "note": "All legitimate URLs are constructed from Tranco top-1000 domains; no Tranco URL was crawled or verified.",
                       "top_domains_collision_with_phishing": collision, "tranco_domains_total": tranco_total},
        "groups": {"total": len({r['registered_domain'] for r in phish + legit}),
                   "ratios": [0.8, 0.1, 0.1],
                   "registered_domain_overlap_across_splits": sorted(overlap) or "NONE - verified disjoint"},
        "splits": {k: {"rows": len(v), "classes": class_counts(v)} for k, v in splits.items()},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
