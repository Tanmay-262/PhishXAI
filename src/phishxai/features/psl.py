"""Public Suffix List handling (B3).

Local-file only; no network access. PSL is read from
data/raw/public_suffix_list.dat (gitignored, present on the dev machine).

MVP limitation: PSL exception rules (lines starting '!') are counted but
ignored. At most a handful of top-10k domains are affected; this is
documented in docs/DATASET_PROVENANCE.md and the report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Set, Tuple


def load_psl(path: str | Path) -> Tuple[Set[str], Set[str], int]:
    """Parse a PSL file into (exact_suffixes, wildcard_suffixes, n_ignored_exceptions).

    Exact rules like 'co.uk' go to exact; wildcard rules like '*.ck' are
    stored WITHOUT the leading '*' ('ck'), meaning ANY label under .ck is
    itself a public suffix.
    """
    exact: Set[str] = set()
    wildcard: Set[str] = set()
    ignored_exceptions = 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        if s.startswith("!"):
            ignored_exceptions += 1
            continue
        if s.startswith("*"):
            wildcard.add(s[2:])
        else:
            exact.add(s)
    return exact, wildcard, ignored_exceptions


def registered_domain(host: str, exact: Set[str], wildcard: Set[str]) -> str:
    """Return the registered domain (eTLD+1) for a host using PSL rules.

    Checks candidate public-suffix lengths 3, 2, 1 (longest first). When
    no PSL rule matches (IPs, private domains, unknown TLDs) the whole
    host is returned - a documented fallback, not an error.
    """
    host = (host or "").lower().rstrip(".")
    if not host:
        return ""
    labels = host.split(".")
    for n in (3, 2, 1):
        if n > len(labels):
            continue
        suffix = ".".join(labels[-n:])
        is_wildcard_rule = n >= 2 and ".".join(suffix.split(".")[1:]) in wildcard
        if suffix in exact or is_wildcard_rule:
            return ".".join(labels[-(n + 1):]) if len(labels) > n else host
    return host
