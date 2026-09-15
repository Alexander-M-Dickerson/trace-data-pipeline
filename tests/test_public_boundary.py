# -*- coding: utf-8 -*-
"""
test_public_boundary.py
=======================
Nothing in this repository may carry the author's machine or the private
repositories that sit beside it on that machine.

This is a PUBLIC repository. Everything tracked here is read by people who will never
see the machine it was written on, and two kinds of detail leak across that boundary
without anyone meaning them to:

  * an absolute path with a real home directory in it -- `C:\\Users\\<someone>\\...`,
    `/home/<someone>/...`. It names a person, and it makes the line unrunnable
    anywhere else.
  * a path into a sibling repository that is not public -- the licensed Lehman/ICE
    panel, the factor-production repo, a Dropbox tree. A reader cannot follow it and it
    discloses how the author's disk is laid out.

`stage3/tests/test_stage3_contract.py` has guarded exactly this since stage 3 shipped,
but ONLY under `stage3/`. Stages 0, 1 and 2 had no equivalent, which is where new work
actually lands -- the duration-adjusted benchmark work of 2026-09 added files to
`stage2/` and nothing would have looked at them. This covers every tracked file.

WHAT IT DELIBERATELY DOES NOT FLAG. Naming a private DATASET is not leaking a path.
"our Lehman-ICE panel" is provenance a user needs in order to know where the extended
BBW series came from, and the series itself is published. `tab:monthly_data_availability`
is a LaTeX label, not a directory. The rule is therefore about paths and usernames, not
about words.

    python -m pytest tests/test_public_boundary.py -q
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {".py", ".md", ".txt", ".sh", ".json", ".yml", ".yaml", ".cfg",
                 ".ini", ".toml", ".tex", ".csv", ".bat", ".r", ".do"}

# Files that must name the forbidden things in order to forbid them.
SELF = {
    "tests/test_public_boundary.py",
    "stage3/tests/test_stage3_contract.py",
}

# Stand-ins a reader is meant to replace. A placeholder is the CORRECT way to write
# these paths, so the guard must not push anyone into deleting them.
PLACEHOLDERS = {
    "yourname", "your_name", "username", "user", "<user>", "youruser", "your_user",
    "proj", "01_trace", "me", "name", "path", "you",
    # the WRDS cloud home is /home/<your institution>/<your login>/ -- both stand-ins
    "university", "institution", "wrds_username",
}

# A private sibling repository, flagged only where it reads as a PATH.
PRIVATE_TREES = ("lehman-ice", "trace_duckdb", "osbap_data", "PyBondLab-Dev",
                 "monthly_data", "drr_replication", "Dropbox", "OneDrive",
                 "private_lhm_ice", "DRR_MUA")

# `C:\Users\<name>`, `/home/<name>`, `/Users/<name>` -- the name is captured so a
# placeholder can be let through.
ABS_USER = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/home/|/Users/)([A-Za-z0-9._{}<>$-]+)")

# A private tree with a path separator on one side of it.
PRIVATE_PATH = re.compile(
    r"(?:[\\/]|\b[A-Za-z]:[\\/])(?:" + "|".join(re.escape(t) for t in PRIVATE_TREES) + r")\b"
    r"|(?:" + "|".join(re.escape(t) for t in PRIVATE_TREES) + r")[\\/]")

# A credential that identifies a person rather than a role.
IDENTITY = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"          # an email address
    r"|wrds_username\s*=\s*[\"'](?!\s*$)(?!YOUR)[A-Za-z0-9_]+[\"']",
    re.IGNORECASE)

# Addresses that are meant to be here. The maintainer's university address is the
# PUBLISHED contact for this project -- it is in the README, the FAQ and every stage
# guide on purpose, so that someone who hits a problem can write to a person. Publishing
# a contact address is the opposite of leaking one; what this guard is for is a home
# directory or a private tree that nobody chose to disclose. Any OTHER personal address
# still fails, which is the point of naming this one rather than dropping the rule.
PUBLISHED_CONTACT = ("alexander.dickerson1@unsw.edu.au",)
PUBLIC_EMAIL_OK = re.compile(
    "|".join([r"openbondassetpricing", r"noreply@", r"example\.com", r"@wrds\b"]
             + [re.escape(a) for a in PUBLISHED_CONTACT]), re.I)


def tracked_text_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    files = []
    for rel in out.split("\0"):
        if not rel or rel in SELF:
            continue
        p = ROOT / rel
        if p.suffix.lower() in TEXT_SUFFIXES and p.is_file():
            files.append(p)
    return sorted(files)


def _scan(text: str) -> list[tuple[int, str, str]]:
    """(line number, what rule fired, the line) for one file's content."""
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for m in ABS_USER.finditer(line):
            if m.group(1).strip("<>{}$").lower() not in PLACEHOLDERS:
                hits.append((i, f"absolute home directory ({m.group(1)})", line))
        if PRIVATE_PATH.search(line):
            hits.append((i, "a path into a private sibling repository", line))
        for m in IDENTITY.finditer(line):
            if not PUBLIC_EMAIL_OK.search(m.group(0)):
                hits.append((i, f"a personal identity ({m.group(0)})", line))
    return hits


def test_no_private_paths_or_identities_anywhere_tracked():
    files = tracked_text_files()
    assert files, "git ls-files returned nothing -- the guard would pass vacuously"

    bad = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for ln, why, line in _scan(text):
            bad.append(f"{p.relative_to(ROOT).as_posix()}:{ln}: {why}\n      {line.strip()[:110]}")

    assert not bad, (
        f"{len(bad)} line(s) carry something that must not leave the author's machine:\n  "
        + "\n  ".join(bad))


@pytest.mark.parametrize("line,why", [
    (r'PANEL = r"C:\Users\jsmith\Documents\panel.parquet"', "a real home directory"),
    ('PANEL = "/home/jsmith/data/panel.parquet"', "a real home directory"),
    (r'SRC = r"..\..\lehman-ice\unified-panel\outputs"', "a private sibling repo path"),
    ('SRC = "/c/Users/x/Dropbox/factors.parquet"', "a Dropbox path"),
    ('wrds_username = "jsmith01"', "a personal WRDS login"),
    ("contact: someone@university.ac.uk", "a personal email address"),
])
def test_the_guard_actually_fires(line, why):
    """A gate nobody has watched fail is decoration. Each rule gets a planted violation."""
    assert _scan(line), f"the guard did NOT catch {why}: {line}"


@pytest.mark.parametrize("line", [
    r'| `{local_destination}` | Local path | `~/Downloads` or `C:\Users\YourName\Downloads` |',
    "# Or, put your path, e.g., for Windows: C:\\Users\\proj\\",
    "file was extended by a join against our Lehman-ICE panel, not rebuilt.",
    r'"tab:monthly_data_availability":',
    "so a synced folder (Dropbox, OneDrive) sees one complete file instead of every",
    "questions -> openbondassetpricing.com",
])
def test_the_guard_leaves_legitimate_lines_alone(line):
    """The cost of a noisy gate is that it gets switched off. These must all pass."""
    assert not _scan(line), f"false positive on: {line}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
