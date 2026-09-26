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

A third kind is checked as WORDS, anywhere in a line: the names of private repositories
and builds, and of the second PyBondLab path Stage 3 had until 2026-09-26. Those are kept
as SHA-256 digests, so this file does not spell out what it forbids.

WHAT IT DELIBERATELY DOES NOT FLAG. Naming a private DATASET is not leaking a path.
"our Lehman-ICE panel" is provenance a user needs in order to know where the extended
BBW series came from, and the series itself is published. `tab:monthly_data_availability`
is a LaTeX label, not a directory.

    python -m pytest tests/test_public_boundary.py -q
"""
from __future__ import annotations

import hashlib
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

# Folders a reader may legitimately NAME but that must not appear as a PATH.
PRIVATE_TREES = ("lehman-ice", "monthly_data", "Dropbox", "OneDrive")

# Words that must not appear at all: private repositories and builds, and the second
# PyBondLab path Stage 3 used to have (its flags, environment variable and helpers). SHA-256 of
# the lower-cased word. To add one: hashlib.sha256(word.lower().encode()).hexdigest().
BANNED_WORD_DIGESTS = frozenset({
    "04ba0ac9de0f44dee32f7cd36e0e9152709854d78d5599ccad50fade36319284",
    "2adafe2386fed30974135c2860ed7b7284d63f15544f1d13fc3c200b30cdc7ba",
    "363713cd232bde2660a6362ea98620105cbddecaa6982bc6cec7b0706a95431e",
    "39f2e1ae0c1ca50f521d0b02c17f0a42f0d1d1226d09856ccf49907596c8337e",
    "3a0aa3e491040af444fef23107c093c074da7a6fb67cd75867447d73bf1b8428",
    "49a1f1970f6b603c2ffb6a73465fcb37805d97e88313eb058ae1c449b2de0197",
    "5470ce4ec924f94d6a4dcbd99d065d9975443da3a611776f8d880ff10ecee014",
    "5c0f82ae4bcff38562b710c116bf6284473336441f72dd024e17aae33fb85ed7",
    "771d8f69769f04134f7aaae132da911196000935e3f08235ddab7ca49857f832",
    "824a7440f943a0586b3eee21f50b1daf4d67f4c6c61f24fe7bc2e525cda0500b",
    "8f839dd5cea4375f2500294b434c75f58ff20dfab50e72a14ce35f1146b497ca",
    "9bd9fea0e95fcc334caf71a6f83fc83fabecc1cfa3102033451ac89cdfd6807d",
    "a6e76a03220d9ffc7949ea4a6f953206d5e53c243db104e729f76faa72efa80b",
    "b28a917cb6b16338aeee6b594af21c310cebdfe5c8884b10fa9da71057121b49",
    "b9c60f4cafdd3fde0ee8193835a2c00ef021f35a9f31f1d01209e315a6762c94",
    "cd66d98816ff075497685284c55ea2212eca8e4c425ea60c212f1dfb6ba936e6",
    "cffb7f6824d4d2620a4c1ef65046db58b877ac08ffbd55aef0f4f92caa3ed488",
    "d9bfd0382658b4bdc9d9728223f50cccf2aad0c4dd82fd6b4a236316910bcd4d",
    "ecce767b3f4dc8c4f002d6aa4f7612e45b43a166aa4d05c2791b9507c46388b4",
})

_TOKEN = re.compile(r"[a-z0-9_./-]+")
_PART_SEP = re.compile(r"([-_./])")


def word_candidates(line: str):
    """Every string a banned word could be, in one line: each token, each run of a token's
    parts (split on - _ . /, so a word inside a path or an identifier is found), and each pair
    of adjacent tokens (for two-word phrases). Lower-cased."""
    toks = _TOKEN.findall(line.lower())
    for t in toks:
        pieces = _PART_SEP.split(t)
        parts, seps = pieces[0::2], pieces[1::2]
        for i in range(len(parts)):
            s = parts[i]
            yield s
            for j in range(i + 1, len(parts)):
                s = s + seps[j - 1] + parts[j]
                yield s
    for a, b in zip(toks, toks[1:]):
        yield f"{a} {b}"


def banned_words_in(line: str, digests=BANNED_WORD_DIGESTS) -> bool:
    return any(hashlib.sha256(c.encode()).hexdigest() in digests for c in word_candidates(line))

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
        if banned_words_in(line):
            hits.append((i, "a private name, or the retired second PyBondLab path", line))
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
    "see C:/x/Xyzzy-Quux's branch",          # inside a path, with a possessive
    "import zork_plugh as z",                # inside an identifier
    "then take the frob route, longer",      # a two-word phrase
])
def test_the_word_guard_actually_fires(line):
    """The banned words are digests, so they cannot be planted here without spelling them out.
    The matching is planted instead, with stand-in words: a guard that misses these would miss
    the real ones."""
    stand_ins = {hashlib.sha256(w.encode()).hexdigest()
                 for w in ("xyzzy-quux", "zork", "frob route")}
    assert banned_words_in(line, stand_ins), f"the word guard did NOT catch: {line}"


def test_every_banned_word_is_a_sha256_digest():
    assert len(BANNED_WORD_DIGESTS) >= 19
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in BANNED_WORD_DIGESTS)


@pytest.mark.parametrize("line", [
    r'| `{local_destination}` | Local path | `~/Downloads` or `C:\Users\YourName\Downloads` |',
    "# Or, put your path, e.g., for Windows: C:\\Users\\proj\\",
    "file was extended by a join against our Lehman-ICE panel, not rebuilt.",
    r'"tab:monthly_data_availability":',
    "so a synced folder (Dropbox, OneDrive) sees one complete file instead of every",
    "questions -> openbondassetpricing.com",
    "python -m pip install --no-deps pybondlab==0.3.0",
    "DataUncertaintyAnalysis(use_fast_path=True) and fastrun.py",
])
def test_the_guard_leaves_legitimate_lines_alone(line):
    """The cost of a noisy gate is that it gets switched off. These must all pass."""
    assert not _scan(line), f"false positive on: {line}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
