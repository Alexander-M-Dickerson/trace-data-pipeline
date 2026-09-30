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

# Words that must not appear at all: private repositories, builds and datasets, the author's
# WRDS login, the paper's source files, and the second PyBondLab path Stage 3 used to have (its
# flags, environment variable and helpers). SHA-256 of the lower-cased word. A hyphen and an
# underscore count as the same letter, so either spelling is caught. To add one:
# hashlib.sha256(word.lower().encode()).hexdigest().
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
    # added 2026-09-26, from what an adversarial read got past this list
    "d898e36222543a96537ac800fb8304b73102c81deb27155c5fd345bcd302a224",
    "bc9afd4d92878a2608dca20fdb6eb882649c35cba2e109358fee2cb9d056803f",
    "5c11fe1bd5a7f2ed5c2b13dd18d9fc668891588bd071f12efe2418b5b68c33ce",
    "cb81cbc1d9c6ac7acd52879e6ac8e57bf4e9042fe2b0ddb7a65fa06c6a1a1714",
    "f67098301b6788c19af2853a0e5ad776fda05572b1b1224df46c42d1ad323cbb",
    "37e964e4dcf4743b3bf941d41a527921d6068173184df92406fe1039675c5279",
    "c483466bcf4be5925ad6f868c6706e507ce9ef8aec9470a6f4ba91816ba15ef0",
})

# A second list. This package's code, comments, tests, docs and CHANGELOG describe what a
# change does to the method or an exhibit, never where the request came from; these words say
# the latter. Digests as above, so this file does not spell them (added 2026-09-29).
REVIEW_WORD_DIGESTS = frozenset({
    "81803dd4e410a92bc5d510ea539a5d38493e3fe26b707dbac6a63b9e3a7b4d90",
    "f9da47b6ea1c414fb0fb0f2516948f296f2fcee5e74b059312311880b766d3aa",
    "895e8bde950d49da947cb09031a3d637dc2617011ceed6e7cdf884f05ac7c0f0",
    "b26ed6c1fec883c3d1e803062fc1b97d573e2c1178833999c6385e647fbca25a",
    "64432b10de6549a3cfe841359250df4f58e5d76d1c9a3efe531462a9ebe2584e",
    "7616aa5837a030b97cff231b22e5436b40eb30495c099db48e9f1a1b3581f93a",
    "d764a2aeb0d30f81330412082c861eaccfd59ad95f4a74ee7351283a1297ec93",
    "4c8709a63b2afbf43067c609c926177c118094cd663c72af517cbc047f01a1e6",
    "f77b9b98c0984493a61a9fa535cfbe1940aa3fc6b757c6159e917e948445280f",
    "45805b0c133f5f57ca74e3c66b822b3380f920e0e370fc36197027fbc4c1d763",
    "a97c407b937e02704d8988143d95f045d78d0bb9aae73d8e1da1af62d8bf0ca4",
    "cd78a69d6f80defe66074ac2e726a0154f44375e7327ac1336c58c6e3cbf1448",
    "679ea14b8ea6d7b60201670a0f81f3c0b778252f05a16c72ecd6b9c4ed1d9ef6",
    "3fb5ae0df969c63535d35614cdb406119a8743082304671086590f5c44a4ef45",
    "6fae67dfa5027cc5ef21c65fb4504c67f90702658a83578ded9ad974a712ca01",
    "d84faaa4c1ed6cd076b6d9ad84bc5322d7e002c6f1290b0340a8deb77b81e78a",
    "45c8b8b5edc67f8d4dd039cd1c8d7987120e1ea0bea794ee45d6973dc5d535c9",
    "341fc7f2a16ea1127287819855c9ddfe6e91665654ed7688877235d03d36a2b7",
    "1c7291c2e0b65e2ed0524ab041fd8ef4307218f3c9c1879404d5308aac9a1bf4",
    "3ffd102f9fbe494b258612b6f53f80073ca090493dc33d999dbbb20e626bfa25",
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
    """True if any candidate, or the same candidate with its hyphens read as underscores or as
    spaces, hashes to a banned word. A digest stores ONE spelling; this catches the others."""
    for c in word_candidates(line):
        for v in {c, c.replace("-", "_"), c.replace("_", "-"), c.replace("-", " "),
                  c.replace("_", " ")}:
            if hashlib.sha256(v.encode()).hexdigest() in digests:
                return True
    return False

# `C:\Users\<name>`, `/home/<name>`, `/Users/<name>` -- the name is captured so a
# placeholder can be let through.
ABS_USER = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/home/|/Users/)([A-Za-z0-9._{}<>$-]+)")

# A path on a lettered drive. `C:\Users\<name>` is ABS_USER's to judge (placeholders pass);
# any other drive path -- E:\work\..., F:/data -- is the author's disk layout. The letter must
# start a token, so `https://` is not a drive.
DRIVE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]{1,2}(?!Users[\\/])[A-Za-z]")

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


def tracked_paths() -> list[str]:
    # --others --exclude-standard: a new file is checked before `git add`, not after.
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return sorted({r for r in out.split("\0") if r})


def tracked_text_files() -> list[Path]:
    """EVERY tracked file, whatever its suffix: `.gitignore` and `LICENSE` have none, and a
    notebook or a spreadsheet added later would not be on any list of text suffixes. Binary
    content is decoded with replacement, so a name inside it is still read."""
    return [ROOT / rel for rel in tracked_paths() if rel not in SELF and (ROOT / rel).is_file()]


def _scan(text: str) -> list[tuple[int, str, str]]:
    """(line number, what rule fired, the line) for one file's content."""
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for m in ABS_USER.finditer(line):
            if m.group(1).strip("<>{}$").lower() not in PLACEHOLDERS:
                hits.append((i, f"absolute home directory ({m.group(1)})", line))
        if PRIVATE_PATH.search(line):
            hits.append((i, "a path into a private sibling repository", line))
        if DRIVE_PATH.search(line):
            hits.append((i, "a path on the author's drive", line))
        for m in IDENTITY.finditer(line):
            if not PUBLIC_EMAIL_OK.search(m.group(0)):
                hits.append((i, f"a personal identity ({m.group(0)})", line))
        if banned_words_in(line):
            hits.append((i, "a private name, or the retired second PyBondLab path", line))
        if banned_words_in(line, REVIEW_WORD_DIGESTS):
            hits.append((i, "a change described by where it came from: say what it does "
                            "instead", line))
    return hits


def test_no_private_paths_or_identities_anywhere_tracked():
    files = tracked_text_files()
    assert files, "git ls-files returned nothing -- the guard would pass vacuously"

    bad = []
    for p in files:
        text = p.read_bytes().decode("utf-8", errors="replace")
        for ln, why, line in _scan(text):
            bad.append(f"{p.relative_to(ROOT).as_posix()}:{ln}: {why}\n      {line.strip()[:110]}")

    assert not bad, (
        f"{len(bad)} line(s) carry something that must not leave the author's machine:\n  "
        + "\n  ".join(bad))


def test_no_tracked_file_or_folder_is_named_for_something_private():
    """The contents are scanned above. A NAME is published too, in every listing of the repo."""
    bad = [rel for rel in tracked_paths()
           if rel not in SELF and (banned_words_in(rel) or PRIVATE_PATH.search("/" + rel))]
    assert not bad, f"tracked paths named for something private: {bad}"


# ---------------------------------------------------------------- pointers a reader cannot follow
# A file name in this repository must name a file IN it. The exceptions live elsewhere by
# design, and say where. Found three times before this existed: a private linker contract, the
# paper's LaTeX sources, and notes files from the private build.
FILES_THAT_LIVE_ELSEWHERE = {
    "verify_release.py": "ships inside each published release archive",
    "SCHEMA.md": "ships inside the bond-firm linker bundle",
    "extract.py": "PyBondLab's own module",
    "illiq_helper_functions.py": "in the reference build a maintainer points STAGE2_REFERENCE_ROOT at",
    "resource_tracker.py": "Python's own module, quoted in a traceback",
    "summary.md": "written by stage3/tools/compare_runs.py into the comparison folder of a run",
}
# A bare name or one written as a path (`docs/CONTRACTS.md`); the last part is what is checked.
_FILE_REF = re.compile(r"(?<![\w/.*-])((?:[\w.-]+/)*[A-Za-z_][\w-]*\.(?:py|md))\b")
# Item codes of private notes: "debug M10", "learnings L13", "upstream lines 232-355".
_NOTE_CODE = re.compile(r"\b(?:debug(?:\.md)?|learnings)\s+[A-Z]\d+\b|perf_learnings\w*"
                        r"|\bupstream lines\s+\d+|\(lines\s+\d{3,}-\d{3,}", re.I)


def unfollowable(text: str, present: set[str]) -> list[tuple[int, str]]:
    """(line, what) for every file name that is not in the repository, and every note code."""
    hits = []
    # LaTeX writes _ as \_, and inside a Python string as \\_: either way it is an underscore.
    for i, line in enumerate(re.sub(r"\\+_", "_", text).splitlines(), 1):
        for m in _FILE_REF.finditer(line):
            name = m.group(1).rsplit("/", 1)[-1]
            if name not in present and name not in FILES_THAT_LIVE_ELSEWHERE:
                hits.append((i, f"names {m.group(1)}, which is not in this repository"))
        for m in _NOTE_CODE.finditer(line):
            hits.append((i, f"a private note's code ({m.group(0)})"))
    return hits


def test_every_file_a_tracked_file_names_is_in_the_repository():
    present = {Path(r).name for r in tracked_paths()}
    bad = []
    for rel in tracked_paths():
        if rel in SELF or not rel.endswith((".py", ".md", ".sh", ".txt", ".json")):
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        bad += [f"{rel}:{i}: {why}" for i, why in unfollowable(text, present)]
    assert not bad, "pointers a public reader cannot follow:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("line", [
    "#   cfg.LINKER_WINDOW; its authority is the linker's own contract (CONTRACTS.md s4).",
    "# monthly_pi_fast (upstream lines 232-355)",
    "       -- NULL count into 0.0 (bit us on p_zro_adj -- debug.md M10)",
    "# Shape matters (measured, learnings L13): 95 separate ungrouped",
    '"""momentum.py -- from the reference implementation (lines 6227-6356)."""',
    "see docs/CONTRACTS.md s4",
    "as recorded in lewellen/LEARNINGS.md",
])
def test_the_pointer_guard_fires_on_what_it_was_built_from(line):
    assert unfollowable(line, {"momentum.py"}), f"not caught: {line}"


@pytest.mark.parametrize("line", [
    "see stage2/README_stage2.md and _run_stage2.py",
    r"set in the \texttt{\_trace\_settings.py} script",
    "- **`_run_*_trace.py` had no `__main__` guard**",
    "the linker bundle's own SCHEMA.md, which ships inside the zip",
])
def test_the_pointer_guard_leaves_real_files_alone(line):
    present = {"README_stage2.md", "_run_stage2.py", "_trace_settings.py"}
    assert not unfollowable(line, present), unfollowable(line, present)


@pytest.mark.parametrize("line,why", [
    (r"SRC = r'E:\work\build-2030-01-01\stage2'", "a path on another drive"),
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


@pytest.mark.parametrize("line", [
    "see the xyzzy_quux build",              # the other spelling of a hyphenated word
    "the zork-plugh tree",                   # stored with neither separator: still a token
    "the frob-route option",                 # a two-word phrase written with a hyphen
])
def test_the_word_guard_reads_a_hyphen_and_an_underscore_as_one(line):
    stand_ins = {hashlib.sha256(w.encode()).hexdigest()
                 for w in ("xyzzy-quux", "zork", "frob route")}
    assert banned_words_in(line, stand_ins), f"the word guard did NOT catch: {line}"


def test_every_banned_word_is_a_sha256_digest():
    assert len(BANNED_WORD_DIGESTS) >= 26
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
