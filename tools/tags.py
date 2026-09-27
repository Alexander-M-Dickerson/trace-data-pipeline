r"""tags.py -- check the code tags, and write TAGS.md, the index of them.

    python tools/tags.py            # check the tags, then rewrite TAGS.md
    python tools/tags.py --check    # check, and fail if TAGS.md is out of date

A tag names a place other code or a doc relies on: where a panel column is computed, where a
filter is applied, where a rule is enforced. A reference points to a tag, so a reader, or an AI
assistant, can go from a doc straight to the code. TAGS.md lists every tag.

The syntax is tagref's (https://github.com/stepchowfun/tagref), so `tagref check` reads this
repository too. This script applies the same rules with nothing to install, and adds this
repository's own. The five kinds, each written in square brackets with a colon:

    tag      one place                         a Python or shell comment, or a doc
    group    the same thing in several places  (both stage 0 cleaners, for example)
    ref      a pointer to a tag or a group
    file     a pointer to a file               from the repository root, or ./ ../ from here
    dir      a pointer to a folder             the same

For a real pair, see the column contract's check, tagged in stage2/lib/contract.py, and the
pointer to it in AGENTS.md.

The rules, from tagref:
  1. every ref names a tag or a group that exists;
  2. no two tags share a name, and no name is both a tag and a group;
  3. a group has at least two members;
  4. a file or dir pointer names one that exists.
And this repository's own:
  5. every name starts with a namespace in NAMESPACES;
  6. every tag and group carries a description, after it on the same line;
  7. every column of the monthly panel (`PANEL_COLUMNS` in stage2/lib/contract.py) has
     exactly one `col.` name, and no `col.` name is a column the panel lacks; the same for the
     daily panel (`daily.`, against the column tables of stage1/DATA_DICTIONARY.md);
  8. in a Python file, a tag or group sits in a `#` comment;
  9. TAGS.md is exactly what this script writes (with --check).

Tags go in comments only, never in a docstring or a string, so tagging a Python file leaves
its parsed code unchanged.
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = "TAGS.md"

# The order is the order of TAGS.md's sections.
NAMESPACES = {
    "entry": "The commands you run: each stage's entry point.",
    "rule": "A rule the code enforces. The tag sits on the check, or on the one setting that fixes the rule.",
    "trap": "A mistake that has caught people, at the line that guards against it.",
    "filter": "A cleaning filter or correction, applied in stage 0 or stage 1.",
    "daily": "Where a column of the stage 1 daily panel is computed.",
    "col": ("Where a column of the stage 2 monthly panel is computed. Of the 38 price-based "
            "signals (`MMN_TWINNED` in stage2/lib/contract.py), the panel carries the "
            "gap-adjusted form, read at the signal trade or without the bond's last day of "
            "trading in the month, and the `_mmn` sidecar the unadjusted one."),
}

KINDS = ("tag", "group", "ref", "file", "dir")
# tagref's pattern, character for character: [^\]]*? then \s*\] trims the label's end.
_PATTERN = {k: re.compile(r"(?i)\[\s*" + k + r"\s*:\s*([^\]]*?)\s*\]") for k in KINDS}
# What may sit between a tag and its description: punctuation and spaces.
_LEAD = re.compile(r"^[\s\-:;,.=>|]*")


@dataclass(frozen=True)
class Directive:
    kind: str
    label: str
    path: str       # relative to ROOT, forward slashes
    line: int
    after: str      # the rest of the line after the directive
    in_comment: bool = True     # False: a Python file, outside a comment

    def where(self) -> str:
        return f"{self.path}:{self.line}"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def files_to_scan(root: Path = ROOT) -> list[str]:
    """What tagref scans: every file git does not ignore, tracked or not."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        # No git (a downloaded zip): skip the environments, and the data a run writes, which
        # .gitignore would have excluded. Tags live in text files far smaller than 5 MB.
        skip = {".git", ".venv", "venv", "__pycache__", "node_modules", "data", "output",
                "release", "reports", "logs", "smoke"}
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                      if p.is_file() and not skip & set(p.relative_to(root).parts)
                      and p.stat().st_size < 5_000_000)
    return sorted({p for p in out.decode("utf-8").split("\0") if p and (root / p).is_file()})


def _comment_starts(text: str) -> dict[int, int]:
    """Line -> the column a `#` comment starts at, for a Python file."""
    starts = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                starts[tok.start[0]] = tok.start[1]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return starts


def parse_text(text_lines, path: str) -> list[Directive]:
    found = []
    comments = None
    for n, line in enumerate(text_lines, 1):
        for kind, pat in _PATTERN.items():
            for m in pat.finditer(line):
                in_comment = True
                if path.endswith(".py") and kind in ("tag", "group"):
                    if comments is None:
                        comments = _comment_starts("\n".join(text_lines) + "\n")
                    in_comment = n in comments and comments[n] <= m.start()
                found.append(Directive(kind, m.group(1), path, n, line[m.end():], in_comment))
    return found


def read_directives(root: Path = ROOT, paths=None) -> list[Directive]:
    found = []
    for rel in (files_to_scan(root) if paths is None else paths):
        raw = (root / rel).read_bytes()
        lines = []
        for chunk in raw.split(b"\n"):
            try:        # tagref skips a line that is not UTF-8; so does this
                lines.append(chunk.rstrip(b"\r").decode("utf-8"))
            except UnicodeDecodeError:
                lines.append("")
        found += parse_text(lines, rel)
    return found


def panel_columns(root: Path = ROOT) -> tuple[str, ...]:
    """PANEL_COLUMNS from stage2/lib/contract.py, read without importing it."""
    tree = ast.parse((root / "stage2/lib/contract.py").read_text(encoding="utf-8"))
    for node in tree.body:
        target = getattr(node, "target", None) or (node.targets[0] if getattr(node, "targets", None) else None)
        if isinstance(target, ast.Name) and target.id == "PANEL_COLUMNS":
            return tuple(ast.literal_eval(node.value))
    raise SystemExit("stage2/lib/contract.py: PANEL_COLUMNS not found")


def daily_columns(root: Path = ROOT) -> tuple[str, ...]:
    """The columns of the stage 1 daily panel: the rows of the tables under "Variable
    Reference" in stage1/DATA_DICTIONARY.md whose header's first cell is "Column"."""
    names, in_ref, in_table = [], False, False
    for line in (root / "stage1/DATA_DICTIONARY.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_ref = line.strip() == "## Variable Reference"
            continue
        if not in_ref:
            continue
        if line.startswith("| Column |"):
            in_table = True
        elif not line.startswith("|"):
            in_table = False
        elif in_table and (m := re.match(r"\|\s*`([^`]+)`", line)):
            names.append(m.group(1))
    return tuple(dict.fromkeys(names))


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------
def _resolve(root: Path, source: str, target: str) -> Path:
    """tagref's path_util: ./ and ../ are relative to the file, anything else to the root."""
    if target.startswith(("./", "../", ".\\", "..\\")) or target in (".", ".."):
        return root / Path(source).parent / target
    return root / target


def problems(directives: list[Directive], root: Path = ROOT, *,
             panel=None, daily=None) -> list[str]:
    tags, groups = defaultdict(list), defaultdict(list)
    for d in directives:
        if d.kind == "tag":
            tags[d.label].append(d)
        elif d.kind == "group":
            groups[d.label].append(d)
    out = []
    # tagref's rules
    for label, ds in sorted(tags.items()):
        if len(ds) > 1:
            out.append(f"tag `{label}` is declared {len(ds)} times: "
                       + ", ".join(d.where() for d in ds)
                       + " (the same thing in several places is a group)")
    for label, ds in sorted(groups.items()):
        if len(ds) < 2:
            out.append(f"group `{label}` has one member, {ds[0].where()}: make it a tag")
        if label in tags:
            out.append(f"`{label}` is both a tag and a group")
    for d in directives:
        if d.kind == "ref" and d.label not in tags and d.label not in groups:
            out.append(f"{d.where()}: ref `{d.label}` names no tag or group")
        elif d.kind == "file" and not _resolve(root, d.path, d.label).is_file():
            out.append(f"{d.where()}: file `{d.label}` does not exist")
        elif d.kind == "dir" and not _resolve(root, d.path, d.label).is_dir():
            out.append(f"{d.where()}: dir `{d.label}` does not exist")
    # this repository's rules
    for d in directives:
        if d.kind not in ("tag", "group"):
            continue
        ns = d.label.split(".", 1)[0]
        if ns not in NAMESPACES or "." not in d.label:
            out.append(f"{d.where()}: `{d.label}` has no known namespace "
                       f"({', '.join(n + '.' for n in NAMESPACES)})")
        if not description(d):
            out.append(f"{d.where()}: `{d.label}` has no description after it on its line")
        if not d.in_comment:
            out.append(f"{d.where()}: `{d.label}` is outside a `#` comment; a tag in a string or "
                       "docstring changes the code it describes")
    for ns, want in (("col", panel_columns(root) if panel is None else panel),
                     ("daily", daily_columns(root) if daily is None else daily)):
        have = defaultdict(list)
        for label in list(tags) + list(groups):
            if label.startswith(ns + "."):
                have[label[len(ns) + 1:]].append(label)
        missing = [c for c in want if c not in have]
        extra = sorted(set(have) - set(want))
        if missing:
            out.append(f"{len(missing)} {ns} column(s) have no tag: {', '.join(missing)}")
        if extra:
            out.append(f"{ns} tag(s) for column(s) the panel does not have: {', '.join(extra)}")
    return out


def description(d: Directive) -> str:
    """The text after the directive, minus leading punctuation and any other directive."""
    rest = d.after
    for pat in _PATTERN.values():
        rest = pat.sub("", rest)
    rest = _LEAD.sub("", rest).strip()
    return rest if re.search(r"[A-Za-z]{2}", rest) else ""


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------
def render(directives: list[Directive]) -> str:
    targets = defaultdict(list)        # label -> the tag or the group members
    refs = defaultdict(set)
    for d in directives:
        if d.kind in ("tag", "group"):
            targets[d.label].append(d)
        elif d.kind == "ref":
            refs[d.label].add(d.path)
    ref_dirs = [d for d in directives if d.kind == "ref"]
    lines = [
        "# Code tags",
        "",
        "**Generated by `tools/tags.py`. Do not edit by hand** -- `python tools/tags.py --check` fails",
        "if this file and the code disagree.",
        "",
        "A tag marks a place in the code that other code or a doc relies on. To go to one, search the",
        "file for the tag's name with `tag:` or `group:` in front of it. A group is the same thing done",
        "in several places, which must be kept in step. The syntax is",
        "[tagref's](https://github.com/stepchowfun/tagref); `tools/tags.py` explains it and checks it.",
        "",
        f"{len(targets)} names in {sum(len(v) for v in targets.values())} places, and "
        f"{len(ref_dirs)} references to them from {len({d.path for d in ref_dirs})} files.",
        "",
    ]
    for ns, what in NAMESPACES.items():
        names = sorted(label for label in targets if label.split(".", 1)[0] == ns)
        if not names:
            continue
        lines += [f"## {ns}", "", what, "", "| name | where | what | pointed to from |", "|---|---|---|---|"]
        for label in names:
            ds = sorted(targets[label], key=lambda d: (d.path, d.line))
            where = "<br>".join(f"[{f}]({f})" for f in dict.fromkeys(d.path for d in ds))
            what_ = description(ds[0]).replace("|", "\\|")
            from_ = ", ".join(f"`{p}`" for p in sorted(refs.get(label, ())))
            lines.append(f"| `{label}` | {where} | {what_} | {from_} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="fail if a tag is broken or TAGS.md is out of date; write nothing")
    args = ap.parse_args(argv)

    directives = read_directives()
    bad = problems(directives)
    for p in bad:
        print("  " + p)
    text = render(directives)
    path = ROOT / INDEX
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if args.check:
        if current != text:
            print(f"  {INDEX} is out of date: run `python tools/tags.py`")
            bad.append("stale index")
    elif not bad and current != text:
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {INDEX}")
    n = sum(d.kind in ("tag", "group") for d in directives)
    print(f"{n} tags and group members, {sum(d.kind == 'ref' for d in directives)} refs: "
          + ("FAILED" if bad else "ok"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
