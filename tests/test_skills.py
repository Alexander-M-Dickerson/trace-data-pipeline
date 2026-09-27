"""The assistant skills: two identical copies, valid for every tool, pointing at real files.

    python -m pytest tests/test_skills.py -q

Claude Code reads skills from .claude/skills/; Codex, Cursor, Copilot and Gemini read
.agents/skills/. The two copies must be the same file, or one tool follows a procedure the
other has already corrected.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLAUDE, SHARED = ROOT / ".claude" / "skills", ROOT / ".agents" / "skills"
SKILLS = sorted(p.name for p in SHARED.iterdir() if p.is_dir())
# The fields of the Agent Skills specification (agentskills.io). A field outside it works in
# one tool and is ignored or refused by another.
SPEC_FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
STAGE_DIRS = ("", "stage0", "stage1", "stage2", "stage3", "stage4", "tests")


def frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert m, "no frontmatter"
    return dict(line.split(": ", 1) for line in m.group(1).splitlines())


def test_there_are_skills_and_each_tool_gets_the_same_set():
    assert SKILLS
    assert sorted(p.name for p in CLAUDE.iterdir() if p.is_dir()) == SKILLS


@pytest.mark.parametrize("name", SKILLS)
def test_the_two_copies_are_the_same_file(name):
    assert (CLAUDE / name / "SKILL.md").read_bytes() == (SHARED / name / "SKILL.md").read_bytes()


@pytest.mark.parametrize("name", SKILLS)
def test_the_frontmatter_is_the_specs(name):
    fm = frontmatter((SHARED / name / "SKILL.md").read_text(encoding="utf-8"))
    assert set(fm) <= SPEC_FIELDS, set(fm) - SPEC_FIELDS
    assert fm["name"] == name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) and len(name) <= 64
    assert 0 < len(fm["description"]) <= 1024


@pytest.mark.parametrize("name", SKILLS)
def test_every_file_a_skill_names_exists(name):
    text = (SHARED / name / "SKILL.md").read_text(encoding="utf-8")
    named = set(re.findall(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|sh|md|json|txt))`", text))
    missing = [f for f in named if not any((ROOT / d / f).exists() for d in STAGE_DIRS)]
    assert not missing, missing


def test_agents_md_and_claude_md_list_exactly_these_skills():
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    table = re.findall(r"^\| `([a-z-]+)` \|", agents, re.M)
    assert sorted(table) == SKILLS
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert sorted(re.findall(r"`/([a-z-]+)`", claude)) == SKILLS


def test_every_command_settings_json_allows_names_a_real_script():
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    for rule in settings["permissions"]["allow"]:
        m = re.match(r"Bash\((?:python3?|bash) ([\w./-]+\.(?:py|sh))", rule)
        if m:
            assert any((ROOT / d / m.group(1)).exists() for d in STAGE_DIRS), rule
