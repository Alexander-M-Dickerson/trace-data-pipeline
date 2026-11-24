# Instructions for GitHub Copilot

Read [AGENTS.md](../AGENTS.md) first. It is the one set of instructions every AI assistant
working in this repository follows, and it points to the rest.

This is a summary of it.

- The pipeline runs in five stages. Stages 0 and 1 run on the WRDS Cloud, and stages 2 to 4 on
  the user's own computer. `python doctor.py` says what is ready and what to run next.
- [INDEX.md](../INDEX.md) says which doc answers a question, [CODE_MAP.md](../CODE_MAP.md) what
  each code file does, and [TAGS.md](../TAGS.md) the line where each panel column, filter, rule
  and trap lives.
- A failing check is the answer, not an obstacle. Never weaken or skip one to make a run finish.
- Before changing code, run `python -m pytest stage2/tests tests stage3/tests stage4/tests -q`.
