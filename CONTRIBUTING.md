# Contributing to TRACE Data Pipeline

Thank you for your interest in contributing to the TRACE Data Pipeline! This document provides guidelines for contributing to the project.
The paper that underlies the data work is currently under a "Revise & Resubmit (R&R)" -- your contributions will directly benefit this research.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear, descriptive title
- Detailed steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (Python version, OS, WRDS setup)
- Relevant log files or error messages
- Sample code if applicable

### Suggesting Enhancements

We welcome suggestions for new features or improvements! Please create an issue with:
- A clear description of the enhancement
- The motivation/use case for the enhancement
- Any relevant examples or references
- Whether you're willing to implement it yourself

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear, descriptive commit messages
3. **Test thoroughly** - ensure your changes don't break existing functionality
4. **Update documentation** - reflect your changes in README files
5. **Submit a pull request** with a clear description of changes

## Development Setup

### Prerequisites
- Python 3.10 or higher
- Access to WRDS (for testing)
- Git

### Setup Steps

1. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/trace-data-pipeline.git
cd trace-data-pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies (no `--user` -- pip refuses it inside an active virtualenv):
```bash
python -m pip install -r requirements.txt
```

4. Create a branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

## Testing

The repo ships a test suite. Run it before submitting a pull request.

**No WRDS needed** -- seconds:

```bash
python3 tests/test_chunk_plan.py        # chunk-partition properties
python3 tests/test_chunk_scheduler.py   # ordering + failure handling
```

**Whole chain, WRDS needed** -- ~10 minutes. Runs the real Stage 0 and Stage 1 code on a
handful of CUSIP chunks and asserts 28 cross-stage invariants. Writes to `smoke/` and
never touches production output:

```bash
bash download_inputs.sh          # once, on a machine with internet
./run_smoke_test.sh              # locally
qsub run_smoke_test.sh           # on WRDS -- head nodes forbid heavy work
```

❗**For any change to Stage 0's scheduling, chunking or ordering, the bar is
byte-identical output.** Bank the parquet files from `smoke/stage0/` before your change,
re-run after, and compare with `sha256sum`. This is possible because Stage 0 sorts
canonically before export, and it is the only way to tell a real change from a reshuffle.
The audit tables must match too -- the data reports reconstruct each chunk's filter
sequence from row order.

Then:

1. **Check logs** for errors or warnings
2. **Verify outputs** match the published schema (21 columns from Stage 0, 44 from Stage 1)
3. **Run on a small sample** before the full dataset (`STAGE0_LIMIT_CHUNKS=5`)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Email: alexander.dickerson1@unsw.edu.au

## Recognition

Contributors will be acknowledged in:
- The project README
- Release notes for significant contributions
- The academic paper underlying this data

Thank you for helping improve the TRACE Data Pipeline!
