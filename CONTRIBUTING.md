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

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

## Testing

Before submitting a pull request:

1. **Test locally** if possible
2. **Test on WRDS Cloud** for production scenarios
3. **Check logs** for errors or warnings
4. **Verify outputs** match expected format
5. **Run on small sample** before full dataset

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
