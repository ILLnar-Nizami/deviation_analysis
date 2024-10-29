# Contributing to Deviation Analysis Project

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/ILLnar-Nizami/deviation_analysis.git
cd deviation_analysis
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
make install
```

## Development Workflow

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```
2. Make your changes
3. Run tests:
```bash
make test
```
4. Run linting:
```bash
make lint
```
5. Run profiling (if needed):
```bash
make profile
```
# Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for functions and classes
- Keep functions focused and small