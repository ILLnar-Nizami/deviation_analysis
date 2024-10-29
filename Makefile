.PHONY: install test lint format clean profile profile-memory profile-line docs

# Development Setup
install:
	pip install -e ".[dev,notebook]"

# Testing
test:
	pytest

coverage:
	pytest --cov=src --cov-report=html

# Code Quality
lint:
	flake8 src tests
	black --check src tests

format:
	black src tests

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache
	rm -rf profile.stats
	rm -rf profile_results/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Profiling
profile:
	python -m cProfile -o profile.stats main.py
	python -m pstats profile.stats

profile-test:
	pytest --profile tests/

profile-memory:
	python -m memory_profiler main.py

profile-line:
	kernprof -l -v main.py

view-profile:
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(30)"

# Documentation
docs:
	pdoc --html --output-dir docs/api src/

# Run Analysis
run:
	python main.py

# Development Tools
setup-dev: install
	pre-commit install
	pip install -r requirements.txt

# Docker
docker-build:
	docker build -t deviation-analysis .

docker-run:
	docker run -it --rm deviation-analysis

# Help
help:
	@echo "Available commands:"
	@echo "  make install      - Install project dependencies"
	@echo "  make test        - Run tests"
	@echo "  make coverage    - Run tests with coverage report"
	@echo "  make lint        - Check code style"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean temporary files"
	@echo "  make profile     - Run profiling"
	@echo "  make docs        - Generate documentation"
	@echo "  make run         - Run analysis"
	@echo "  make setup-dev   - Setup development environment"