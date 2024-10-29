# Deviation Analysis Project

Analysis tool for comparing floor and ceiling corner deviations in room measurements.

## Features

- Load and analyze deviation data from JSON
- Generate visualization plots:
  - Scatter plot: Ground truth vs predicted corners
  - Box plot: Distribution of deviation values
  - Histogram: Mean deviations
  - Floor vs ceiling comparison
- Statistical analysis:
  - Corner prediction accuracy
  - Mean and maximum deviations
  - Floor vs ceiling comparison
  - Room-wise analysis
- Performance profiling tools
- Comprehensive test suite

## Project Structure
```
deviation_analysis/
├── docs/ # Documentation
│ ├── README.md # Main documentation
│ ├── CONTRIBUTING.md # Contribution guidelines
│ └── CHANGELOG.md # Version history
├── src/ # Source code
│ ├── data_loader.py # Data loading utilities
│ ├── plot_utils.py # Plotting functions
│ └── utils/
│   └── profiling.py # Profiling utilities
├── tests/ # Test suite
│ └── __init__.py
│ └── conftest.py
│ └── test_data_loader.py
│ └── test_error_handling.py
│ └── test_main_functionality.py
│ └── test_plot_creation.py
│ └── test_plot_utils_extended.py
│ └── test_profiling.py
│ └── tests/test_statistics.py
├── scripts/ # Utility scripts
│ └── analyze_profile.py
├── plots/ # Generated plots (gitignored)
├── Notebook.ipynb # Jupyter notebook for analysis
└── configuration files # Various config files
```
## Installation

1. Clone the repository:
```bash
git clone https://github.com/ILLnar-Nizami/deviation_analysis.git
cd deviation_analysis
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate 
# On Windows: 
venv\Scripts\activate
```
3. Install dependencies:
```bash
make install
```

## Usage

### Command Line

Run the analysis:
```bash
make run
```

### Python API

```python
from src.plot_utils import PlotUtils
Using default URL
plot_paths = PlotUtils.draw_plots()
Using custom URL
plot_paths = PlotUtils.draw_plots(json_url="https://your-custom-url.json")
```

### Jupyter Notebook

Open and run `Notebook.ipynb` for interactive analysis:

```python
jupyter notebook Notebook.ipynb
```

## Development

### Running Tests

#### Run all tests
```bash
make test
```

#### Run with coverage
```bash
make coverage
```
#### Run specific test categories
```python
pytest tests/test_data_loader.py
pytest tests/test_error_handling.py
pytest tests/test_main_functionality.py
pytest tests/test_plot_creation.py
pytest tests/test_plot_utils_extended.py
pytest tests/test_profiling.py
pytest tests/tests/test_statistics.py
```

### Code Quality

#### Check code style
```bash
make lint
```
#### Format code
```bash
make format
```

### Profiling

#### Run profiling
```bash
make profile
```
#### Memory profiling
```bash
make profile-memory
```
#### Line-by-line profiling
```bash
make profile-line
```
#### View profile results
```bash
make view-profile
```
## Data Format

The input JSON should contain:
- `name`: Room name
- `gt_corners`: Ground truth corner count
- `rb_corners`: Predicted corner count
- `mean`, `max`, `min`: General deviation values
- `floor_mean`, `floor_max`, `floor_min`: Floor deviations
- `ceiling_mean`, `ceiling_max`, `ceiling_min`: Ceiling deviations

Example:
```json
json
{
"name": "Kitchen",
"gt_corners": 4,
"rb_corners": 4,
"mean": 1.5,
"max": 3.0,
"min": 0.5,
"floor_mean": 1.2,
"ceiling_mean": 1.8
}
```

## Performance Considerations

- Uses cProfile for performance profiling
- Memory usage monitored with memory_profiler
- Line-by-line profiling available for detailed analysis

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## Changelog

See [CHANGELOG.md](docs/CHANGELOG.md) for version history.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data provided by AI Process
- Built with Python, pandas, and matplotlib