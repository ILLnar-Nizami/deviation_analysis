import pytest
import pandas as pd
from unittest.mock import patch
from src.plot_utils import PlotUtils
import os


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'gt_corners': [4, 4, 6, 8],
        'rb_corners': [4, 5, 6, 7],
        'mean': [1.5, 2.0, 1.8, 2.2],
        'max': [3.0, 3.5, 3.2, 3.8],
        'min': [0.5, 0.8, 0.6, 0.9],
        'floor_mean': [1.2, 1.8, 1.5, 2.0],
        'ceiling_mean': [1.8, 2.2, 2.1, 2.4],
        'name': ['Bathroom', 'Kitchen', 'Bedroom', 'Living']
    })


@patch('matplotlib.pyplot.savefig')
def test_create_scatter_plot(mock_savefig, sample_data, tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    path = PlotUtils._create_scatter_plot(sample_data, str(plots_dir))
    assert path == str(plots_dir / "corners_comparison.png")
    assert mock_savefig.called


@patch('matplotlib.pyplot.savefig')
def test_create_box_plot(mock_savefig, sample_data, tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    path = PlotUtils._create_box_plot(sample_data, str(plots_dir))
    assert path == str(plots_dir / "deviation_distribution.png")
    assert mock_savefig.called


@patch('matplotlib.pyplot.savefig')
def test_create_histogram(mock_savefig, sample_data, tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    path = PlotUtils._create_histogram(sample_data, str(plots_dir))
    assert path == str(plots_dir / "mean_deviation_histogram.png")
    assert mock_savefig.called


@patch('matplotlib.pyplot.savefig')
def test_create_floor_ceiling_comparison(mock_savefig, sample_data, tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    path = PlotUtils._create_floor_ceiling_comparison(sample_data, str(plots_dir))
    assert path == str(plots_dir / "floor_ceiling_comparison.png")
    assert mock_savefig.called


@patch('matplotlib.pyplot.savefig')
def test_plot_creation(mock_savefig, sample_data, tmp_path):
    """Test plot file creation"""
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    paths = PlotUtils.draw_plots(data=sample_data, plots_dir=str(plots_dir))
    assert len(paths) > 0

    assert mock_savefig.call_count >= 4

    expected_files = {
        'corners_comparison.png',
        'deviation_distribution.png',
        'mean_deviation_histogram.png',
        'floor_ceiling_comparison.png'
    }

    actual_files = {os.path.basename(path) for path in paths}
    assert expected_files == actual_files, f"Missing files: {expected_files - actual_files}"
