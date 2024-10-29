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
        'min': [0.5, 0.8, 0.7, 0.9],
        'floor_mean': [1.2, 1.8, 1.5, 2.0],
        'ceiling_mean': [1.8, 2.2, 2.1, 2.4],
        'name': ['Bedroom', 'Kitchen', 'Bath', 'Living']
    })


@patch('matplotlib.pyplot.savefig')
@patch('src.data_loader.load_data')
def test_draw_plots(mock_load_data, mock_savefig, sample_data, tmp_path):
    mock_load_data.return_value = sample_data
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    paths = PlotUtils.draw_plots(data=sample_data, plots_dir=str(plots_dir))
    assert mock_savefig.call_count >= 4
    assert len(paths) == 4


def test_draw_plots_with_empty_data():
    with pytest.raises(ValueError, match="Empty DataFrame"):
        PlotUtils.draw_plots(data=pd.DataFrame(), plots_dir="plots")


def test_draw_plots_with_missing_columns():
    incomplete_data = pd.DataFrame({
        'gt_corners': [4, 4, 6, 8],
        'rb_corners': [4, 5, 6, 7],
    })
    with pytest.raises(KeyError):
        PlotUtils.draw_plots(data=incomplete_data, plots_dir="plots")


def test_draw_plots_with_invalid_data_types():
    invalid_data = pd.DataFrame({
        'gt_corners': [4, 4, 'invalid', 8],  # Invalid type
        'rb_corners': [4, 5, 6, 7],
        'mean': [1.5, 2.0, 1.8, 2.2],
        'max': [3.0, 3.5, 3.2, 3.8],
        'min': [0.5, 0.8, 0.7, 0.9],
        'floor_mean': [1.2, 1.8, 1.5, 2.0],
        'ceiling_mean': [1.8, 2.2, 2.1, 2.4],
        'name': ['Bedroom', 'Kitchen', 'Bath', 'Living']
    })
    with pytest.raises(ValueError, match="Invalid data type for corner values"):
        PlotUtils.draw_plots(data=invalid_data, plots_dir="plots")


def test_create_scatter_plot(sample_data, tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    path = PlotUtils._create_scatter_plot(sample_data, str(plots_dir))
    assert path == str(plots_dir / "corners_comparison.png")
    assert os.path.exists(path)


@patch('matplotlib.pyplot.savefig')
def test_draw_plots_cleanup_on_error(mock_savefig, sample_data, tmp_path):
    mock_savefig.side_effect = Exception("Save failed")
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    with pytest.raises(Exception):
        PlotUtils.draw_plots(data=sample_data, plots_dir=str(plots_dir))
    assert not any(plots_dir.iterdir())
