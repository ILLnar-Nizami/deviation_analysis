import pytest
import pandas as pd
import os
import json
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt
from src.plot_utils import PlotUtils


@pytest.fixture
def complex_data():
    """Create more complex test data"""
    return pd.DataFrame({
        'gt_corners': [4, 4, 6, 8, 10],
        'rb_corners': [4, 5, 6, 7, 9],
        'mean': [1.5, 2.0, 1.8, 2.2, 2.5],
        'max': [3.0, 3.5, 3.2, 3.8, 4.0],
        'min': [0.5, 0.8, 0.7, 0.9, 1.0],
        'floor_mean': [1.2, 1.8, 1.5, 2.0, 2.3],
        'ceiling_mean': [1.8, 2.2, 2.1, 2.4, 2.7],
        'name': ['Bedroom', 'Kitchen', 'Bath', 'Living', 'Office']
    })


def test_plot_creation_with_custom_settings(complex_data, tmp_path):
    """Test plot creation with custom settings"""
    plots_dir = tmp_path / "custom_plots"
    plots_dir.mkdir(exist_ok=True)
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': [12, 8],
        'lines.linewidth': 2,
        'axes.grid': True
    })
    paths = PlotUtils.draw_plots(data=complex_data, plots_dir=str(plots_dir))
    assert len(paths) == 4
    for path in paths:
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_error_handling_during_plotting(complex_data, tmp_path):
    """Test error handling during plot creation"""
    plots_dir = tmp_path / "error_plots"
    plots_dir.mkdir(exist_ok=True)
    bad_data = complex_data.copy()
    bad_data.loc[0, 'mean'] = float('nan')
    with pytest.raises(ValueError,
                       match="Missing values detected in column: mean"):
        PlotUtils.draw_plots(data=bad_data, plots_dir=str(plots_dir))


def test_plot_with_missing_values(complex_data, tmp_path):
    """Test plotting with missing values"""
    plots_dir = tmp_path / "missing_plots"
    plots_dir.mkdir(exist_ok=True)
    data_with_nan = complex_data.copy()
    data_with_nan.loc[0, ['mean', 'max', 'min']] = pd.NA
    with pytest.raises(ValueError,
                       match="Missing values detected in column: mean"):
        PlotUtils.draw_plots(data=data_with_nan, plots_dir=str(plots_dir))


def test_invalid_numeric_values(complex_data, tmp_path):
    """Test handling of invalid numeric values"""
    plots_dir = tmp_path / "invalid_plots"
    plots_dir.mkdir(exist_ok=True)
    invalid_data = complex_data.copy()
    invalid_data.loc[0, 'mean'] = float('inf')
    with pytest.raises((ValueError, TypeError),
                       match=r"(Invalid|infinite) value[s]? detected"):
        PlotUtils.draw_plots(data=invalid_data, plots_dir=str(plots_dir))


def test_plot_with_custom_url(tmp_path):
    """Test plot creation with custom URL"""
    plots_dir = tmp_path / "url_plots"
    plots_dir.mkdir(exist_ok=True)
    test_json = {
        "gt_corners": [4, 4],
        "rb_corners": [4, 5],
        "mean": [1.5, 2.0],
        "max": [3.0, 3.5],
        "min": [0.5, 0.8],
        "floor_mean": [1.2, 1.8],
        "ceiling_mean": [1.8, 2.2],
        "name": ["Room1", "Room2"]
    }
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(test_json)
        mock_get.return_value = mock_response
        paths = PlotUtils.draw_plots(json_url='http://test.url',
                                     plots_dir=str(plots_dir))
        assert len(paths) == 4
        for path in paths:
            assert os.path.exists(path)


def test_plot_with_error_handling(complex_data, tmp_path):
    """Test error handling in plot creation"""
    plots_dir = tmp_path / "error_plots"
    plots_dir.mkdir(exist_ok=True)
    scenarios = [
        (lambda df: df.drop('mean', axis=1),
         KeyError, "Required columns missing"),
        (lambda df: df.assign(mean=float('inf')),
         ValueError, "Invalid values detected"),
        (lambda df: df.assign(gt_corners='invalid'),
         ValueError, "Invalid data type"),
    ]
    for modify_data, error_type, error_msg in scenarios:
        modified_data = modify_data(complex_data.copy())
        with pytest.raises(error_type, match=error_msg):
            PlotUtils.draw_plots(data=modified_data, plots_dir=str(plots_dir))


def test_plot_cleanup_on_error(complex_data, tmp_path):
    """Test cleanup of temporary files when error occurs"""
    plots_dir = tmp_path / "cleanup_plots"
    plots_dir.mkdir(exist_ok=True)
    bad_data = complex_data.copy()
    bad_data.loc[2:, 'mean'] = float('inf')
    try:
        PlotUtils.draw_plots(data=bad_data, plots_dir=str(plots_dir))
    except ValueError:
        assert len(os.listdir(plots_dir)) == 0


def test_plot_with_complex_error_handling(complex_data, tmp_path):
    """Test complex error handling scenarios in plot creation"""
    plots_dir = tmp_path / "complex_error_plots"
    plots_dir.mkdir(exist_ok=True)
    bad_data = complex_data.copy()
    bad_data.loc[:, 'mean'] = [float('inf'), float('-inf'),
                               float('nan'), 1.0, 2.0]

    with pytest.raises(ValueError) as exc_info:
        PlotUtils.draw_plots(data=bad_data, plots_dir=str(plots_dir))
    error_message = str(exc_info.value)
    assert any(msg in error_message for msg in [
        "Invalid values detected",
        "Missing values detected",
        "cannot handle missing values"
    ])
    assert len(os.listdir(plots_dir)) == 0
    with patch('requests.get') as mock_get:
        mock_get.side_effect = Exception("Connection error")
        with pytest.raises(Exception) as exc_info:
            PlotUtils.draw_plots(json_url='http://invalid.url')
        assert "during plot creation" in str(exc_info.value)


def test_plot_with_infinite_values(complex_data, tmp_path):
    """Test handling of infinite values in data"""
    plots_dir = tmp_path / "infinite_plots"
    plots_dir.mkdir(exist_ok=True)
    inf_data = complex_data.copy()
    inf_data.loc[0, 'mean'] = float('inf')
    inf_data.loc[1, 'max'] = float('-inf')
    with pytest.raises(ValueError) as exc_info:
        PlotUtils.draw_plots(data=inf_data, plots_dir=str(plots_dir))
    assert "Invalid values detected" in str(exc_info.value)
    assert len(os.listdir(plots_dir)) == 0


def test_plot_with_custom_plot_settings(complex_data, tmp_path):
    """Test plot creation with custom plot settings"""
    plots_dir = tmp_path / "custom_settings_plots"
    plots_dir.mkdir(exist_ok=True)
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': [10, 6],
        'axes.titlesize': 14,
        'axes.labelsize': 12
    })
    paths = PlotUtils.draw_plots(
        data=complex_data,
        plots_dir=str(plots_dir)
    )
    assert len(paths) == 4
    for path in paths:
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_plot_with_data_transformations(complex_data, tmp_path):
    """Test plot creation with data transformations"""
    plots_dir = tmp_path / "transform_plots"
    plots_dir.mkdir(exist_ok=True)
    extended_data = complex_data.copy()
    extended_data.loc[len(extended_data)] = {
        'gt_corners': 12,
        'rb_corners': 11,
        'mean': 3.0,
        'max': 4.5,
        'min': 1.5,
        'floor_mean': 2.8,
        'ceiling_mean': 3.2,
        'name': 'Extra Room'
    }
    paths = PlotUtils.draw_plots(data=extended_data, plots_dir=str(plots_dir))
    assert len(paths) == 4
    assert all(os.path.exists(p) for p in paths)


def test_plot_resource_cleanup(complex_data, tmp_path):
    """Test proper cleanup of resources during plotting"""
    plots_dir = tmp_path / "cleanup_test_plots"
    plots_dir.mkdir(exist_ok=True)
    initial_figures = len(plt.get_fignums())
    paths = PlotUtils.draw_plots(data=complex_data, plots_dir=str(plots_dir))
    assert len(plt.get_fignums()) == initial_figures
    assert len(paths) == 4
    assert all(os.path.exists(p) for p in paths)
    bad_data = complex_data.copy()
    bad_data.loc[0, 'mean'] = float('inf')
    try:
        PlotUtils.draw_plots(data=bad_data, plots_dir=str(plots_dir))
    except ValueError:
        assert len(plt.get_fignums()) == initial_figures
        assert len([f for f in os.listdir(plots_dir) if f.endswith('.png')]) == 4


def test_plot_with_invalid_data_types(complex_data, tmp_path):
    """Test plot creation with invalid data types"""
    plots_dir = tmp_path / "invalid_data_plots"
    plots_dir.mkdir(exist_ok=True)
    invalid_data = complex_data.copy()
    invalid_data.loc[0, 'mean'] = float('nan')
    with pytest.raises(ValueError,
                       match="Missing values detected in column: mean"):
        PlotUtils.draw_plots(data=invalid_data, plots_dir=str(plots_dir))


def test_plot_with_missing_columns(complex_data, tmp_path):
    """Test plot creation with missing columns"""
    plots_dir = tmp_path / "missing_columns_plots"
    plots_dir.mkdir(exist_ok=True)
    missing_column_data = complex_data.drop(columns=['mean'])
    with pytest.raises(KeyError, match="Required columns missing"):
        PlotUtils.draw_plots(data=missing_column_data, plots_dir=str(plots_dir))
