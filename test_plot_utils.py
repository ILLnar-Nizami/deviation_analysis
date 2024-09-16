import pytest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from plot_utils import PlotUtils
import os


@pytest.fixture
def sample_data():
    # Create a sample DataFrame for testing
    return pd.DataFrame({
        'gt_corners': [4, 4, 6, 8],
        'rb_corners': [4, 5, 6, 7],
        'mean': [1.5, 2.0, 1.8, 2.2],
        'max': [3.0, 3.5, 3.2, 3.8],
        'min': [0.5, 0.8, 0.6, 0.9]
    })
    

@patch('plot_utils.requests.get')
@patch('plot_utils.pd.read_json')
@patch('plot_utils.plt.savefig')
def test_draw_plots(mock_savefig, mock_read_json, mock_get, sample_data):
    # Set up mocks
    mock_response = MagicMock()
    mock_response.text = 'sample_json_data'
    mock_get.return_value = mock_response
    mock_read_json.return_value = sample_data

    # Call the function under test
    plot_paths = PlotUtils.draw_plots()

    # Assertions
    assert len(plot_paths) == 3
    assert all(path.startswith('plots/') for path in plot_paths)
    assert mock_get.called
    assert mock_read_json.called
    assert mock_savefig.call_count == 3


def test_draw_plots_creates_folder():
    # Test if the 'plots' folder is created
    with patch('plot_utils.os.makedirs') as mock_makedirs:
        PlotUtils.draw_plots()
        mock_makedirs.assert_called_once_with('plots', exist_ok=True)


@patch('plot_utils.requests.get')
@patch('plot_utils.pd.read_json')
@patch('plot_utils.plt.savefig')
def test_draw_plots_custom_url(mock_savefig, mock_read_json, mock_get, sample_data):
    # Test the function with a custom URL
    custom_url = "https://example.com/custom.json"
    mock_response = MagicMock()
    mock_response.text = 'custom_json_data'
    mock_get.return_value = mock_response
    mock_read_json.return_value = sample_data

    PlotUtils.draw_plots(json_url=custom_url)

    mock_get.assert_called_once_with(custom_url)


@patch('plot_utils.requests.get')
@patch('plot_utils.pd.read_json')
@patch('builtins.print')
def test_draw_plots_statistics(mock_print, mock_read_json, mock_get, sample_data):
    # Test if the correct statistics are printed
    mock_response = MagicMock()
    mock_response.text = 'sample_json_data'
    mock_get.return_value = mock_response
    mock_read_json.return_value = sample_data

    PlotUtils.draw_plots()

    mock_print.assert_any_call("Corner prediction accuracy: 50.00%")
    mock_print.assert_any_call("Average mean deviation: 1.88 degrees")
    mock_print.assert_any_call("Maximum deviation: 3.80 degrees")
