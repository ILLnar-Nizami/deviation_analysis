import pytest
import pandas as pd
from src.plot_utils import PlotUtils


def test_empty_dataframe(tmp_path):
    """Test handling of empty DataFrame"""
    empty_df = pd.DataFrame()
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="Empty DataFrame"):
        PlotUtils.draw_plots(data=empty_df, plots_dir=str(plots_dir))


def test_missing_columns(tmp_path):
    """Test handling of DataFrame with missing columns"""
    incomplete_df = pd.DataFrame({
        'gt_corners': [4, 4],
        'rb_corners': [4, 5]
    })
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    with pytest.raises(KeyError, match="Required columns missing: .*"):
        PlotUtils.draw_plots(data=incomplete_df, plots_dir=str(plots_dir))


def test_invalid_data_types(tmp_path):
    """Test handling of invalid data types"""
    invalid_df = pd.DataFrame({
        'gt_corners': ['a', 'b'],
        'rb_corners': [4, 5],
        'mean': [1.5, 2.0],
        'max': [3.0, 3.5],
        'min': [0.5, 0.8],
        'floor_mean': [1.2, 1.8],
        'ceiling_mean': [1.8, 2.2],
        'name': ['Room1', 'Room2']
    })
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="Invalid data type for corner values"):
        PlotUtils.draw_plots(data=invalid_df, plots_dir=str(plots_dir))
