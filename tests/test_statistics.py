import pytest
import pandas as pd
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


def test_plot_creation(sample_data, tmp_path):
    """Test plot file creation"""
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    try:
        paths = PlotUtils.draw_plots(data=sample_data, plots_dir=str(plots_dir))
        assert len(paths) > 0

        expected_files = {
            'corners_comparison.png',
            'deviation_distribution.png',
            'mean_deviation_histogram.png',
            'floor_ceiling_comparison.png'
        }

        actual_files = {os.path.basename(path) for path in paths}
        assert expected_files == actual_files, f"Missing files: {expected_files - actual_files}"

    finally:
        import shutil
        if plots_dir.exists():
            shutil.rmtree(plots_dir)


def test_extended_statistics(sample_data):
    """Test extended statistics calculations"""
    floor_stats = sample_data['floor_mean'].agg(['mean', 'std', 'min', 'max'])
    ceiling_stats = sample_data['ceiling_mean'].agg(['mean', 'std', 'min', 'max'])
    assert round(floor_stats['mean'], 2) == 1.62
    assert round(ceiling_stats['mean'], 2) == 2.12
    room_stats = sample_data.groupby('name')['mean'].mean()
    assert len(room_stats) == 4
    assert 'Bedroom' in room_stats.index
    assert 'Living' in room_stats.index


def test_data_validation(sample_data):
    """Test data validation methods"""
    required_columns = ['gt_corners', 'rb_corners', 'mean', 'max', 'min',
                        'floor_mean', 'ceiling_mean', 'name']
    for col in required_columns:
        assert col in sample_data.columns
    assert sample_data['mean'].dtype in ['float64', 'float32']
    assert sample_data['name'].dtype == 'object'


@pytest.mark.parametrize("test_case", [
    {'gt': [4, 4], 'rb': [4, 4], 'expected_accuracy': 100.0},
    {'gt': [4, 4], 'rb': [3, 5], 'expected_accuracy': 0.0},
    {'gt': [4, 4, 4], 'rb': [4, 3, 4], 'expected_accuracy': 66.67}
])
def test_corner_accuracy_cases(test_case):
    """Test various corner prediction accuracy scenarios"""
    data = pd.DataFrame({
        'gt_corners': test_case['gt'],
        'rb_corners': test_case['rb']
    })
    correct_predictions = (data['gt_corners'] == data['rb_corners']).sum()
    accuracy = (correct_predictions / len(data)) * 100
    assert round(accuracy, 2) == round(test_case['expected_accuracy'], 2)
