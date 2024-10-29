import pytest
import requests
import json
from unittest.mock import Mock, patch
from src.data_loader import load_data


def test_load_data_success():
    """Test successful data loading"""
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
        data = load_data('http://test.url')
        assert not data.empty
        assert all(col in data.columns for col in ['gt_corners', 'rb_corners', 'mean'])


def test_load_data_failure():
    """Test data loading failure"""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        with pytest.raises(Exception):
            load_data('http://test.url')
