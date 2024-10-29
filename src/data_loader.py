import requests
import pandas as pd
from io import StringIO
from .utils.profiling import profile, profile_memory


@profile
@profile_memory
def load_data(json_url):
    """Load data from a JSON URL."""
    response = requests.get(json_url)
    return pd.read_json(StringIO(response.text))
