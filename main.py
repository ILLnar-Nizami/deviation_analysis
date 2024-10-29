from src.plot_utils import PlotUtils
from src.data_loader import load_data
from src.utils.profiling import profile

@profile
def main():
    """Main function with profiling."""
    json_url = "https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json"
    data = load_data(json_url)
    plot_paths = PlotUtils.draw_plots(data=data)
    print(f"Generated plots: {plot_paths}")


if __name__ == "__main__":
    main()
