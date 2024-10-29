import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from .data_loader import load_data
from .utils.profiling import profile, profile_memory, profile_line
import matplotlib

matplotlib.use("Agg")


class PlotUtils:
    @staticmethod
    @profile
    def draw_plots(data=None, plots_dir="plots", json_url=None):
        """Draw all plots and return list of file paths."""
        print("Starting to draw plots...")
        try:
            if json_url:
                data = load_data(json_url)

            if data is None or data.empty:
                raise ValueError("Empty DataFrame")
            PlotUtils.validate_data(data)
            os.makedirs(plots_dir, exist_ok=True)
            paths = []
            try:
                paths.append(PlotUtils._create_scatter_plot(data, plots_dir))
                paths.append(PlotUtils._create_box_plot(data, plots_dir))
                paths.append(PlotUtils._create_histogram(data, plots_dir))
                paths.append(PlotUtils._create_floor_ceiling_comparison(data, plots_dir))
                print("All plots created successfully.")
                return paths
            except Exception as e:
                print(f"Error while creating plots: {e}")
                for path in paths:
                    if os.path.exists(path):
                        os.remove(path)
                raise
        except Exception as e:
            print(f"Error during plot creation: {str(e)}")
            raise type(e)(f"{str(e)} during plot creation") from e

    @staticmethod
    def validate_data(data):
        """Validate the input data."""
        required_columns = [
            "gt_corners",
            "rb_corners",
            "mean",
            "max",
            "min",
            "floor_mean",
            "ceiling_mean",
            "name",
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise KeyError(f"Required columns missing: {missing_columns}")

        try:
            data["gt_corners"] = pd.to_numeric(data["gt_corners"])
            data["rb_corners"] = pd.to_numeric(data["rb_corners"])
        except (ValueError, TypeError):
            raise ValueError("Invalid data type for corner values")

        numeric_columns = ["mean", "max", "min", "floor_mean", "ceiling_mean"]
        for col in numeric_columns:
            if data[col].dtype == "object":
                raise ValueError(f"Invalid data type for column: {col}")
            if data[col].isna().any():
                raise ValueError(f"Missing values detected in column: {col}")
            if not np.isfinite(data[col]).all():
                raise ValueError(f"Invalid values detected in column: {col}")

    @staticmethod
    def create_all_plots(data, plots_dir="plots"):
        """Create all plots and return their paths."""
        paths = []
        for plot_type, create_func in [
            ("scatter", PlotUtils._create_scatter_plot),
            ("box", PlotUtils._create_box_plot),
            ("histogram", PlotUtils._create_histogram),
            ("floor_ceiling", PlotUtils._create_floor_ceiling_comparison),
        ]:
            path = os.path.join(plots_dir, f"{plot_type}_plot.png")
            paths.append(create_func(data, path))
        return paths

    @staticmethod
    @profile_line
    def _create_scatter_plot(data, plots_dir="plots"):
        """Create a scatter plot and return the file path."""
        print("Creating scatter plot...")
        plt.figure(figsize=(10, 6))
        plt.scatter(data["gt_corners"], data["rb_corners"])
        plt.xlabel("Ground Truth Corners")
        plt.ylabel("Reconstructed Corners")
        plt.title("Scatter Plot of Corners")
        path = os.path.join(plots_dir, "corners_comparison.png")
        plt.savefig(path)
        plt.close()
        print(f"Scatter plot saved to {path}")
        return path

    @staticmethod
    @profile_line
    def _create_box_plot(data, plots_dir="plots"):
        """Create box plot and return the file path."""
        print("Creating box plot...")
        plt.figure(figsize=(10, 6))
        data[["mean", "max", "min"]].boxplot()
        plt.ylabel("Deviation (degrees)")
        plt.title("Distribution of Deviation Values")
        path = os.path.join(plots_dir, "deviation_distribution.png")
        plt.savefig(path)
        plt.close()
        print(f"Box plot saved to {path}")
        return path

    @staticmethod
    @profile_line
    def _create_histogram(data, plots_dir="plots"):
        """Create histogram with line profiling."""
        plt.figure(figsize=(10, 6))
        plt.hist(data["mean"], bins=20)
        plt.xlabel("Mean Deviation (degrees)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Mean Deviation")
        path = os.path.join(plots_dir, "mean_deviation_histogram.png")
        plt.savefig(path)
        plt.close()
        return path

    @staticmethod
    @profile_line
    def _create_floor_ceiling_comparison(data, plots_dir="plots"):
        """Create floor vs ceiling comparison plot with line profiling."""
        plt.figure(figsize=(12, 6))
        x = range(len(data))
        plt.plot(x, data["floor_mean"], label="Floor Mean", marker="o")
        plt.plot(x, data["ceiling_mean"], label="Ceiling Mean", marker="s")
        plt.xticks(x, data["name"], rotation=45)
        plt.ylabel("Mean Deviation (degrees)")
        plt.title("Floor vs Ceiling Mean Deviation by Room")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(plots_dir, "floor_ceiling_comparison.png")
        plt.savefig(path)
        plt.close()
        return path

    @staticmethod
    @profile_memory
    def _print_statistics(data):
        """Print basic statistics with memory profiling."""
        corner_accuracy = (data["gt_corners"] == data["rb_corners"]).mean()
        mean_deviation = data["mean"].mean()
        max_deviation = data["max"].max()

        print(f"Corner prediction accuracy: {corner_accuracy:.2%}")
        print(f"Average mean deviation: {mean_deviation:.2f} degrees")
        print(f"Maximum deviation: {max_deviation:.2f} degrees")

    @staticmethod
    @profile_memory
    def _print_extended_statistics(data):
        """Print extended statistics with memory profiling."""
        print("Basic Statistics:")
        corner_accuracy = (data["gt_corners"] == data["rb_corners"]).mean()
        mean_deviation = data["mean"].mean()
        max_deviation = data["max"].max()
        print(f"Corner prediction accuracy: {corner_accuracy:.2%}")
        print(f"Average mean deviation: {mean_deviation:.2f} degrees")
        print(f"Maximum deviation: {max_deviation:.2f} degrees")
        print("\nFloor vs Ceiling Analysis:")
        floor_mean_dev = data["floor_mean"].mean()
        ceiling_mean_dev = data["ceiling_mean"].mean()
        print(f"Average floor deviation: {floor_mean_dev:.2f} degrees")
        print(f"Average ceiling deviation: {ceiling_mean_dev:.2f} degrees")
        print("\nRoom-wise Analysis:")
        for room_type in data["name"].unique():
            room_data = data[data["name"] == room_type]
            print(f"\n{room_type}:")
            print(f"  Floor mean deviation: {room_data['floor_mean'].mean():.2f} degrees")
            print(f"  Ceiling mean deviation: {room_data['ceiling_mean'].mean():.2f} degrees")
            print(
                f"  Corner accuracy: {
                  (room_data['gt_corners'] == room_data['rb_corners']).mean():.2%}"
            )
