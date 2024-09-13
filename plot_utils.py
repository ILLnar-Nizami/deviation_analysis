import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import os
import requests
import pdb

class PlotUtils:
    @staticmethod
    def draw_plots(json_url=None):
        pdb.set_trace()  
        if json_url is None:
            json_url = "https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json"

        # Download and read JSON file
        response = requests.get(json_url)
        data = pd.read_json(StringIO(response.text))

        # Create plots folder if it doesn't exist
        os.makedirs('plots', exist_ok=True)

        plot_paths = []

        # Plot 1: Scatter plot of gt_corners vs rb_corners
        plt.figure(figsize=(10, 6))
        plt.scatter(data['gt_corners'], data['rb_corners'])
        plt.xlabel('Ground Truth Corners')
        plt.ylabel('Model Predicted Corners')
        plt.title('Ground Truth vs Model Predicted Corners')
        plt.savefig('plots/corners_comparison.png')
        plt.close()
        plot_paths.append('plots/corners_comparison.png')

        # Plot 2: Box plot of deviation values
        deviation_columns = ['mean', 'max', 'min']
        plt.figure(figsize=(10, 6))
        data[deviation_columns].boxplot()
        plt.ylabel('Deviation (degrees)')
        plt.title('Distribution of Deviation Values')
        plt.savefig('plots/deviation_distribution.png')
        plt.close()
        plot_paths.append('plots/deviation_distribution.png')

        # Plot 3: Histogram of mean deviation
        plt.figure(figsize=(10, 6))
        plt.hist(data['mean'], bins=20)
        plt.xlabel('Mean Deviation (degrees)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Mean Deviation')
        plt.savefig('plots/mean_deviation_histogram.png')
        plt.close()
        plot_paths.append('plots/mean_deviation_histogram.png')

        # Calculate and print some statistics
        corner_accuracy = (data['gt_corners'] == data['rb_corners']).mean()
        mean_deviation = data['mean'].mean()
        max_deviation = data['max'].max()

        print(f"Corner prediction accuracy: {corner_accuracy:.2%}")
        print(f"Average mean deviation: {mean_deviation:.2f} degrees")
        print(f"Maximum deviation: {max_deviation:.2f} degrees")

        return plot_paths
