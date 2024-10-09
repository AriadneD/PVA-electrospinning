import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# Function to create a directory for saving plots
def create_plot_directory(file_name):
    # Extract the base file name without extension
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    
    # Create the directory name
    plot_dir = f'plots_{base_name}'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    return plot_dir

# Define a function to create the dot plot and save it to the folder
def dotPlot(original_file, interpolated_file, feature):
    # Create the directory for saving plots
    save_dir = create_plot_directory(interpolated_file)

    # Load the original and interpolated datasets
    original_data = pd.read_csv(original_file)
    interpolated_data = pd.read_csv(interpolated_file)
    
    # Extract the feature (x) and target (y: diameter) from both datasets
    x_original = original_data[feature]
    y_original = original_data['diameter']
    
    # Randomly select 20 points from the interpolated data
    random_indices = np.random.choice(interpolated_data.index, size=20, replace=False)
    x_interpolated = interpolated_data.loc[random_indices, feature]
    y_interpolated = interpolated_data.loc[random_indices, 'diameter']
    
    # Create a dot plot: original points in blue, interpolated points in red
    plt.scatter(x_original, y_original, color='blue', label='Original', alpha=0.7)
    plt.scatter(x_interpolated, y_interpolated, color='red', label='Interpolated', alpha=0.7)
    
    # Fit lines of best fit for both original and original + interpolated
    # For original data only
    x_original_reshaped = np.array(x_original).reshape(-1, 1)
    y_original_reshaped = np.array(y_original).reshape(-1, 1)
    model_original = LinearRegression()
    model_original.fit(x_original_reshaped, y_original_reshaped)
    y_pred_original = model_original.predict(x_original_reshaped)
    
    # For original + randomly selected interpolated data
    x_combined = np.concatenate([x_original, x_interpolated]).reshape(-1, 1)
    y_combined = np.concatenate([y_original, y_interpolated]).reshape(-1, 1)
    model_combined = LinearRegression()
    model_combined.fit(x_combined, y_combined)
    y_pred_combined = model_combined.predict(x_combined)
    
    # Plot the lines of best fit
    plt.plot(x_original, y_pred_original, color='blue', label='Best Fit (Original)', linestyle='--')
    plt.plot(np.concatenate([x_original, x_interpolated]), y_pred_combined, color='red', label='Best Fit (Original + Interpolated', linestyle='--')
    
    # Add labels and title
    plt.xlabel(feature.capitalize())
    plt.ylabel('Diameter')
    plt.title(f'Dot Plot: {feature.capitalize()} vs Diameter')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the specified directory
    plot_file_name = f"{save_dir}/dot_plot_{feature}.png"
    plt.savefig(plot_file_name)
    plt.close()

# Example usage:
dotPlot('data1.csv', 'augmented_data1.csv', 'concentration')
dotPlot('data1.csv', 'augmented_data1.csv', 'voltage')
dotPlot('data1.csv', 'augmented_data1.csv', 'distance')

dotPlot('data2.csv', 'augmented_data2.csv', 'concentration')
dotPlot('data2.csv', 'augmented_data2.csv', 'voltage')
dotPlot('data2.csv', 'augmented_data2.csv', 'distance')

dotPlot('data3.csv', 'augmented_data3.csv', 'concentration')
dotPlot('data3.csv', 'augmented_data3.csv', 'voltage')
dotPlot('data3.csv', 'augmented_data3.csv', 'distance')
