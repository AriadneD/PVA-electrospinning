import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

# Function to perform PCHIP interpolation with reduced noise and better fit
def interpolate_pchip(data, noise_power=0.05, num_new_points=100):
    # Create a new dataframe to hold the augmented data
    new_data = pd.DataFrame(columns=data.columns)
    
    for col in data.columns:
        # Interpolate each column separately
        x = np.arange(len(data))
        y = data[col]
        
        # Apply PCHIP interpolation
        f = PchipInterpolator(x, y, extrapolate=True)
        
        # Generate new x values (including both original and new interpolated points)
        new_x = np.linspace(0, len(data)-1, len(data) + num_new_points)
        
        # Apply interpolation
        interpolated_values = f(new_x)
        
        # Add less Gaussian white noise to the interpolated values
        gaussian_noise = np.random.normal(0, noise_power * np.std(interpolated_values), len(interpolated_values))
        
        # Add less uniform noise to the interpolated values
        # uniform_noise = np.random.uniform(-0.05 * np.std(interpolated_values), 0.05 * np.std(interpolated_values), len(interpolated_values))
        
        # Combine the original values with less Gaussian and Uniform noise
        interpolated_values_with_less_noise = interpolated_values + gaussian_noise
        
        # Store the augmented values with less noise in the new dataframe
        new_data[col] = interpolated_values_with_less_noise
    
    return new_data

# Function to load, apply PCHIP interpolation, and save the dataset
def augment_and_save_pchip_data(input_file, output_file, noise_power=0.02, num_new_points=200):
    # Load the dataset
    data = pd.read_csv(input_file)
    
    # Apply PCHIP interpolation with less noise
    interpolated_data_pchip = interpolate_pchip(data, noise_power=noise_power, num_new_points=num_new_points)
    
    # Save the augmented dataset to a new CSV file
    interpolated_data_pchip.to_csv(output_file, index=False)

# Example usage: load "data1.csv" and save augmented data to "augmented_data_pchip.csv"
augment_and_save_pchip_data('data1.csv', 'augmented_data1.csv')
augment_and_save_pchip_data('data3.csv', 'augmented_data3.csv')
