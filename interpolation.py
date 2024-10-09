import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

# Function to perform linear and cubic interpolation with Gaussian and Uniform noise
def interpolate_with_noise(data, noise_power=0.19, num_new_points=100, method='linear'):
    # Create a new dataframe to hold the augmented data
    new_data = pd.DataFrame(columns=data.columns)
    
    for col in data.columns:
        # Interpolate each column separately
        x = np.arange(len(data))
        y = data[col]
        
        # Choose interpolation method
        if method == 'linear':
            f = interp1d(x, y, kind='linear', fill_value="extrapolate")
        elif method == 'cubic':
            f = CubicSpline(x, y, extrapolate=True)
        
        # Generate new x values (including both original and new interpolated points)
        new_x = np.linspace(0, len(data)-1, len(data) + num_new_points)
        
        # Apply interpolation
        interpolated_values = f(new_x)
        
        # Add Gaussian white noise to the interpolated values
        gaussian_noise = np.random.normal(0, noise_power * np.std(interpolated_values), len(interpolated_values))
        
        # Add uniform noise to the interpolated values
        uniform_noise = np.random.uniform(-0.1 * np.std(interpolated_values), 0.1 * np.std(interpolated_values), len(interpolated_values))
        
        # Combine the original values with both Gaussian and Uniform noise
        interpolated_values_with_noise = interpolated_values + gaussian_noise + uniform_noise
        
        # Store the augmented values with noise in the new dataframe
        new_data[col] = interpolated_values_with_noise
    
    return new_data

# Function to load, augment, and save the dataset with multiple augmentation techniques
def augment_and_save_data(input_file, output_file, noise_power=0.02, num_new_points=200):
    # Load the dataset
    data = pd.read_csv(input_file)
    
    # Apply interpolation with noise (both linear and cubic)
    interpolated_data_linear = interpolate_with_noise(data, noise_power=noise_power, num_new_points=num_new_points, method='linear')
    interpolated_data_cubic = interpolate_with_noise(data, noise_power=noise_power, num_new_points=num_new_points, method='cubic')
    
    # Combine both interpolated datasets
    combined_interpolated_data = pd.concat([interpolated_data_linear, interpolated_data_cubic], axis=0).reset_index(drop=True)
    
    # Apply further augmentations (scaling and flipping)
    #fully_augmented_data = further_augment_data(combined_interpolated_data)
    
    # Save the augmented dataset to a new CSV file
    combined_interpolated_data.to_csv(output_file, index=False)

# Example usage: load "data3.csv" and save augmented data to "augmented_data.csv"
augment_and_save_data('data2.csv', 'augmented_data2.csv')
