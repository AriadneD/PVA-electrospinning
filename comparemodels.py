import matplotlib.pyplot as plt
import pandas as pd

# Data for R² scores of Random Forest and Gradient Boosting for 3 datasets
data = {
    "Dataset 1": {
        "Random Forest": 0.9165,
        "Gradient Boosting": 0.9125
    },
    "Dataset 2": {
        "Random Forest": 0.9521,
        "Gradient Boosting": 0.9313
    },
    "Dataset 3": {
        "Random Forest": 0.9130,
        "Gradient Boosting": 0.8988
    }
}

# Convert the dictionary to a DataFrame for easier plotting
df = pd.DataFrame(data)

# Plotting the R² comparison
df.T.plot(kind='bar', figsize=(10, 6))
plt.title('R² Comparison: Random Forest vs Gradient Boosting')
plt.ylabel('R² Score')
plt.xlabel('Datasets')
plt.ylim(0.85, 1.0)
plt.legend(title='Models')
plt.xticks(rotation=0)
plt.grid(axis='y')

# Display the plot
plt.show()
