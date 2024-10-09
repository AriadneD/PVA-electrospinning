import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import plot_partial_dependence
from treeinterpreter import treeinterpreter as ti

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

# Function to perform LIME explanations and save the plot
def lime_explanation(X_train, X_test, model, feature_names, save_dir, instance_index=0):
    # Initialize LIME explainer
    explainer = LimeTabularExplainer(X_train.values, feature_names=feature_names, class_names=['diameter'], verbose=True, mode='regression')
    
    # Explain a single instance
    exp = explainer.explain_instance(X_test.iloc[instance_index].values, model.predict)
    
    # Save the explanation figure
    exp.as_pyplot_figure()
    plt.savefig(f"{save_dir}/LIME_Explanation.png")
    plt.close()

# Function to perform PDP (Partial Dependence Plot) and save the plot
def partial_dependence_plot(model, X_test, feature_indices, feature_names, save_dir):
    # Plot Partial Dependence Plot
    plot_partial_dependence(model, X_test, features=feature_indices, feature_names=feature_names, grid_resolution=50)
    plt.savefig(f"{save_dir}/Partial_Dependence_Plot.png")
    plt.close()

# Function to perform TreeInterpreter explanation
def tree_interpreter_explanation(model, X_test, feature_names, instance_index=0):
    # Explain a single prediction
    prediction, bias, contributions = ti.predict(model, X_test.iloc[[instance_index]])
    
    print(f"Prediction: {prediction}")
    print(f"Bias (average prediction): {bias}")
    print(f"Feature Contributions:")
    
    for name, contribution in zip(feature_names, contributions[0]):
        print(f"{name}: {contribution}")
        
    return contributions

# Function to add SHAP dependence plot with line of best fit and save the plot
def shap_dependence_with_fit(feature, shap_values, X_test, save_dir):
    # Create SHAP dependence plot
    shap.dependence_plot(feature, shap_values, X_test, show=False)
    
    # Get SHAP values and feature values for the specified feature
    shap_vals = shap_values[:, X_test.columns.get_loc(feature)]
    feature_vals = X_test[feature]
    
    # Plot line of best fit using seaborn's regplot
    sns.regplot(x=feature_vals, y=shap_vals, scatter=False, color='red', line_kws={"linewidth": 2})
    
    # Save the plot
    plt.savefig(f"{save_dir}/SHAP_Dependence_Plot_{feature}.png")
    plt.close()

# Define a function to train models and plot various diagnostics
def trainModel(input_file, features):
    # Create directory to save plots
    save_dir = create_plot_directory(input_file)
    
    # Load the dataset
    data = pd.read_csv(input_file)
    
    # Split the data into features (X) and target (y)
    X = data[features]
    y = data['diameter']
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Compute the variance of the target variable (diameter)
    target_variance = np.var(y, ddof=1)  # ddof=1 for sample variance
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree (max_depth=5)': DecisionTreeRegressor(max_depth=5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR()
    }
    
    # Dictionary to store performance metrics
    performance_metrics = {}
    
    # Train each model, make predictions, and evaluate performance
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        rmse_to_variance_ratio = rmse / np.sqrt(target_variance)
        
        performance_metrics[name] = [mae, rmse, r2, rmse_to_variance_ratio]
        
        # Feature importance (only for tree-based models)
        if name in ['Random Forest', 'Gradient Boosting']:
            importance = model.feature_importances_
            plt.barh(features, importance)
            plt.title(f'Feature Importance for {name}')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.savefig(f"{save_dir}/Feature_Importance_{name}.png")
            plt.close()
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig(f"{save_dir}/Correlation_Heatmap.png")
    plt.close()
    
    # 2a. Model Performance Comparison
    metrics_df = pd.DataFrame(performance_metrics, index=['MAE', 'RMSE', 'R²', 'RMSE to Variance Ratio']).T
    metrics_df[['MAE', 'RMSE']].plot(kind='bar', title='Model Performance Comparison', figsize=(12, 6))
    plt.savefig(f"{save_dir}/Model_Performance_Comparison.png")
    plt.close()
    
    # 3. Residuals Plot (for Random Forest)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    residuals = y_test - y_pred_rf
    plt.scatter(y_pred_rf, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals Plot (Random Forest)')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.savefig(f"{save_dir}/Residuals_Plot_Random_Forest.png")
    plt.close()
    
    # 4. Actual vs Predicted Plot
    plt.scatter(y_test, y_pred_rf)
    plt.title('Actual vs Predicted (Random Forest)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.savefig(f"{save_dir}/Actual_vs_Predicted.png")
    plt.close()
    
    # 5. Learning Curve for Random Forest
    train_sizes, train_scores, test_scores = learning_curve(RandomForestRegressor(), X_train, y_train, cv=5, n_jobs=-1)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score')
    plt.title('Learning Curve (Random Forest)')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f"{save_dir}/Learning_Curve_Random_Forest.png")
    plt.close()
    
    # 6. Error Distribution Plot
    sns.histplot(residuals, kde=True)
    plt.title('Error Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig(f"{save_dir}/Error_Distribution.png")
    plt.close()
    
    # 7. SHAP Summary Plot for Random Forest
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    # SHAP Summary Plot
    shap.summary_plot(shap_values, X_test)
    plt.savefig(f"{save_dir}/SHAP_Summary_Plot.png")
    plt.close()

    # SHAP Dependence Plot for "voltage" with line of best fit
    shap_dependence_with_fit("voltage", shap_values, X_test, save_dir)

    # SHAP Dependence Plot for "concentration" with line of best fit
    shap_dependence_with_fit("concentration", shap_values, X_test, save_dir)

    # SHAP Dependence Plot for "distance" with line of best fit
    shap_dependence_with_fit("distance", shap_values, X_test, save_dir)

    # SHAP Waterfall Plot (fix: convert to Explanation object)
    shap_values_instance = shap.Explanation(
        values=shap_values[0], 
        base_values=explainer.expected_value[0], 
        data=X_test.iloc[0], 
        feature_names=X_test.columns
    )
    shap.waterfall_plot(shap_values_instance)
    plt.savefig(f"{save_dir}/SHAP_Waterfall_Plot.png")
    plt.close()

    # LIME Explanation for the first instance in X_test
    print("\nLIME Explanation:")
    lime_explanation(X_train, X_test, rf, feature_names=X_test.columns.tolist(), save_dir=save_dir)
    
    # PDP for the first two features
    print("\nPartial Dependence Plot:")
    partial_dependence_plot(rf, X_test, [0, 1], feature_names=X_test.columns.tolist(), save_dir=save_dir)
    
    # Tree Interpreter Explanation for the first instance in X_test
    print("\nTreeInterpreter Explanation:")
    tree_interpreter_explanation(rf, X_test, feature_names=X_test.columns.tolist())
    
    return performance_metrics

# Example usage:
result1 = trainModel('augmented_data1.csv', ['voltage', 'concentration', 'rotationalSpeed', 'distance', 'flowRate'])
result2 = trainModel('augmented_data2.csv', ['voltage', 'concentration', 'feedRate', 'distance'])
result3 = trainModel('augmented_data3.csv', ['voltage', 'concentration','distance'])

print(result1)
print(result2)
print(result3)
