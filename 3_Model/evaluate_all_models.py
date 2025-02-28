import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Function to calculate MAPE
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# Dictionary to store results
results = {}

# Product category names
category_names = {
    1: "Bread",
    2: "Rolls",
    3: "Croissant",
    4: "Confectionery",
    5: "Cake",
    6: "Seasonal Bread"
}

# Evaluate each model
for wg in range(1, 7):
    try:
        # Load the model
        model_path = f"umsatz_model_wg{wg}.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            
            # Load test data
            subdirectory = "pickle_data"
            test_features = pd.read_pickle(f"{subdirectory}/test_features.pkl")
            test_labels = pd.read_pickle(f"{subdirectory}/test_labels.pkl")
            
            # Make predictions
            test_predictions = model.predict(test_features)
            
            # Calculate MAPE
            mape_value = mape(test_labels, test_predictions)
            
            # Store result
            results[wg] = mape_value
            
            print(f"Category {wg} ({category_names[wg]}): MAPE = {mape_value:.2f}%")
        else:
            print(f"Model for category {wg} not found")
    except Exception as e:
        print(f"Error evaluating model for category {wg}: {e}")

# Print summary
print("\nResults Summary:")
print("----------------")
for wg in range(1, 7):
    if wg in results:
        print(f"{category_names[wg]} (WG {wg}): {results[wg]:.2f}%")
    else:
        print(f"{category_names[wg]} (WG {wg}): N/A")

# Update README.md with results
try:
    with open("../README.md", "r") as f:
        readme_content = f.read()
    
    # Find the Results Summary section and update it
    start_marker = "-   **Result by Category** (Identifier):"
    end_marker = "\n\n## Documentation"
    
    if start_marker in readme_content and end_marker in readme_content:
        start_idx = readme_content.find(start_marker) + len(start_marker)
        end_idx = readme_content.find(end_marker)
        
        # Create new results section
        new_results = "\n"
        for wg in range(1, 7):
            if wg in results:
                new_results += f"    -   **{category_names[wg]}** ({wg}): {results[wg]:.2f}%\n"
            else:
                new_results += f"    -   **{category_names[wg]}** ({wg}): N/A\n"
        
        # Replace the old results with new ones
        updated_readme = readme_content[:start_idx] + new_results + readme_content[end_idx:]
        
        # Also update the best model name
        best_model_marker = "-   **Best Model:** "
        if best_model_marker in updated_readme:
            best_model_idx = updated_readme.find(best_model_marker) + len(best_model_marker)
            end_best_model_idx = updated_readme.find("\n", best_model_idx)
            updated_readme = updated_readme[:best_model_idx] + "Neural Network with Optimized Architecture" + updated_readme[end_best_model_idx:]
        
        # Write updated README
        with open("../README.md", "w") as f:
            f.write(updated_readme)
        
        print("\nREADME.md has been updated with the results.")
    else:
        print("\nCouldn't find the Results Summary section in README.md")
except Exception as e:
    print(f"\nError updating README.md: {e}")
