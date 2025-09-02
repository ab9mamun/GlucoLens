import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import shap
import dice_ml
from dice_ml import Dice
import random
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('dataset_all_features.csv')
print(f"Original data shape: {data.shape}")

data = data.dropna()
print(f"Data shape after dropping NaN: {data.shape}")

data = data.drop(columns=['absolute_auc','respective_auc','max_postprandial_gluc'])

X = data.drop('postprandial_hyperglycemia_140', axis=1).astype(float)
y = data['postprandial_hyperglycemia_140'].astype(int)

class HybridClassifier:
    """Custom ensemble classifier for SHAP compatibility"""
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.xgb = XGBClassifier(n_estimators=100, random_state=random_state,
                                use_label_encoder=False, eval_metric='logloss')
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(160, 80, 40, 40, 40, 40, 20, 10),
            max_iter=500,
            random_state=random_state
        )

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        self.mlp.fit(X, y)
        return self

    def predict_proba(self, X):
        rf_probs = self.rf.predict_proba(X)
        xgb_probs = self.xgb.predict_proba(X)
        mlp_probs = self.mlp.predict_proba(X)
        return (rf_probs + xgb_probs + mlp_probs) / 3

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def calculate_dice_metrics(dice_counterfactuals, original_instances, feature_names):
    """Calculate DiCE evaluation metrics with enhanced error handling"""
    metrics = {
        'validity': [],
        'diversity': [],
        'normalized_distance': [],
        'num_features_changed': []
    }

    for i, cf_examples in enumerate(dice_counterfactuals):
        try:
            if cf_examples is None:
                # No counterfactuals generated
                metrics['validity'].append(0)
                metrics['diversity'].append(0)
                metrics['normalized_distance'].append(0)
                metrics['num_features_changed'].append(0)
                continue

            # Extract counterfactuals from DiCE object
            cf_list = None

            # Method 1: Check for cf_examples_list attribute
            if hasattr(cf_examples, 'cf_examples_list'):
                cf_list = cf_examples.cf_examples_list
                print(f"Instance {i}: Found cf_examples_list with {len(cf_list)} items")

                # Each item in cf_list might also be a CounterfactualExamples object
                actual_cf_data = []
                for j, cf_item in enumerate(cf_list):
                    print(f"  CF item {j} type: {type(cf_item)}")

                    if hasattr(cf_item, 'cf_examples_list'):
                        # Nested CounterfactualExamples
                        nested_cfs = cf_item.cf_examples_list
                        actual_cf_data.extend(nested_cfs)
                        print(f"    Nested cf_examples_list with {len(nested_cfs)} items")
                    elif hasattr(cf_item, 'final_cfs_df'):
                        # DataFrame of counterfactuals
                        cf_df = cf_item.final_cfs_df
                        print(f"    Found final_cfs_df: {cf_df.shape}")

                        # Remove target column if present
                        target_col = 'postprandial_hyperglycemia_140'
                        if target_col in cf_df.columns:
                            cf_df = cf_df.drop(columns=[target_col])
                            print(f"    After removing target column: {cf_df.shape}")

                        actual_cf_data.append(cf_df)
                        print(f"    Added DataFrame with {len(cf_df)} counterfactuals")
                    elif hasattr(cf_item, 'cf_examples_df'):
                        # Alternative DataFrame attribute
                        actual_cf_data.append(cf_item.cf_examples_df)
                        print(f"    Found cf_examples_df: {cf_item.cf_examples_df.shape}")
                    elif isinstance(cf_item, pd.DataFrame):
                        actual_cf_data.append(cf_item)
                        print(f"    Direct DataFrame: {cf_item.shape}")
                    else:
                        # Try to extract data from the object
                        for attr in ['data', 'examples', 'cf_data', 'counterfactuals']:
                            if hasattr(cf_item, attr):
                                attr_value = getattr(cf_item, attr)
                                if isinstance(attr_value, pd.DataFrame):
                                    actual_cf_data.append(attr_value)
                                    print(f"    Found {attr}: {attr_value.shape}")
                                    break
                        else:
                            print(f"    Could not extract data from {type(cf_item)}")
                            # Try to convert to dict and then DataFrame
                            try:
                                if hasattr(cf_item, '__dict__'):
                                    item_dict = cf_item.__dict__
                                    print(f"    Object attributes: {list(item_dict.keys())}")
                            except:
                                pass

                cf_list = actual_cf_data

            # Method 2: Direct final_cfs_df access
            elif hasattr(cf_examples, 'final_cfs_df'):
                cf_list = [cf_examples.final_cfs_df]
                print(f"Instance {i}: Found direct final_cfs_df: {cf_examples.final_cfs_df.shape}")

            # Method 3: Direct cf_examples_df access
            elif hasattr(cf_examples, 'cf_examples_df'):
                cf_list = [cf_examples.cf_examples_df]
                print(f"Instance {i}: Found direct cf_examples_df: {cf_examples.cf_examples_df.shape}")

            # Method 4: Check if it's directly a DataFrame
            elif isinstance(cf_examples, pd.DataFrame):
                cf_list = [cf_examples]
                print(f"Instance {i}: Direct DataFrame: {cf_examples.shape}")

            else:
                print(f"Instance {i}: Exploring object attributes...")
                if hasattr(cf_examples, '__dict__'):
                    attrs = cf_examples.__dict__
                    print(f"  Available attributes: {list(attrs.keys())}")

                    # Look for DataFrame-like attributes
                    for attr_name, attr_value in attrs.items():
                        if isinstance(attr_value, pd.DataFrame):
                            cf_list = [attr_value]
                            print(f"  Using attribute '{attr_name}': {attr_value.shape}")
                            break

                if cf_list is None:
                    cf_list = []

            if not cf_list or len(cf_list) == 0:
                # No valid counterfactuals found
                print(f"No valid counterfactuals found for instance {i}")
                metrics['validity'].append(0)
                metrics['diversity'].append(0)
                metrics['normalized_distance'].append(0)
                metrics['num_features_changed'].append(0)
                continue

            print(f"Processing {len(cf_list)} counterfactual DataFrames for instance {i}")
            original = original_instances.iloc[i:i+1]

            # Calculate feature ranges for normalization early (using all original instances)
            feature_ranges = np.ptp(original_instances.values, axis=0)  # peak-to-peak
            feature_ranges[feature_ranges == 0] = 1  # Avoid division by zero

            # Convert counterfactual DataFrames to numpy arrays
            cf_arrays = []
            for j, cf_df in enumerate(cf_list):
                try:
                    if isinstance(cf_df, pd.DataFrame):
                        print(f"  Processing CF DataFrame {j}: {cf_df.shape}")

                        # Remove the target column if present
                        target_col = 'postprandial_hyperglycemia_140'
                        if target_col in cf_df.columns:
                            cf_df = cf_df.drop(columns=[target_col])
                            print(f"    After removing target: {cf_df.shape}")

                        # Ensure we only use features that exist in original data
                        common_features = [col for col in cf_df.columns if col in original.columns]
                        if len(common_features) > 0:
                            cf_df = cf_df[common_features]
                            print(f"    Using {len(common_features)} common features")

                        cf_array = cf_df.values
                        print(f"    CF array shape: {cf_array.shape}")

                        # Handle multiple rows in counterfactual DataFrame
                        if cf_array.ndim == 2:
                            if cf_array.shape[0] > 1:
                                # Multiple counterfactuals in one DataFrame
                                print(f"    Extracting {cf_array.shape[0]} individual counterfactuals")
                                for row_idx in range(cf_array.shape[0]):
                                    cf_arrays.append(cf_array[row_idx])
                                    print(f"      Added counterfactual {len(cf_arrays)}: shape {cf_array[row_idx].shape}")
                            elif cf_array.shape[0] == 1:
                                # Single counterfactual
                                cf_arrays.append(cf_array.flatten())
                                print(f"      Added single counterfactual: shape {cf_array.flatten().shape}")
                        elif cf_array.ndim == 1:
                            # Already flattened
                            cf_arrays.append(cf_array)
                            print(f"      Added flattened counterfactual: shape {cf_array.shape}")
                        else:
                            print(f"    Unexpected array shape {cf_array.shape}, skipping")

                    elif isinstance(cf_df, np.ndarray):
                        cf_array = cf_df
                        print(f"  CF {j}: NumPy array with shape {cf_array.shape}")
                        if cf_array.ndim == 2 and cf_array.shape[0] > 1:
                            for row_idx in range(cf_array.shape[0]):
                                cf_arrays.append(cf_array[row_idx])
                        else:
                            cf_arrays.append(cf_array.flatten() if cf_array.ndim > 1 else cf_array)
                    else:
                        print(f"  CF {j}: Unexpected type {type(cf_df)}, skipping")
                        continue

                except Exception as e:
                    print(f"Error processing counterfactual {j} for instance {i}: {str(e)}")
                    continue

            print(f"Total counterfactuals extracted: {len(cf_arrays)}")

            if len(cf_arrays) == 0:
                # No valid counterfactuals could be processed
                print(f"No processable counterfactuals for instance {i}")
                metrics['validity'].append(0)
                metrics['diversity'].append(0)
                metrics['normalized_distance'].append(0)
                metrics['num_features_changed'].append(0)
                continue

            print(f"Successfully processed {len(cf_arrays)} counterfactual arrays for instance {i}")

            # Verify we have the expected number of counterfactuals
            if len(cf_arrays) < 3:
                print(f"Warning: Expected 3 counterfactuals, but only got {len(cf_arrays)} for instance {i}")

            # Validity (all processed counterfactuals are considered valid)
            validity = 1.0 if len(cf_arrays) > 0 else 0.0
            metrics['validity'].append(validity)

            # Ensure all arrays have the same length as original
            valid_cf_arrays = []
            original_array = original.values.flatten()
            print(f"Original array length: {len(original_array)}")

            for idx, cf_array in enumerate(cf_arrays):
                print(f"Checking CF {idx}: length {len(cf_array)} vs original {len(original_array)}")
                if len(cf_array) == len(original_array):
                    valid_cf_arrays.append(cf_array)
                    print(f"  ✓ Added CF {idx} to valid list")
                else:
                    print(f"  ✗ Shape mismatch: original {len(original_array)}, cf {len(cf_array)}")

            print(f"Valid counterfactuals: {len(valid_cf_arrays)} out of {len(cf_arrays)}")

            if len(valid_cf_arrays) == 0:
                # No counterfactuals with matching dimensions
                print(f"No valid counterfactuals with matching dimensions for instance {i}")
                metrics['diversity'].append(0)
                metrics['normalized_distance'].append(0)
                metrics['num_features_changed'].append(0)
                continue

            # Diversity (normalized average pairwise distance between counterfactuals)
            if len(valid_cf_arrays) > 1:
                distances = []
                for j in range(len(valid_cf_arrays)):
                    for k in range(j+1, len(valid_cf_arrays)):
                        raw_dist = np.abs(valid_cf_arrays[j] - valid_cf_arrays[k])
                        # Normalize distance by feature ranges
                        normalized_dist = np.sum(raw_dist / feature_ranges)
                        distances.append(normalized_dist)
                diversity = np.mean(distances) if distances else 0
            else:
                diversity = 0
            metrics['diversity'].append(diversity)

            # Normalized distance (average distance from original)
            distances_from_orig = []

            for cf_array in valid_cf_arrays:
                # Calculate normalized distance
                raw_diff = np.abs(cf_array - original_array)
                normalized_dist = np.sum(raw_diff / feature_ranges)
                distances_from_orig.append(normalized_dist)

            norm_dist = np.mean(distances_from_orig) if distances_from_orig else 0
            metrics['normalized_distance'].append(norm_dist)

            # Number of features changed
            features_changed_list = []
            for cf_array in valid_cf_arrays:
                # Use small threshold for floating point comparison
                changed = np.sum(np.abs(cf_array - original_array) > 1e-6)
                features_changed_list.append(changed)

            avg_features_changed = np.mean(features_changed_list) if features_changed_list else 0
            metrics['num_features_changed'].append(avg_features_changed)

        except Exception as e:
            print(f"Error processing counterfactuals for instance {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Add default values for failed instance
            metrics['validity'].append(0)
            metrics['diversity'].append(0)
            metrics['normalized_distance'].append(0)
            metrics['num_features_changed'].append(0)

    return metrics

def train_and_test_hybrid_with_explanations(X, y, repeat):
    """Enhanced version with SHAP and DiCE explanations"""

     
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

     
    results = {
        'iteration': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    shap_values_all = []
    dice_metrics_all = []

    for i in range(repeat):
        print(f"\n=== Iteration {i+1}/{repeat} ===")
        seed = 42 * i
        np.random.seed(seed)
        random.seed(seed)

        # Separate positive and negative examples
        positive_idx = y[y == 1].index
        negative_idx = y[y == 0].index

        # Randomly select 10 positives and 10 negatives for the test set
        pos_test_idx = np.random.choice(positive_idx, 10, replace=False)
        neg_test_idx = np.random.choice(negative_idx, 10, replace=False)
        test_idx = np.concatenate([pos_test_idx, neg_test_idx])

        # Create test and remaining datasets
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]
        X_remaining = X.drop(test_idx)
        y_remaining = y.drop(test_idx)

         
        adasyn = ADASYN(random_state=seed)
        X_train, y_train = adasyn.fit_resample(X_remaining, y_remaining)

         
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

         
        hybrid_model = HybridClassifier(random_state=seed)
        hybrid_model.fit(X_train, y_train)

        
        y_pred = hybrid_model.predict(X_test)

        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        print(f'Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')

        # Store results
        results['iteration'].append(i)
        results['accuracy'].append(accuracy)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)

        # SHAP Analysis (every 10th iteration to save time)
        if i % 10 == 0:
            print("Computing SHAP values...")
            try:
                # Use a sample for SHAP background (to speed up computation)
                background_sample = X_train.sample(min(100, len(X_train)), random_state=seed)
                explainer = shap.Explainer(hybrid_model.predict_proba, background_sample)

                # Compute SHAP values for test set
                shap_values = explainer(X_test)

                # Store SHAP values
                shap_values_all.append({
                    'iteration': i,
                    'shap_values': shap_values,
                    'test_data': X_test.copy(),
                    'predictions': y_pred,
                    'actual': y_test.values
                })

                # Create beeswarm plot (only for class 1 predictions to avoid multi-dimensional issue)
                plt.figure(figsize=(12, 8))
                # For binary classification, use shap_values for positive class only
                if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
                    # Multi-class case: use positive class (index 1)
                    shap_values_class1 = shap_values[..., 1]
                    shap.plots.beeswarm(shap_values_class1, max_display=15, show=False)
                else:
                    # Binary case or already single dimension
                    shap.plots.beeswarm(shap_values, max_display=15, show=False)
                plt.title(f'SHAP Beeswarm Plot - Iteration {i} (Positive Class)')
                plt.tight_layout()
                plt.savefig(f'shap_beeswarm_iteration_{i}.png', dpi=300, bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"SHAP analysis failed for iteration {i}: {str(e)}")

        # DiCE Analysis (every 20th iteration to save time)
        if i % 20 == 0:
            print("Computing DiCE counterfactuals...")
            try:
                # Prepare data for DiCE
                train_data = X_train.copy()
                train_data['postprandial_hyperglycemia_140'] = y_train

                dice_data = dice_ml.Data(dataframe=train_data,
                                       continuous_features=list(X_train.columns),
                                       outcome_name='postprandial_hyperglycemia_140')

                dice_model = dice_ml.Model(model=hybrid_model, backend="sklearn")
                dice_exp = Dice(dice_data, dice_model, method="random")

                counterfactuals_list = []
                for idx in range(min(5, len(X_test))):  # Limit to 5 instances to save time
                    try:
                        query_instance = X_test.iloc[idx:idx+1]
                        cf = dice_exp.generate_counterfactuals(
                            query_instance,
                            total_CFs=3,
                            desired_class="opposite"
                        )
                        if cf is not None:
                            counterfactuals_list.append(cf)
                        else:
                            print(f"No counterfactuals generated for instance {idx}")
                            counterfactuals_list.append(None)
                    except Exception as e:
                        print(f"Failed to generate counterfactual for instance {idx}: {str(e)}")
                        counterfactuals_list.append(None)

                # Calculate DiCE metrics
                dice_metrics = calculate_dice_metrics(
                    counterfactuals_list,
                    X_test.iloc[:len(counterfactuals_list)],
                    X.columns
                )

                dice_metrics_all.append({
                    'iteration': i,
                    'avg_validity': np.mean(dice_metrics['validity']),
                    'avg_diversity': np.mean(dice_metrics['diversity']),
                    'avg_normalized_distance': np.mean(dice_metrics['normalized_distance']),
                    'avg_num_features_changed': np.mean(dice_metrics['num_features_changed'])
                })

                print(f"DiCE metrics - Validity: {np.mean(dice_metrics['validity']):.3f}, "
                      f"Diversity: {np.mean(dice_metrics['diversity']):.3f}, "
                      f"Norm Distance: {np.mean(dice_metrics['normalized_distance']):.3f}, "
                      f"Features Changed: {np.mean(dice_metrics['num_features_changed']):.1f}")

            except Exception as e:
                print(f"DiCE analysis failed for iteration {i}: {str(e)}")

        # Save intermediate results every 25 iterations
        if (i + 1) % 25 == 0:
            results_df = pd.DataFrame(results)
            results_df.to_csv(f'ml_results_intermediate_{i+1}.csv', index=False)

    return results, shap_values_all, dice_metrics_all

print("Training and testing with Hybrid RF + XGBoost + MLP model with explanations")
results, shap_results, dice_results = train_and_test_hybrid_with_explanations(X, y, 50)

# Save final results
print("\nSaving results...")

# Main ML results
results_df = pd.DataFrame(results)
results_df.to_csv('ml_results_final.csv', index=False)

# SHAP results summary
if shap_results:
    shap_summary = []
    for shap_result in shap_results:
        # Calculate feature importance (mean absolute SHAP values)
        shap_values = shap_result['shap_values']
        if hasattr(shap_values, 'values'):
            if len(shap_values.values.shape) == 3:
                # Multi-class case: use positive class (index 1)
                mean_shap = np.mean(np.abs(shap_values.values[:, :, 1]), axis=0)
            else:
                # Binary case
                mean_shap = np.mean(np.abs(shap_values.values), axis=0)
        else:
            # Fallback
            mean_shap = np.mean(np.abs(shap_values), axis=0) if isinstance(shap_values, np.ndarray) else []

        for j, feature in enumerate(X.columns):
            shap_summary.append({
                'iteration': shap_result['iteration'],
                'feature': feature,
                'mean_abs_shap': mean_shap[j] if j < len(mean_shap) else 0
            })

    shap_df = pd.DataFrame(shap_summary)
    shap_df.to_csv('shap_feature_importance.csv', index=False)

# DiCE results
if dice_results:
    dice_df = pd.DataFrame(dice_results)
    dice_df.to_csv('dice_counterfactual_metrics.csv', index=False)

# Overall summary
print(f"\n=== FINAL RESULTS SUMMARY ===")
print(f"Average Metrics over {len(results['accuracy'])} runs:")
print(f"Accuracy: {np.mean(results['accuracy']):.4f} ± {np.std(results['accuracy']):.4f}")
print(f"Precision: {np.mean(results['precision']):.4f} ± {np.std(results['precision']):.4f}")
print(f"Recall: {np.mean(results['recall']):.4f} ± {np.std(results['recall']):.4f}")
print(f"F1 Score: {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")

if dice_results:
    print(f"\nDiCE Counterfactual Metrics (averaged across iterations):")
    print(f"Average Validity: {np.mean([d['avg_validity'] for d in dice_results]):.4f}")
    print(f"Average Diversity: {np.mean([d['avg_diversity'] for d in dice_results]):.4f}")
    print(f"Average Normalized Distance: {np.mean([d['avg_normalized_distance'] for d in dice_results]):.4f}")
    print(f"Average Features Changed: {np.mean([d['avg_num_features_changed'] for d in dice_results]):.1f}")

# Create overall SHAP summary plot
if shap_results:
    print("\nCreating overall SHAP feature importance plot...")
    shap_df = pd.read_csv('shap_feature_importance.csv')
    feature_importance = shap_df.groupby('feature')['mean_abs_shap'].mean().sort_values(ascending=True)

    plt.figure(figsize=(10, 8))
    feature_importance.tail(15).plot(kind='barh')
    plt.title('Top 15 Features by Mean Absolute SHAP Value')
    plt.xlabel('Mean |SHAP Value|')
    plt.tight_layout()
    plt.savefig('overall_shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\nAll results saved to CSV files!")
print("Files created:")
print("- ml_results_final.csv: Main ML performance metrics")
print("- shap_feature_importance.csv: SHAP feature importance scores")
print("- dice_counterfactual_metrics.csv: DiCE counterfactual evaluation metrics")
print("- shap_beeswarm_iteration_*.png: SHAP beeswarm plots")
print("- overall_shap_feature_importance.png: Overall feature importance plot")

