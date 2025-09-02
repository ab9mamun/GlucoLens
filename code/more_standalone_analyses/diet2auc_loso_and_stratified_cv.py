# keep one import to main block active at any given point

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import LeaveOneGroupOut
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

 
data = pd.read_csv('dataset_all_features_user_id.csv')
print(f"Original data shape: {data.shape}")

data = data.dropna()
print(f"Data shape after removing NaN: {data.shape}")

data = data.drop(columns=['absolute_auc','respective_auc','max_postprandial_gluc'])

 
user_ids = data['user_id']
X = data.drop(['postprandial_hyperglycemia_140', 'user_id'], axis=1).astype(float)
y = data['postprandial_hyperglycemia_140'].astype(int)

print(f"Number of unique subjects: {len(user_ids.unique())}")
print(f"Class distribution: {y.value_counts().to_dict()}")

def train_hybrid_model(X_train, y_train, seed=42):
     
     
    adasyn = ADASYN(random_state=seed)
    X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)

     
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(X_train_balanced, y_train_balanced)

     
    xgb = XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train_balanced, y_train_balanced)

     
    mlp = MLPClassifier(
        hidden_layer_sizes=(160, 80, 40, 40, 40, 40, 20, 10),
        max_iter=500,
        random_state=seed
    )
    mlp.fit(X_train_balanced, y_train_balanced)

    return rf, xgb, mlp

def predict_hybrid(rf, xgb, mlp, X_test):
    # Make predictions using the hybrid ensemble
    rf_probs = rf.predict_proba(X_test)
    xgb_probs = xgb.predict_proba(X_test)
    mlp_probs = mlp.predict_proba(X_test)

    # Average the predicted probabilities (soft voting)
    avg_probs = (rf_probs + xgb_probs + mlp_probs) / 3

    # Final prediction: class with highest average probability
    y_pred = np.argmax(avg_probs, axis=1)

    return y_pred, avg_probs

def augment_data_with_noise(X, y, augmentation_factor=5, noise_std=0.1, seed=42):
    #Augment dataset by adding Gaussian noise to normalized features
    np.random.seed(seed)

     
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

     
    X_augmented = []
    y_augmented = []

     
    X_augmented.append(X.values)
    y_augmented.extend(y.values)

     
    for _ in range(augmentation_factor - 1):
         
        noise = np.random.normal(0, noise_std, X_normalized.shape)
        X_noisy_normalized = X_normalized + noise

         
        X_noisy = scaler.inverse_transform(X_noisy_normalized)

        X_augmented.append(X_noisy)
        y_augmented.extend(y.values)

     
    X_final = np.vstack(X_augmented)
    y_final = np.array(y_augmented)

    return pd.DataFrame(X_final, columns=X.columns), pd.Series(y_final)

def step1_repeated_loso_cv(X, y, user_ids, n_trials=20):
    # Step 1: Standard LOSO CV repeated for n_trials
    print("\n" + "="*80)
    print("STEP 1: Standard Leave-One-Subject-Out CV (20 trials)")
    print("="*80)

    all_trial_results = []
    all_f1_scores = []

    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        print("-" * 40)

        logo = LeaveOneGroupOut()
        all_predictions = []
        all_true_labels = []

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, user_ids)):
            test_subject = user_ids.iloc[test_idx[0]]

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

             
            if len(y_test.unique()) < 2:
                continue

             
            rf, xgb, mlp = train_hybrid_model(X_train, y_train, seed=trial*100 + fold)

             
            y_pred, _ = predict_hybrid(rf, xgb, mlp, X_test)

            all_predictions.extend(y_pred)
            all_true_labels.extend(y_test)

         
        accuracy = accuracy_score(all_true_labels, all_predictions)
        precision = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_true_labels, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)

        all_trial_results.append({
            'trial': trial + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        all_f1_scores.append(f1)

        print(f"Trial {trial + 1} - Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")

     
    results_df = pd.DataFrame(all_trial_results)

    print(f"\n" + "="*50)
    print("STEP 1 SUMMARY RESULTS")
    print("="*50)
    print(f"Average Accuracy:  {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"Average Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"Average Recall:    {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    print(f"Average F1 Score:  {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")

    # Save results
    results_df.to_csv('step1_results.csv', index=False)

    return all_f1_scores, results_df

def step2_loso_with_personalization(X, y, user_ids, n_trials=20):
    # Step 2: LOSO CV with 1 normal + 1 hyperglycemia sample for personalization
    print("\n" + "="*80)
    print("STEP 2: LOSO CV with Personalization (20 trials)")
    print("="*80)

    all_trial_results = []
    all_f1_scores = []

    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        print("-" * 40)

        logo = LeaveOneGroupOut()
        all_predictions = []
        all_true_labels = []

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, user_ids)):
            test_subject = user_ids.iloc[test_idx[0]]

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

             
            if len(y_test.unique()) < 2:
                continue

            # Find one normal (0) and one hyperglycemia (1) sample from test set for personalization
            test_normal_idx = y_test[y_test == 0].index
            test_hyper_idx = y_test[y_test == 1].index

            personalization_indices = []
            if len(test_normal_idx) > 0:
                np.random.seed(trial*100 + fold)
                personalization_indices.append(np.random.choice(test_normal_idx))
            if len(test_hyper_idx) > 0:
                np.random.seed(trial*100 + fold + 1)
                personalization_indices.append(np.random.choice(test_hyper_idx))

            if personalization_indices:
                # Move personalization samples from test to train
                X_personalization = X_test.loc[personalization_indices]
                y_personalization = y_test.loc[personalization_indices]

                # Add to training set for personalization
                X_train_personalized = pd.concat([X_train, X_personalization], ignore_index=True)
                y_train_personalized = pd.concat([y_train, y_personalization], ignore_index=True)

                # Remove from test set
                X_test_clean = X_test.drop(personalization_indices)
                y_test_clean = y_test.drop(personalization_indices)

                # Skip if cleaned test set is empty or has only one class
                if len(y_test_clean) == 0 or len(y_test_clean.unique()) < 2:
                    continue

                rf, xgb, mlp = train_hybrid_model(X_train_personalized, y_train_personalized, seed=trial*100 + fold)
                y_pred, _ = predict_hybrid(rf, xgb, mlp, X_test_clean)

                all_predictions.extend(y_pred)
                all_true_labels.extend(y_test_clean)
            else:
                # No personalization possible, use original sets
                rf, xgb, mlp = train_hybrid_model(X_train, y_train, seed=trial*100 + fold)
                y_pred, _ = predict_hybrid(rf, xgb, mlp, X_test)

                all_predictions.extend(y_pred)
                all_true_labels.extend(y_test)

         
        accuracy = accuracy_score(all_true_labels, all_predictions)
        precision = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_true_labels, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)

        all_trial_results.append({
            'trial': trial + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        all_f1_scores.append(f1)

        print(f"Trial {trial + 1} - Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")

     
    results_df = pd.DataFrame(all_trial_results)

    print(f"\n" + "="*50)
    print("STEP 2 SUMMARY RESULTS")
    print("="*50)
    print(f"Average Accuracy:  {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"Average Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"Average Recall:    {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    print(f"Average F1 Score:  {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")

    # Save results
    results_df.to_csv('step2_results.csv', index=False)

    return all_f1_scores, results_df

def step3_loso_with_personalization_and_augmentation(X, y, user_ids, n_trials=20):
    # Step 3: LOSO CV with personalization and data augmentation
    print("\n" + "="*80)
    print("STEP 3: LOSO CV with Personalization + Data Augmentation (20 trials)")
    print("="*80)

    all_trial_results = []
    all_f1_scores = []

    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        print("-" * 40)

        logo = LeaveOneGroupOut()
        all_predictions = []
        all_true_labels = []

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, user_ids)):
            test_subject = user_ids.iloc[test_idx[0]]

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

             
            if len(y_test.unique()) < 2:
                continue

            # Find one normal (0) and one hyperglycemia (1) sample from test set for personalization
            test_normal_idx = y_test[y_test == 0].index
            test_hyper_idx = y_test[y_test == 1].index

            personalization_indices = []
            if len(test_normal_idx) > 0:
                np.random.seed(trial*100 + fold)
                personalization_indices.append(np.random.choice(test_normal_idx))
            if len(test_hyper_idx) > 0:
                np.random.seed(trial*100 + fold + 1)
                personalization_indices.append(np.random.choice(test_hyper_idx))

            if personalization_indices:
                # Move personalization samples from test to train
                X_personalization = X_test.loc[personalization_indices]
                y_personalization = y_test.loc[personalization_indices]

                # Add to training set for personalization
                X_train_personalized = pd.concat([X_train, X_personalization], ignore_index=True)
                y_train_personalized = pd.concat([y_train, y_personalization], ignore_index=True)

                # Remove from test set
                X_test_clean = X_test.drop(personalization_indices)
                y_test_clean = y_test.drop(personalization_indices)

                # Skip if cleaned test set is empty or has only one class
                if len(y_test_clean) == 0 or len(y_test_clean.unique()) < 2:
                    continue

                # Augment training data with Gaussian noise (5x original size)
                X_train_aug, y_train_aug = augment_data_with_noise(
                    X_train_personalized, y_train_personalized,
                    augmentation_factor=5,
                    seed=trial*100 + fold
                )

                rf, xgb, mlp = train_hybrid_model(X_train_aug, y_train_aug, seed=trial*100 + fold)

                y_pred, _ = predict_hybrid(rf, xgb, mlp, X_test_clean)

                all_predictions.extend(y_pred)
                all_true_labels.extend(y_test_clean)
            else:
                # No personalization possible, use original sets with augmentation
                X_train_aug, y_train_aug = augment_data_with_noise(
                    X_train, y_train,
                    augmentation_factor=5,
                    seed=trial*100 + fold
                )

                rf, xgb, mlp = train_hybrid_model(X_train_aug, y_train_aug, seed=trial*100 + fold)
                y_pred, _ = predict_hybrid(rf, xgb, mlp, X_test)

                all_predictions.extend(y_pred)
                all_true_labels.extend(y_test)

         
        accuracy = accuracy_score(all_true_labels, all_predictions)
        precision = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_true_labels, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)

        all_trial_results.append({
            'trial': trial + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        all_f1_scores.append(f1)

        print(f"Trial {trial + 1} - Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")

     
    results_df = pd.DataFrame(all_trial_results)

    print(f"\n" + "="*50)
    print("STEP 3 SUMMARY RESULTS")
    print("="*50)
    print(f"Average Accuracy:  {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"Average Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"Average Recall:    {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    print(f"Average F1 Score:  {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")

    # Save results
    results_df.to_csv('step3_results.csv', index=False)

    return all_f1_scores, results_df

def perform_anova_analysis(f1_step1, f1_step2, f1_step3):
    # Perform ANOVA test on F1 scores from three steps
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - ANOVA")
    print("="*80)

    # Perform one-way ANOVA
    f_statistic, p_value = stats.f_oneway(f1_step1, f1_step2, f1_step3)

    print(f"One-way ANOVA Results:")
    print(f"F-statistic: {f_statistic:.6f}")
    print(f"P-value: {p_value:.6f}")

    # Interpret results
    alpha = 0.05
    if p_value < alpha:
        print(f"\nResult: Significant difference between groups (p < {alpha})")
    else:
        print(f"\nResult: No significant difference between groups (p >= {alpha})")

    # Perform pairwise t-tests for post-hoc analysis
    print(f"\nPost-hoc pairwise t-tests:")

    # Step 1 vs Step 2
    t_stat_12, p_val_12 = stats.ttest_rel(f1_step1, f1_step2)
    print(f"Step 1 vs Step 2: t = {t_stat_12:.4f}, p = {p_val_12:.6f}")

    # Step 1 vs Step 3
    t_stat_13, p_val_13 = stats.ttest_rel(f1_step1, f1_step3)
    print(f"Step 1 vs Step 3: t = {t_stat_13:.4f}, p = {p_val_13:.6f}")

    # Step 2 vs Step 3
    t_stat_23, p_val_23 = stats.ttest_rel(f1_step2, f1_step3)
    print(f"Step 2 vs Step 3: t = {t_stat_23:.4f}, p = {p_val_23:.6f}")

    # Save statistical results
    anova_results = pd.DataFrame({
        'test': ['ANOVA', 'Step1_vs_Step2', 'Step1_vs_Step3', 'Step2_vs_Step3'],
        'statistic': [f_statistic, t_stat_12, t_stat_13, t_stat_23],
        'p_value': [p_value, p_val_12, p_val_13, p_val_23]
    })
    anova_results.to_csv('statistical_analysis.csv', index=False)

    # Save F1 scores for all steps
    f1_scores_df = pd.DataFrame({
        'trial': range(1, 21),
        'step1_f1': f1_step1,
        'step2_f1': f1_step2,
        'step3_f1': f1_step3
    })
    f1_scores_df.to_csv('all_f1_scores.csv', index=False)

    return anova_results

 
if __name__ == "__main__":
    print("Starting Three-Step Machine Learning Analysis")
    print("="*80)

    # Step 1: Standard LOSO CV
    f1_step1, results_step1 = step1_repeated_loso_cv(X, y, user_ids, n_trials=20)

    # Step 2: LOSO CV with personalization
    f1_step2, results_step2 = step2_loso_with_personalization(X, y, user_ids, n_trials=20)

    # Step 3: LOSO CV with personalization and data augmentation
    f1_step3, results_step3 = step3_loso_with_personalization_and_augmentation(X, y, user_ids, n_trials=20)

    # Display F1 score arrays
    print("\n" + "="*80)
    print("F1 SCORE ARRAYS")
    print("="*80)

    print(f"\nStep 1 F1 Scores (20 trials):")
    print(f"{f1_step1}")

    print(f"\nStep 2 F1 Scores (20 trials):")
    print(f"{f1_step2}")

    print(f"\nStep 3 F1 Scores (20 trials):")
    print(f"{f1_step3}")

    # Perform statistical analysis
    anova_results = perform_anova_analysis(f1_step1, f1_step2, f1_step3)

    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Files saved:")
    print("- step1_results.csv")
    print("- step2_results.csv")
    print("- step3_results.csv")
    print("- all_f1_scores.csv")
    print("- statistical_analysis.csv")

# import pandas as pd
# import numpy as np
# import random
# from sklearn.model_selection import LeaveOneGroupOut
# from imblearn.over_sampling import ADASYN
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')

# data = pd.read_csv('dataset_all_features_user_id.csv')
# print(f"Original data shape: {data.shape}")

# data = data.dropna()
# print(f"Data shape after removing NaN: {data.shape}")

# data = data.drop(columns=['absolute_auc','respective_auc','max_postprandial_gluc'])

# user_ids = data['user_id']
# X = data.drop(['postprandial_hyperglycemia_140', 'user_id'], axis=1).astype(float)
# y = data['postprandial_hyperglycemia_140'].astype(int)

# print(f"Number of unique subjects: {len(user_ids.unique())}")
# print(f"Class distribution: {y.value_counts().to_dict()}")

# def train_hybrid_model(X_train, y_train, seed=42):  
#     adasyn = ADASYN(random_state=seed)
#     X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)

#     rf = RandomForestClassifier(n_estimators=100, random_state=seed)
#     rf.fit(X_train_balanced, y_train_balanced)

#     xgb = XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss')
#     xgb.fit(X_train_balanced, y_train_balanced)
    
#     mlp = MLPClassifier(
#         hidden_layer_sizes=(160, 80, 40, 40, 40, 40, 20, 10),
#         max_iter=500,
#         random_state=seed
#     )
#     mlp.fit(X_train_balanced, y_train_balanced)

#     return rf, xgb, mlp

# def predict_hybrid(rf, xgb, mlp, X_test):
#     rf_probs = rf.predict_proba(X_test)
#     xgb_probs = xgb.predict_proba(X_test)
#     mlp_probs = mlp.predict_proba(X_test)

#     # Average the predicted probabilities (soft voting)
#     avg_probs = (rf_probs + xgb_probs + mlp_probs) / 3

#     # Final prediction: class with highest average probability
#     y_pred = np.argmax(avg_probs, axis=1)

#     return y_pred, avg_probs

# def augment_data_with_noise(X, y, augmentation_factor=5, noise_std=0.1, seed=42):
#     np.random.seed(seed)

     
#     scaler = StandardScaler()
#     X_normalized = scaler.fit_transform(X)

     
#     X_augmented = []
#     y_augmented = []

     
#     X_augmented.append(X.values)
#     y_augmented.extend(y.values)
   
#     for _ in range(augmentation_factor - 1):
         
#         noise = np.random.normal(0, noise_std, X_normalized.shape)
#         X_noisy_normalized = X_normalized + noise

         
#         X_noisy = scaler.inverse_transform(X_noisy_normalized)

#         X_augmented.append(X_noisy)
#         y_augmented.extend(y.values)

#     X_final = np.vstack(X_augmented)
#     y_final = np.array(y_augmented)

#     return pd.DataFrame(X_final, columns=X.columns), pd.Series(y_final)

# def step4_loso_with_augmentation_only(X, y, user_ids, n_trials=20):
#     #Step 4: LOSO CV with data augmentation only (no personalization)
#     print("\n" + "="*80)
#     print("STEP 4: LOSO CV with Data Augmentation Only (20 trials)")
#     print("="*80)

#     all_trial_results = []
#     all_f1_scores = []

#     for trial in range(n_trials):
#         print(f"\nTrial {trial + 1}/{n_trials}")
#         print("-" * 40)

#         logo = LeaveOneGroupOut()
#         all_predictions = []
#         all_true_labels = []

#         for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, user_ids)):
#             test_subject = user_ids.iloc[test_idx[0]]

#             X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#             y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

             
#             if len(y_test.unique()) < 2:
#                 continue

#             # Augment training data with Gaussian noise (5x original size)
#             X_train_aug, y_train_aug = augment_data_with_noise(
#                 X_train, y_train,
#                 augmentation_factor=5,
#                 seed=trial*100 + fold
#             )

#             rf, xgb, mlp = train_hybrid_model(X_train_aug, y_train_aug, seed=trial*100 + fold)

#             y_pred, _ = predict_hybrid(rf, xgb, mlp, X_test)

#             all_predictions.extend(y_pred)
#             all_true_labels.extend(y_test)

         
#         accuracy = accuracy_score(all_true_labels, all_predictions)
#         precision = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0)
#         recall = recall_score(all_true_labels, all_predictions, average='macro', zero_division=0)
#         f1 = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)

#         all_trial_results.append({
#             'trial': trial + 1,
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1
#         })
#         all_f1_scores.append(f1)

#         print(f"Trial {trial + 1} - Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")

     
#     results_df = pd.DataFrame(all_trial_results)

#     print(f"\n" + "="*50)
#     print("STEP 4 SUMMARY RESULTS")
#     print("="*50)
#     print(f"Average Accuracy:  {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
#     print(f"Average Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
#     print(f"Average Recall:    {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
#     print(f"Average F1 Score:  {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")

#     # Save results
#     results_df.to_csv('step4_results.csv', index=False)

#     return all_f1_scores, results_df

#  
# if __name__ == "__main__":
#     print("Starting Step 4: Data Augmentation Only Analysis")
#     print("="*80)

#     # Step 4: LOSO CV with data augmentation only
#     f1_step4, results_step4 = step4_loso_with_augmentation_only(X, y, user_ids, n_trials=20)

#     # Display F1 score array
#     print("\n" + "="*80)
#     print("F1 SCORE ARRAY")
#     print("="*80)

#     print(f"\nStep 4 F1 Scores (20 trials):")
#     print(f"{f1_step4}")

#     # Display basic statistics
#     print(f"\nStep 4 F1 Score Statistics:")
#     print(f"Mean: {np.mean(f1_step4):.4f}")
#     print(f"Std:  {np.std(f1_step4):.4f}")
#     print(f"Min:  {np.min(f1_step4):.4f}")
#     print(f"Max:  {np.max(f1_step4):.4f}")

#     # Save F1 scores
#     f1_scores_df = pd.DataFrame({
#         'trial': range(1, 21),
#         'step4_f1': f1_step4
#     })
#     f1_scores_df.to_csv('step4_f1_scores.csv', index=False)

#     print(f"\n" + "="*80)
#     print("STEP 4 ANALYSIS COMPLETE")
#     print("="*80)
#     print("Files saved:")
#     print("- step4_results.csv")
#     print("- step4_f1_scores.csv")

# import pandas as pd
# import numpy as np
# import random
# from sklearn.model_selection import StratifiedKFold
# from imblearn.over_sampling import ADASYN
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')

 
# data = pd.read_csv('dataset_all_features_user_id.csv')
# print(f"Original data shape: {data.shape}")

# data = data.dropna()
# print(f"Data shape after removing NaN: {data.shape}")

# data = data.drop(columns=['absolute_auc','respective_auc','max_postprandial_gluc'])

 
# user_ids = data['user_id']
# X = data.drop(['postprandial_hyperglycemia_140', 'user_id'], axis=1).astype(float)
# y = data['postprandial_hyperglycemia_140'].astype(int)

# print(f"Total number of samples: {len(X)}")
# print(f"Class distribution: {y.value_counts().to_dict()}")
# print(f"Class ratio: {y.value_counts(normalize=True).to_dict()}")

# def train_hybrid_model(X_train, y_train, seed=42):
     
     
#     rf = RandomForestClassifier(n_estimators=100, random_state=seed)
#     rf.fit(X_train, y_train)

     
#     xgb = XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss')
#     xgb.fit(X_train, y_train)

     
#     mlp = MLPClassifier(
#         hidden_layer_sizes=(160, 80, 40, 40, 40, 40, 20, 10),
#         max_iter=500,
#         random_state=seed
#     )
#     mlp.fit(X_train, y_train)

#     return rf, xgb, mlp

# def predict_hybrid(rf, xgb, mlp, X_test):
#     # Make predictions using the hybrid ensemble
#     rf_probs = rf.predict_proba(X_test)
#     xgb_probs = xgb.predict_proba(X_test)
#     mlp_probs = mlp.predict_proba(X_test)

#     # Average the predicted probabilities (soft voting)
#     avg_probs = (rf_probs + xgb_probs + mlp_probs) / 3

#     # Final prediction: class with highest average probability
#     y_pred = np.argmax(avg_probs, axis=1)

#     return y_pred, avg_probs

# def balance_training_data(X_train, y_train, seed=42):
#     # Balance the training data using ADASYN
#     adasyn = ADASYN(random_state=seed)
#     X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)
#     return X_train_balanced, y_train_balanced

# def augment_data_with_noise(X, y, augmentation_factor=5, noise_std=0.1, seed=42):
#     # Augment dataset by adding Gaussian noise to normalized features
#     np.random.seed(seed)

     
#     scaler = StandardScaler()
#     X_normalized = scaler.fit_transform(X)

     
#     X_augmented = []
#     y_augmented = []

     
#     X_augmented.append(X.values)
#     y_augmented.extend(y.values)

     
#     for _ in range(augmentation_factor - 1):
         
#         noise = np.random.normal(0, noise_std, X_normalized.shape)
#         X_noisy_normalized = X_normalized + noise

         
#         X_noisy = scaler.inverse_transform(X_noisy_normalized)

#         X_augmented.append(X_noisy)
#         y_augmented.extend(y.values)

     
#     X_final = np.vstack(X_augmented)
#     y_final = np.array(y_augmented)

#     return pd.DataFrame(X_final, columns=X.columns), pd.Series(y_final)

# def stratified_kfold_without_augmentation(X, y, n_trials=3, n_folds=10):
#     # 10-Fold Stratified CV without data augmentation
#     print("\n" + "="*80)
#     print("10-FOLD STRATIFIED CV WITHOUT AUGMENTATION (3 trials)")
#     print("="*80)

#     all_trial_results = []
#     all_f1_scores = []

#     for trial in range(n_trials):
#         print(f"\nTrial {trial + 1}/{n_trials}")
#         print("-" * 40)

#         # Initialize stratified k-fold
#         skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=trial*100)

#         fold_predictions = []
#         fold_true_labels = []
#         fold_metrics = []

#         for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
#             X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#             y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#             print(f"  Fold {fold + 1}: Train={len(y_train)} (Class dist: {y_train.value_counts().to_dict()}), "
#                   f"Test={len(y_test)} (Class dist: {y_test.value_counts().to_dict()})")

             
#             X_train_balanced, y_train_balanced = balance_training_data(X_train, y_train, seed=trial*100 + fold)

#             print(f"    After balancing: Train={len(y_train_balanced)} (Class dist: {y_train_balanced.value_counts().to_dict()})")

            
#             rf, xgb, mlp = train_hybrid_model(X_train_balanced, y_train_balanced, seed=trial*100 + fold)

             
#             y_pred, _ = predict_hybrid(rf, xgb, mlp, X_test)

#             # Calculate fold metrics
#             fold_acc = accuracy_score(y_test, y_pred)
#             fold_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
#             fold_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
#             fold_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

#             fold_metrics.append({
#                 'fold': fold + 1,
#                 'accuracy': fold_acc,
#                 'precision': fold_prec,
#                 'recall': fold_rec,
#                 'f1_score': fold_f1
#             })

#             fold_predictions.extend(y_pred)
#             fold_true_labels.extend(y_test)

#             print(f"    Fold {fold + 1} - Acc: {fold_acc:.4f}, Prec: {fold_prec:.4f}, Rec: {fold_rec:.4f}, F1: {fold_f1:.4f}")

         
#         overall_accuracy = accuracy_score(fold_true_labels, fold_predictions)
#         overall_precision = precision_score(fold_true_labels, fold_predictions, average='macro', zero_division=0)
#         overall_recall = recall_score(fold_true_labels, fold_predictions, average='macro', zero_division=0)
#         overall_f1 = f1_score(fold_true_labels, fold_predictions, average='macro', zero_division=0)

#         # Calculate average across folds
#         fold_metrics_df = pd.DataFrame(fold_metrics)
#         avg_fold_accuracy = fold_metrics_df['accuracy'].mean()
#         avg_fold_precision = fold_metrics_df['precision'].mean()
#         avg_fold_recall = fold_metrics_df['recall'].mean()
#         avg_fold_f1 = fold_metrics_df['f1_score'].mean()

#         all_trial_results.append({
#             'trial': trial + 1,
#             'overall_accuracy': overall_accuracy,
#             'overall_precision': overall_precision,
#             'overall_recall': overall_recall,
#             'overall_f1_score': overall_f1,
#             'avg_fold_accuracy': avg_fold_accuracy,
#             'avg_fold_precision': avg_fold_precision,
#             'avg_fold_recall': avg_fold_recall,
#             'avg_fold_f1_score': avg_fold_f1
#         })
#         all_f1_scores.append(overall_f1)

#         print(f"\nTrial {trial + 1} Summary:")
#         print(f"  Overall - Acc: {overall_accuracy:.4f}, Prec: {overall_precision:.4f}, Rec: {overall_recall:.4f}, F1: {overall_f1:.4f}")
#         print(f"  Avg Fold - Acc: {avg_fold_accuracy:.4f}, Prec: {avg_fold_precision:.4f}, Rec: {avg_fold_recall:.4f}, F1: {avg_fold_f1:.4f}")

     
#     results_df = pd.DataFrame(all_trial_results)

#     print(f"\n" + "="*50)
#     print("WITHOUT AUGMENTATION SUMMARY RESULTS")
#     print("="*50)
#     print(f"Average Overall Accuracy:  {results_df['overall_accuracy'].mean():.4f} ± {results_df['overall_accuracy'].std():.4f}")
#     print(f"Average Overall Precision: {results_df['overall_precision'].mean():.4f} ± {results_df['overall_precision'].std():.4f}")
#     print(f"Average Overall Recall:    {results_df['overall_recall'].mean():.4f} ± {results_df['overall_recall'].std():.4f}")
#     print(f"Average Overall F1 Score:  {results_df['overall_f1_score'].mean():.4f} ± {results_df['overall_f1_score'].std():.4f}")

#     # Save results
#     results_df.to_csv('stratified_kfold_without_augmentation_results.csv', index=False)

#     return all_f1_scores, results_df

# def stratified_kfold_with_augmentation(X, y, n_trials=3, n_folds=10):
#     # 10-Fold Stratified CV with data augmentation
#     print("\n" + "="*80)
#     print("10-FOLD STRATIFIED CV WITH AUGMENTATION (3 trials)")
#     print("="*80)

#     all_trial_results = []
#     all_f1_scores = []

#     for trial in range(n_trials):
#         print(f"\nTrial {trial + 1}/{n_trials}")
#         print("-" * 40)

#         # Initialize stratified k-fold
#         skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=trial*100)

#         fold_predictions = []
#         fold_true_labels = []
#         fold_metrics = []

#         for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
#             X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#             y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#             print(f"  Fold {fold + 1}: Train={len(y_train)} (Class dist: {y_train.value_counts().to_dict()}), "
#                   f"Test={len(y_test)} (Class dist: {y_test.value_counts().to_dict()})")

#             X_train_balanced, y_train_balanced = balance_training_data(X_train, y_train, seed=trial*100 + fold)

#             print(f"    After balancing: Train={len(y_train_balanced)} (Class dist: {y_train_balanced.value_counts().to_dict()})")

#             # Augment the balanced training data with Gaussian noise (5x original size)
#             X_train_aug, y_train_aug = augment_data_with_noise(
#                 X_train_balanced, y_train_balanced,
#                 augmentation_factor=5,
#                 seed=trial*100 + fold
#             )

#             print(f"    After augmentation: Train={len(y_train_aug)} (Class dist: {y_train_aug.value_counts().to_dict()})")

#             rf, xgb, mlp = train_hybrid_model(X_train_aug, y_train_aug, seed=trial*100 + fold)

#             y_pred, _ = predict_hybrid(rf, xgb, mlp, X_test)

#             # Calculate fold metrics
#             fold_acc = accuracy_score(y_test, y_pred)
#             fold_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
#             fold_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
#             fold_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

#             fold_metrics.append({
#                 'fold': fold + 1,
#                 'accuracy': fold_acc,
#                 'precision': fold_prec,
#                 'recall': fold_rec,
#                 'f1_score': fold_f1
#             })

#             fold_predictions.extend(y_pred)
#             fold_true_labels.extend(y_test)

#             print(f"    Fold {fold + 1} - Acc: {fold_acc:.4f}, Prec: {fold_prec:.4f}, Rec: {fold_rec:.4f}, F1: {fold_f1:.4f}")

         
#         overall_accuracy = accuracy_score(fold_true_labels, fold_predictions)
#         overall_precision = precision_score(fold_true_labels, fold_predictions, average='macro', zero_division=0)
#         overall_recall = recall_score(fold_true_labels, fold_predictions, average='macro', zero_division=0)
#         overall_f1 = f1_score(fold_true_labels, fold_predictions, average='macro', zero_division=0)

#         # Calculate average across folds
#         fold_metrics_df = pd.DataFrame(fold_metrics)
#         avg_fold_accuracy = fold_metrics_df['accuracy'].mean()
#         avg_fold_precision = fold_metrics_df['precision'].mean()
#         avg_fold_recall = fold_metrics_df['recall'].mean()
#         avg_fold_f1 = fold_metrics_df['f1_score'].mean()

#         all_trial_results.append({
#             'trial': trial + 1,
#             'overall_accuracy': overall_accuracy,
#             'overall_precision': overall_precision,
#             'overall_recall': overall_recall,
#             'overall_f1_score': overall_f1,
#             'avg_fold_accuracy': avg_fold_accuracy,
#             'avg_fold_precision': avg_fold_precision,
#             'avg_fold_recall': avg_fold_recall,
#             'avg_fold_f1_score': avg_fold_f1
#         })
#         all_f1_scores.append(overall_f1)

#         print(f"\nTrial {trial + 1} Summary:")
#         print(f"  Overall - Acc: {overall_accuracy:.4f}, Prec: {overall_precision:.4f}, Rec: {overall_recall:.4f}, F1: {overall_f1:.4f}")
#         print(f"  Avg Fold - Acc: {avg_fold_accuracy:.4f}, Prec: {avg_fold_precision:.4f}, Rec: {avg_fold_recall:.4f}, F1: {avg_fold_f1:.4f}")

     
#     results_df = pd.DataFrame(all_trial_results)

#     print(f"\n" + "="*50)
#     print("WITH AUGMENTATION SUMMARY RESULTS")
#     print("="*50)
#     print(f"Average Overall Accuracy:  {results_df['overall_accuracy'].mean():.4f} ± {results_df['overall_accuracy'].std():.4f}")
#     print(f"Average Overall Precision: {results_df['overall_precision'].mean():.4f} ± {results_df['overall_precision'].std():.4f}")
#     print(f"Average Overall Recall:    {results_df['overall_recall'].mean():.4f} ± {results_df['overall_recall'].std():.4f}")
#     print(f"Average Overall F1 Score:  {results_df['overall_f1_score'].mean():.4f} ± {results_df['overall_f1_score'].std():.4f}")

#     # Save results
#     results_df.to_csv('stratified_kfold_with_augmentation_results.csv', index=False)

#     return all_f1_scores, results_df

# def perform_statistical_analysis(f1_without_aug, f1_with_aug):
#     # Perform statistical analysis comparing with and without augmentation
#     print("\n" + "="*80)
#     print("STATISTICAL ANALYSIS - PAIRED T-TEST")
#     print("="*80)

#     # Perform paired t-test
#     t_statistic, p_value = stats.ttest_rel(f1_without_aug, f1_with_aug)

#     print(f"Paired t-test Results:")
#     print(f"T-statistic: {t_statistic:.6f}")
#     print(f"P-value: {p_value:.6f}")

#     # Interpret results
#     alpha = 0.05
#     if p_value < alpha:
#         print(f"\nResult: Significant difference between methods (p < {alpha})")
#         if np.mean(f1_with_aug) > np.mean(f1_without_aug):
#             print("Augmentation significantly improves F1 score")
#         else:
#             print("Augmentation significantly decreases F1 score")
#     else:
#         print(f"\nResult: No significant difference between methods (p >= {alpha})")

#     # Calculate effect size (Cohen's d)
#     mean_diff = np.mean(f1_with_aug) - np.mean(f1_without_aug)
#     pooled_std = np.sqrt(((len(f1_with_aug) - 1) * np.var(f1_with_aug, ddof=1) +
#                           (len(f1_without_aug) - 1) * np.var(f1_without_aug, ddof=1)) /
#                          (len(f1_with_aug) + len(f1_without_aug) - 2))

#     cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0

#     print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")

#     # Interpret effect size
#     if abs(cohens_d) < 0.2:
#         effect_size_interpretation = "negligible"
#     elif abs(cohens_d) < 0.5:
#         effect_size_interpretation = "small"
#     elif abs(cohens_d) < 0.8:
#         effect_size_interpretation = "medium"
#     else:
#         effect_size_interpretation = "large"

#     print(f"Effect size interpretation: {effect_size_interpretation}")

#     # Additional descriptive statistics
#     print(f"\nDescriptive Statistics:")
#     print(f"Without Augmentation - Mean: {np.mean(f1_without_aug):.4f}, Std: {np.std(f1_without_aug):.4f}")
#     print(f"With Augmentation    - Mean: {np.mean(f1_with_aug):.4f}, Std: {np.std(f1_with_aug):.4f}")
#     print(f"Mean Difference: {mean_diff:.4f}")

#     # Save statistical results
#     statistical_results = pd.DataFrame({
#         'test': ['paired_t_test'],
#         'statistic': [t_statistic],
#         'p_value': [p_value],
#         'cohens_d': [cohens_d],
#         'effect_size': [effect_size_interpretation],
#         'mean_without_aug': [np.mean(f1_without_aug)],
#         'mean_with_aug': [np.mean(f1_with_aug)],
#         'std_without_aug': [np.std(f1_without_aug)],
#         'std_with_aug': [np.std(f1_with_aug)],
#         'mean_difference': [mean_diff]
#     })
#     statistical_results.to_csv('stratified_kfold_statistical_analysis.csv', index=False)

#     # Save F1 scores for both methods
#     f1_scores_df = pd.DataFrame({
#         'trial': range(1, len(f1_without_aug) + 1),
#         'f1_without_augmentation': f1_without_aug,
#         'f1_with_augmentation': f1_with_aug
#     })
#     f1_scores_df.to_csv('stratified_kfold_f1_scores_comparison.csv', index=False)

#     return statistical_results

#  
# if __name__ == "__main__":
#     print("Starting 10-Fold Stratified Cross-Validation Analysis")
#     print("="*80)
#     print(f"Total samples: {len(X)}")
#     print(f"Each fold will have approximately {len(X)//10} samples")
#     print("Training sets will be balanced before and after augmentation")

#     # Method 1: 10-Fold Stratified CV without augmentation (with balancing)
#     f1_without_aug, results_without_aug = stratified_kfold_without_augmentation(X, y, n_trials=3, n_folds=10)

#     # Method 2: 10-Fold Stratified CV with augmentation (with balancing)
#     f1_with_aug, results_with_aug = stratified_kfold_with_augmentation(X, y, n_trials=3, n_folds=10)

#     # Display F1 score arrays
#     print("\n" + "="*80)
#     print("F1 SCORE ARRAYS")
#     print("="*80)

#     print(f"\nWithout Augmentation F1 Scores (3 trials):")
#     print(f"{f1_without_aug}")

#     print(f"\nWith Augmentation F1 Scores (3 trials):")
#     print(f"{f1_with_aug}")

#     # Perform statistical analysis
#     statistical_results = perform_statistical_analysis(f1_without_aug, f1_with_aug)

#     print(f"\n" + "="*80)
#     print("10-FOLD STRATIFIED CV ANALYSIS COMPLETE")
#     print("="*80)
#     print("Files saved:")
#     print("- stratified_kfold_without_augmentation_results.csv")
#     print("- stratified_kfold_with_augmentation_results.csv")
#     print("- stratified_kfold_f1_scores_comparison.csv")
#     print("- stratified_kfold_statistical_analysis.csv")

