
## Common block for all

import pandas as pd
import numpy as np

data  = pd.read_csv('dataset_all_features.csv')

data = data.dropna()

data = data.drop(columns=['absolute_auc','respective_auc','max_postprandial_gluc'])

# Split dataset
X = data.drop('postprandial_hyperglycemia_140', axis=1).astype(float)
y = data['postprandial_hyperglycemia_140'].astype(int)



## From the following import to calling the functions, call one function (scroll down until at the end of this file)


### RF, XGB, MLP, RF+XGB, RF+MLP, XGB+MLP, RF+XGB+MLP, other variations of RF+XGB+MLP


### Just RF
def pure_full_RF():
    import random
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import ADASYN
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    def train_and_test(X, y, repeat):
         
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        models = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(repeat):
            seed = 42 * i

             
            np.random.seed(seed)
            random.seed(seed)

             
            positive_idx = y[y == 1].index
            negative_idx = y[y == 0].index

             
            pos_test_idx = np.random.choice(positive_idx, 10, replace=False)
            neg_test_idx = np.random.choice(negative_idx, 10, replace=False)
            test_idx = np.concatenate([pos_test_idx, neg_test_idx])

             
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            X_remaining = X.drop(test_idx)
            y_remaining = y.drop(test_idx)

             
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"Test set class distribution (iteration {i}): {dict(zip(unique, counts))}")

             
            adasyn = ADASYN(random_state=seed)
            X_train, y_train = adasyn.fit_resample(X_remaining, y_remaining)

             
            X_train.to_csv(f'X_train_{i}.csv', index=False)
            X_test.to_csv(f'X_test_{i}.csv', index=False)
            y_train.to_csv(f'y_train_{i}.csv', index=False)
            y_test.to_csv(f'y_test_{i}.csv', index=False)

             
            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_train, y_train)

             
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'Iteration {i}: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}')
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            models.append(rf)

        return models, accuracy_scores, precision_scores, recall_scores, f1_scores

    print("Training and testing with Random Forest models")
    models, accs, precs, recs, f1s = train_and_test(X, y, 100)
    print(f"Average Metrics: Accuracy={np.mean(accs)}, Precision={np.mean(precs)}, Recall={np.mean(recs)}, F1={np.mean(f1s)}")


### Just XGB
def pure_full_XGB():
    import random
    import numpy as np
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import ADASYN
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pandas as pd


    def train_and_test(X, y, repeat):
         
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        models = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(repeat):
            seed = 42 * i

             
            np.random.seed(seed)
            random.seed(seed)

             
            positive_idx = y[y == 1].index
            negative_idx = y[y == 0].index

             
            pos_test_idx = np.random.choice(positive_idx, 10, replace=False)
            neg_test_idx = np.random.choice(negative_idx, 10, replace=False)
            test_idx = np.concatenate([pos_test_idx, neg_test_idx])

             
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            X_remaining = X.drop(test_idx)
            y_remaining = y.drop(test_idx)

             
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"Test set class distribution (iteration {i}): {dict(zip(unique, counts))}")

             
            adasyn = ADASYN(random_state=seed)
            X_train, y_train = adasyn.fit_resample(X_remaining, y_remaining)

             
            X_train.to_csv(f'X_train_{i}.csv', index=False)
            X_test.to_csv(f'X_test_{i}.csv', index=False)
            y_train.to_csv(f'y_train_{i}.csv', index=False)
            y_test.to_csv(f'y_test_{i}.csv', index=False)

             
            xgb = XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)

             
            y_pred = xgb.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'Iteration {i}: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}')
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            models.append(xgb)

        return models, accuracy_scores, precision_scores, recall_scores, f1_scores

    print("Training and testing with XGBoost models")
    models, accs, precs, recs, f1s = train_and_test(X, y, 100)
    print(f"Average Metrics: Accuracy={np.mean(accs)}, Precision={np.mean(precs)}, Recall={np.mean(recs)}, F1={np.mean(f1s)}")


### Just MLP

def pure_full_mlp():
    import random
    import numpy as np
    import pandas as pd
    from imblearn.over_sampling import ADASYN
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    def train_and_test_mlp(X, y, repeat):
         
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(repeat):
            seed = 42 * i

             
            np.random.seed(seed)
            random.seed(seed)

             
            positive_idx = y[y == 1].index
            negative_idx = y[y == 0].index

             
            pos_test_idx = np.random.choice(positive_idx, 10, replace=False)
            neg_test_idx = np.random.choice(negative_idx, 10, replace=False)
            test_idx = np.concatenate([pos_test_idx, neg_test_idx])

             
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            X_remaining = X.drop(test_idx)
            y_remaining = y.drop(test_idx)

             
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"Test set class distribution (iteration {i}): {dict(zip(unique, counts))}")

             
            adasyn = ADASYN(random_state=seed)
            X_train, y_train = adasyn.fit_resample(X_remaining, y_remaining)

             
            X_train.to_csv(f'X_train_{i}.csv', index=False)
            X_test.to_csv(f'X_test_{i}.csv', index=False)
            y_train.to_csv(f'y_train_{i}.csv', index=False)
            y_test.to_csv(f'y_test_{i}.csv', index=False)

            mlp = MLPClassifier(
                hidden_layer_sizes=(160, 80, 40, 40, 40, 40, 20, 10),
                max_iter=500,
                random_state=seed
            )
            mlp.fit(X_train, y_train)

            # Predict on test set
            y_pred = mlp.predict(X_test)

             
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f"Iteration {i}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return accuracy_scores, precision_scores, recall_scores, f1_scores


    print("Training and testing with MLP model")
    accs, precs, recs, f1s = train_and_test_mlp(X, y, 100)
    print(f"Average Metrics over 100 runs:")
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f}")

### RF + XGB
def hybrid_full_rf_xgb():
    import random
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import ADASYN
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    def train_and_test_hybrid(X, y, repeat):
         
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(repeat):
            seed = 42 * i

             
            np.random.seed(seed)
            random.seed(seed)

             
            positive_idx = y[y == 1].index
            negative_idx = y[y == 0].index

             
            pos_test_idx = np.random.choice(positive_idx, 10, replace=False)
            neg_test_idx = np.random.choice(negative_idx, 10, replace=False)
            test_idx = np.concatenate([pos_test_idx, neg_test_idx])

             
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            X_remaining = X.drop(test_idx)
            y_remaining = y.drop(test_idx)

             
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"Test set class distribution (iteration {i}): {dict(zip(unique, counts))}")

             
            adasyn = ADASYN(random_state=seed)
            X_train, y_train = adasyn.fit_resample(X_remaining, y_remaining)

             
            X_train.to_csv(f'X_train_{i}.csv', index=False)
            X_test.to_csv(f'X_test_{i}.csv', index=False)
            y_train.to_csv(f'y_train_{i}.csv', index=False)
            y_test.to_csv(f'y_test_{i}.csv', index=False)

             
            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_train, y_train)

             
            xgb = XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)

            # Predict probabilities from both models
            rf_probs = rf.predict_proba(X_test)
            xgb_probs = xgb.predict_proba(X_test)

             
            avg_probs = (rf_probs + xgb_probs) / 2

             
            y_pred = np.argmax(avg_probs, axis=1)

             
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'Iteration {i}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return accuracy_scores, precision_scores, recall_scores, f1_scores

    print("Training and testing with Hybrid Random Forest + XGBoost model")
    accs, precs, recs, f1s = train_and_test_hybrid(X, y, 100)
    print(f"Average Metrics over 100 runs:")
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f}")


### RF + MLP

def hybrid_full_rf_mlp():
    import random
    import numpy as np
    import pandas as pd
    from imblearn.over_sampling import ADASYN
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    def train_and_test_hybrid_rf_mlp(X, y, repeat):
         
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(repeat):
            seed = 42 * i

             
            np.random.seed(seed)
            random.seed(seed)

             
            positive_idx = y[y == 1].index
            negative_idx = y[y == 0].index

             
            pos_test_idx = np.random.choice(positive_idx, 10, replace=False)
            neg_test_idx = np.random.choice(negative_idx, 10, replace=False)
            test_idx = np.concatenate([pos_test_idx, neg_test_idx])

             
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            X_remaining = X.drop(test_idx)
            y_remaining = y.drop(test_idx)

             
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"Test set class distribution (iteration {i}): {dict(zip(unique, counts))}")

             
            adasyn = ADASYN(random_state=seed)
            X_train, y_train = adasyn.fit_resample(X_remaining, y_remaining)

             
            X_train.to_csv(f'X_train_{i}.csv', index=False)
            X_test.to_csv(f'X_test_{i}.csv', index=False)
            y_train.to_csv(f'y_train_{i}.csv', index=False)
            y_test.to_csv(f'y_test_{i}.csv', index=False)

             
            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_train, y_train)

               
            mlp = MLPClassifier(
                hidden_layer_sizes=(160, 80, 40, 40, 40, 40, 20, 10),
                max_iter=500,
                random_state=seed
            )
            mlp.fit(X_train, y_train)

             
            rf_probs = rf.predict_proba(X_test)
            mlp_probs = mlp.predict_proba(X_test)
            avg_probs = (rf_probs + mlp_probs) / 2

             
            y_pred = np.argmax(avg_probs, axis=1)

             
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f"Iteration {i}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return accuracy_scores, precision_scores, recall_scores, f1_scores

    print("Training and testing with Hybrid Random Forest + MLP model")
    accs, precs, recs, f1s = train_and_test_hybrid_rf_mlp(X, y, 100)
    print(f"Average Metrics over 100 runs:")
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f}")


### XGB + MLP

def hybrid_full_mlp_xgb():
    import random
    import numpy as np
    import pandas as pd
    from imblearn.over_sampling import ADASYN
    from xgboost import XGBClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    def train_and_test_hybrid_mlp_xgb(X, y, repeat):
         
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(repeat):
            seed = 42 * i

             
            np.random.seed(seed)
            random.seed(seed)

             
            positive_idx = y[y == 1].index
            negative_idx = y[y == 0].index

             
            pos_test_idx = np.random.choice(positive_idx, 10, replace=False)
            neg_test_idx = np.random.choice(negative_idx, 10, replace=False)
            test_idx = np.concatenate([pos_test_idx, neg_test_idx])

             
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            X_remaining = X.drop(test_idx)
            y_remaining = y.drop(test_idx)

             
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"Test set class distribution (iteration {i}): {dict(zip(unique, counts))}")

             
            adasyn = ADASYN(random_state=seed)
            X_train, y_train = adasyn.fit_resample(X_remaining, y_remaining)

             
            X_train.to_csv(f'X_train_{i}.csv', index=False)
            X_test.to_csv(f'X_test_{i}.csv', index=False)
            y_train.to_csv(f'y_train_{i}.csv', index=False)
            y_test.to_csv(f'y_test_{i}.csv', index=False)

             
            xgb = XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)

            
            mlp = MLPClassifier(
                hidden_layer_sizes=(160, 80, 40, 40, 40, 40, 20, 10),
                max_iter=500,
                random_state=seed
            )
            mlp.fit(X_train, y_train)

            
            xgb_probs = xgb.predict_proba(X_test)
            mlp_probs = mlp.predict_proba(X_test)
            avg_probs = (xgb_probs + mlp_probs) / 2

             
            y_pred = np.argmax(avg_probs, axis=1)

             
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f"Iteration {i}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return accuracy_scores, precision_scores, recall_scores, f1_scores

    print("Training and testing with Hybrid MLP + XGBoost model")
    accs, precs, recs, f1s = train_and_test_hybrid_mlp_xgb(X, y, 100)
    print(f"Average Metrics over 100 runs:")
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f}")


### RF + XGB + MLP (20 data samples (13%) in the test set.)
def hybrid_full_13_percent():
    import random
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import ADASYN
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.neural_network import MLPClassifier

    def train_and_test_hybrid(X, y, repeat):
         
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(repeat):
            seed = 42 * i

             
            np.random.seed(seed)
            random.seed(seed)

             
            positive_idx = y[y == 1].index
            negative_idx = y[y == 0].index

             
            pos_test_idx = np.random.choice(positive_idx, 10, replace=False)
            neg_test_idx = np.random.choice(negative_idx, 10, replace=False)
            test_idx = np.concatenate([pos_test_idx, neg_test_idx])

             
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            X_remaining = X.drop(test_idx)
            y_remaining = y.drop(test_idx)

             
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"Test set class distribution (iteration {i}): {dict(zip(unique, counts))}")

             
            adasyn = ADASYN(random_state=seed)
            X_train, y_train = adasyn.fit_resample(X_remaining, y_remaining)

             
            X_train.to_csv(f'X_train_{i}.csv', index=False)
            X_test.to_csv(f'X_test_{i}.csv', index=False)
            y_train.to_csv(f'y_train_{i}.csv', index=False)
            y_test.to_csv(f'y_test_{i}.csv', index=False)

             
            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_train, y_train)

             
            xgb = XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)

             
            mlp = MLPClassifier(
                hidden_layer_sizes=(160, 80, 40, 40, 40, 40, 20, 10),
                max_iter=500,
                random_state=seed
            )
            mlp.fit(X_train, y_train)

             
            rf_probs = rf.predict_proba(X_test)
            xgb_probs = xgb.predict_proba(X_test)
            mlp_probs = mlp.predict_proba(X_test)

             
            avg_probs = (rf_probs + xgb_probs + mlp_probs) / 3

             
            y_pred = np.argmax(avg_probs, axis=1)

             
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'Iteration {i}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return accuracy_scores, precision_scores, recall_scores, f1_scores

    print("Training and testing with Hybrid RF + XGBoost + MLP model")
    accs, precs, recs, f1s = train_and_test_hybrid(X, y, 100)
    print(f"Average Metrics over 100 runs:")
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f}")


### RF + XGB + MLP (8 data samples (5%) in the test set.)
def hybrid_full_5_percent():
    import random
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import ADASYN
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.neural_network import MLPClassifier

    def train_and_test_hybrid(X, y, repeat):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(repeat):
            seed = 42 * i

             
            np.random.seed(seed)
            random.seed(seed)

             
            positive_idx = y[y == 1].index
            negative_idx = y[y == 0].index

             
            pos_test_idx = np.random.choice(positive_idx, 4, replace=False)
            neg_test_idx = np.random.choice(negative_idx, 4, replace=False)
            test_idx = np.concatenate([pos_test_idx, neg_test_idx])

             
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            X_remaining = X.drop(test_idx)
            y_remaining = y.drop(test_idx)

             
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"Test set class distribution (iteration {i}): {dict(zip(unique, counts))}")

             
            adasyn = ADASYN(random_state=seed)
            X_train, y_train = adasyn.fit_resample(X_remaining, y_remaining)

             
            X_train.to_csv(f'X_train_{i}.csv', index=False)
            X_test.to_csv(f'X_test_{i}.csv', index=False)
            y_train.to_csv(f'y_train_{i}.csv', index=False)
            y_test.to_csv(f'y_test_{i}.csv', index=False)

             
            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_train, y_train)

             
            xgb = XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)

             
            mlp = MLPClassifier(
                hidden_layer_sizes=(160, 80, 40, 40, 40, 40, 20, 10),
                max_iter=500,
                random_state=seed
            )
            mlp.fit(X_train, y_train)

             
            rf_probs = rf.predict_proba(X_test)
            xgb_probs = xgb.predict_proba(X_test)
            mlp_probs = mlp.predict_proba(X_test)

             
            avg_probs = (rf_probs + xgb_probs + mlp_probs) / 3

             
            y_pred = np.argmax(avg_probs, axis=1)

             
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'Iteration {i}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return accuracy_scores, precision_scores, recall_scores, f1_scores

    print("Training and testing with Hybrid RF + XGBoost + MLP model")
    accs, precs, recs, f1s = train_and_test_hybrid(X, y, 100)
    print(f"Average Metrics over 100 runs:")
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f}")




### RF + XGB + MLP: (16 samples (10%) in the test) 
def hybrid_full_10_percent():
    import random
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import ADASYN
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.neural_network import MLPClassifier

    def train_and_test_hybrid(X, y, repeat):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(repeat):
            seed = 42 * i

             
            np.random.seed(seed)
            random.seed(seed)

             
            positive_idx = y[y == 1].index
            negative_idx = y[y == 0].index

             
            pos_test_idx = np.random.choice(positive_idx, 8, replace=False)
            neg_test_idx = np.random.choice(negative_idx, 8, replace=False)
            test_idx = np.concatenate([pos_test_idx, neg_test_idx])

             
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            X_remaining = X.drop(test_idx)
            y_remaining = y.drop(test_idx)

             
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"Test set class distribution (iteration {i}): {dict(zip(unique, counts))}")

             
            adasyn = ADASYN(random_state=seed)
            X_train, y_train = adasyn.fit_resample(X_remaining, y_remaining)

             
            X_train.to_csv(f'X_train_{i}.csv', index=False)
            X_test.to_csv(f'X_test_{i}.csv', index=False)
            y_train.to_csv(f'y_train_{i}.csv', index=False)
            y_test.to_csv(f'y_test_{i}.csv', index=False)

             
            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_train, y_train)

             
            xgb = XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)

             
            mlp = MLPClassifier(
                hidden_layer_sizes=(160, 80, 40, 40, 40, 40, 20, 10),
                max_iter=500,
                random_state=seed
            )
            mlp.fit(X_train, y_train)

             
            rf_probs = rf.predict_proba(X_test)
            xgb_probs = xgb.predict_proba(X_test)
            mlp_probs = mlp.predict_proba(X_test)

             
            avg_probs = (rf_probs + xgb_probs + mlp_probs) / 3

             
            y_pred = np.argmax(avg_probs, axis=1)

             
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'Iteration {i}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return accuracy_scores, precision_scores, recall_scores, f1_scores

    print("Training and testing with Hybrid RF + XGBoost + MLP model")
    accs, precs, recs, f1s = train_and_test_hybrid(X, y, 100)
    print(f"Average Metrics over 100 runs:")
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f}")



### RF + XGB + MLP (20% data points in the test)
def hybrid_full_20_percent():
    import random
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import ADASYN
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.neural_network import MLPClassifier

    def train_and_test_hybrid(X, y, repeat):
         
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(repeat):
            seed = 42 * i

             
            np.random.seed(seed)
            random.seed(seed)

             
            positive_idx = y[y == 1].index
            negative_idx = y[y == 0].index

             
            pos_test_idx = np.random.choice(positive_idx, 16, replace=False)
            neg_test_idx = np.random.choice(negative_idx, 16, replace=False)
            test_idx = np.concatenate([pos_test_idx, neg_test_idx])

             
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            X_remaining = X.drop(test_idx)
            y_remaining = y.drop(test_idx)

             
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"Test set class distribution (iteration {i}): {dict(zip(unique, counts))}")

             
            adasyn = ADASYN(random_state=seed)
            X_train, y_train = adasyn.fit_resample(X_remaining, y_remaining)

             
            X_train.to_csv(f'X_train_{i}.csv', index=False)
            X_test.to_csv(f'X_test_{i}.csv', index=False)
            y_train.to_csv(f'y_train_{i}.csv', index=False)
            y_test.to_csv(f'y_test_{i}.csv', index=False)

             
            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_train, y_train)

             
            xgb = XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)

             
            mlp = MLPClassifier(
                hidden_layer_sizes=(160, 80, 40, 40, 40, 40, 20, 10),
                max_iter=500,
                random_state=seed
            )
            mlp.fit(X_train, y_train)

             
            rf_probs = rf.predict_proba(X_test)
            xgb_probs = xgb.predict_proba(X_test)
            mlp_probs = mlp.predict_proba(X_test)

             
            avg_probs = (rf_probs + xgb_probs + mlp_probs) / 3

             
            y_pred = np.argmax(avg_probs, axis=1)

             
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'Iteration {i}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return accuracy_scores, precision_scores, recall_scores, f1_scores

    print("Training and testing with Hybrid RF + XGBoost + MLP model")
    accs, precs, recs, f1s = train_and_test_hybrid(X, y, 100)
    print(f"Average Metrics over 100 runs:")
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f}")



### RF + XGB + MLP (30% datapoints in the test set.)

def hybrid_full_30_percent():
    import random
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import ADASYN
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.neural_network import MLPClassifier

    def train_and_test_hybrid(X, y, repeat):
         
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(repeat):
            seed = 42 * i

             
            np.random.seed(seed)
            random.seed(seed)

             
            positive_idx = y[y == 1].index
            negative_idx = y[y == 0].index

            pos_test_idx = np.random.choice(positive_idx, 24, replace=False)
            neg_test_idx = np.random.choice(negative_idx, 24, replace=False)
            test_idx = np.concatenate([pos_test_idx, neg_test_idx])

             
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            X_remaining = X.drop(test_idx)
            y_remaining = y.drop(test_idx)

             
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"Test set class distribution (iteration {i}): {dict(zip(unique, counts))}")

             
            adasyn = ADASYN(random_state=seed)
            X_train, y_train = adasyn.fit_resample(X_remaining, y_remaining)

             
            X_train.to_csv(f'X_train_{i}.csv', index=False)
            X_test.to_csv(f'X_test_{i}.csv', index=False)
            y_train.to_csv(f'y_train_{i}.csv', index=False)
            y_test.to_csv(f'y_test_{i}.csv', index=False)

             
            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_train, y_train)

             
            xgb = XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)

             
            mlp = MLPClassifier(
                hidden_layer_sizes=(160, 80, 40, 40, 40, 40, 20, 10),
                max_iter=500,
                random_state=seed
            )
            mlp.fit(X_train, y_train)

             
            rf_probs = rf.predict_proba(X_test)
            xgb_probs = xgb.predict_proba(X_test)
            mlp_probs = mlp.predict_proba(X_test)

             
            avg_probs = (rf_probs + xgb_probs + mlp_probs) / 3

             
            y_pred = np.argmax(avg_probs, axis=1)

             
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'Iteration {i}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return accuracy_scores, precision_scores, recall_scores, f1_scores

    print("Training and testing with Hybrid RF + XGBoost + MLP model")
    accs, precs, recs, f1s = train_and_test_hybrid(X, y, 100)
    print(f"Average Metrics over 100 runs:")
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f}")


### RF + XGB + MLP (1% datapoints in the test set.)
def hybrid_full_1_percent():

    import random
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import ADASYN
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.neural_network import MLPClassifier

    def train_and_test_hybrid(X, y, repeat):
         
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(repeat):
            seed = 42 * i

             
            np.random.seed(seed)
            random.seed(seed)

             
            positive_idx = y[y == 1].index
            negative_idx = y[y == 0].index

            pos_test_idx = np.random.choice(positive_idx, 1, replace=False)
            neg_test_idx = np.random.choice(negative_idx, 1, replace=False)
            test_idx = np.concatenate([pos_test_idx, neg_test_idx])

             
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            X_remaining = X.drop(test_idx)
            y_remaining = y.drop(test_idx)

             
            unique, counts = np.unique(y_test, return_counts=True)
            print(f"Test set class distribution (iteration {i}): {dict(zip(unique, counts))}")

             
            adasyn = ADASYN(random_state=seed)
            X_train, y_train = adasyn.fit_resample(X_remaining, y_remaining)

             
            X_train.to_csv(f'X_train_{i}.csv', index=False)
            X_test.to_csv(f'X_test_{i}.csv', index=False)
            y_train.to_csv(f'y_train_{i}.csv', index=False)
            y_test.to_csv(f'y_test_{i}.csv', index=False)

             
            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_train, y_train)

             
            xgb = XGBClassifier(n_estimators=100, random_state=seed, use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)

             
            mlp = MLPClassifier(
                hidden_layer_sizes=(160, 80, 40, 40, 40, 40, 20, 10),
                max_iter=500,
                random_state=seed
            )
            mlp.fit(X_train, y_train)

             
            rf_probs = rf.predict_proba(X_test)
            xgb_probs = xgb.predict_proba(X_test)
            mlp_probs = mlp.predict_proba(X_test)

             
            avg_probs = (rf_probs + xgb_probs + mlp_probs) / 3

             
            y_pred = np.argmax(avg_probs, axis=1)

             
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            print(f'Iteration {i}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return accuracy_scores, precision_scores, recall_scores, f1_scores

    print("Training and testing with Hybrid RF + XGBoost + MLP model")
    accs, precs, recs, f1s = train_and_test_hybrid(X, y, 100)
    print(f"Average Metrics over 100 runs:")
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f}")



### Now running the experiments.

hybrid_full_10_percent()
