"""
Machine learning model training and evaluation with automatic feature selection
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

def build_classification_models(df):
    """
    Build and evaluate multiple classification models with automatic feature selection
    """
    
    if df is None:
        print("[ERROR] No dataset provided.")
        return None
    
    print("\n=== MODEL TRAINING & EVALUATION ===")
    
    # [1] Dataset preparation
    print("\n[1] Preparing dataset")
    modeling_df = df[df['sentiment_binary'].notna()].copy()
    print(f" -> Samples with target: {len(modeling_df):,}")
    if len(modeling_df) == 0:
        print("[ERROR] No samples with target variable.")
        return None
    
    # Exclude IDs and timestamps
    exclude = ['sentiment_binary', 'review_id', 'order_id', 'customer_unique_id', 'product_id', 'review_score']
    potential_features = [c for c in modeling_df.columns if c not in exclude]
    datetime_cols = [c for c in potential_features if pd.api.types.is_datetime64_any_dtype(modeling_df[c])]
    potential_features = [c for c in potential_features if c not in datetime_cols]
    if datetime_cols:
        print(f" -> Excluded datetime cols: {datetime_cols}")
    print(f" -> Potential features: {len(potential_features)}")
    
    # [2] Feature typing
    print("\n[2] Feature typing")
    num_features, cat_features = [], []
    for f in potential_features:
        if modeling_df[f].dtype in ['int64', 'float64', 'int32', 'float32']:
            if modeling_df[f].nunique() > 10 or 'float' in str(modeling_df[f].dtype):
                num_features.append(f)
            else:
                cat_features.append(f)
        else:
            cat_features.append(f)
    print(f" -> Numerical: {len(num_features)} | Categorical: {len(cat_features)}")
    
    # [3] Missing value handling
    print("\n[3] Handling missing values")
    for f in num_features:
        if modeling_df[f].isnull().sum() > 0:
            val = modeling_df[f].median()
            modeling_df[f] = modeling_df[f].fillna(val)
            print(f" -> {f}: filled with median ({val:.2f})")
    for f in cat_features:
        if modeling_df[f].isnull().sum() > 0:
            val = modeling_df[f].mode().iloc[0] if not modeling_df[f].mode().empty else "Unknown"
            modeling_df[f] = modeling_df[f].fillna(val)
            print(f" -> {f}: filled with mode ({val})")
    
    # [4] Encoding categorical features
    print("\n[4] Encoding categorical features")
    label_encoders, enc_features = {}, []
    for f in cat_features:
        le = LabelEncoder()
        try:
            modeling_df[f"{f}_encoded"] = le.fit_transform(modeling_df[f].astype(str))
            label_encoders[f] = le
            enc_features.append(f"{f}_encoded")
            print(f" -> Encoded {f}")
        except Exception as e:
            print(f" [Warning] Could not encode {f}: {e}")
    
    # Combine features
    all_features = num_features + enc_features
    X, y = modeling_df[all_features].copy(), modeling_df['sentiment_binary']
    
    print("\n[5] Dataset summary")
    print(f" -> Samples: {len(X):,}, Features: {len(all_features)}")
    print(f" -> Positive: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
    print(f" -> Negative: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    
    # [6] Low-variance feature removal
    print("\n[6] Removing low-variance features")
    low_var = X.var(numeric_only=True)[X.var(numeric_only=True) < 0.01].index.tolist()
    if low_var:
        X = X.drop(columns=low_var)
        all_features = [f for f in all_features if f not in low_var]
        print(f" -> Removed {len(low_var)} features")
    else:
        print(" -> None found")
    
    # [7] Train-test split
    print("\n[7] Splitting dataset")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f" -> Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # [8] Scaling features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    print("\n[8] Features scaled")
    
    # [9] Automatic feature selection
    print("\n[9] Selecting features (SelectKBest)")
    k = min(len(X_train.columns), 25)
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = pd.DataFrame(selector.fit_transform(X_train_scaled, y_train),
                               columns=X_train_scaled.columns[selector.get_support()],
                               index=X_train.index)
    X_test_sel = pd.DataFrame(selector.transform(X_test_scaled),
                              columns=X_train_sel.columns, index=X_test.index)
    selected = list(X_train_sel.columns)
    print(f" -> Selected top {len(selected)} features")
    
    # [10] Train models
    print("\n[10] Training models")
    models = {
        "Logistic Regression": LogisticRegression(C=0.1, penalty="l2", solver="liblinear",
                                                  random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=15,
                                                min_samples_split=7, min_samples_leaf=1,
                                                random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                        max_depth=4, subsample=0.9, random_state=42)
    }
    
    results, feature_importance = {}, {}
    for name, model in models.items():
        print(f" -> {name}")
        if name == "Logistic Regression":
            model.fit(X_train_sel, y_train)
            y_pred, y_prob = model.predict(X_test_sel), model.predict_proba(X_test_sel)[:,1]
            importance = pd.Series(np.abs(model.coef_[0]), index=selected).sort_values(ascending=False)
        else:
            model.fit(X_train[selected], y_train)
            y_pred, y_prob = model.predict(X_test[selected]), model.predict_proba(X_test[selected])[:,1]
            importance = pd.Series(model.feature_importances_, index=selected).sort_values(ascending=False)
        
        results[name] = {
            "predictions": y_pred,
            "probabilities": y_prob,
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        feature_importance[name] = importance
        print(f"    Accuracy: {results[name]['accuracy']:.4f}")
        print("    Top 5 features:")
        for i, (f, v) in enumerate(importance.head(5).items(), 1):
            print(f"      {i}. {f}: {v:.4f}")
    
    # [11] Select best model
    best = max(results, key=lambda m: results[m]['classification_report']['macro avg']['f1-score'])
    print(f"\n[11] Best model: {best}")
    print(f" -> Macro F1: {results[best]['classification_report']['macro avg']['f1-score']:.4f}")
    print(f" -> Accuracy: {results[best]['accuracy']:.4f}")
    
    print("\n=== MODEL TRAINING COMPLETE ===")
    return {
        "models": models,
        "results": results,
        "best_model": models[best],
        "best_model_name": best,
        "selected_features": selected,
        "feature_importance": feature_importance,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "test_predictions": results[best]["predictions"],
        "test_probabilities": results[best]["probabilities"]
    }
