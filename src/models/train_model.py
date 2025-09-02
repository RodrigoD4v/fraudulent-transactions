import joblib
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
data_path = os.path.join(project_dir, 'data', 'processed', 'fraud_features.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"O arquivo não foi encontrado: {data_path}")

df = pd.read_csv(data_path)

X = df.drop(columns=['isFraud'])
y = df['isFraud']

# Função para CV de modelos de ÁRVORE
def cross_validate_tree_model(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    precisions_macro, recalls_macro, f1s_macro = [], [], []
    precisions_weighted, recalls_weighted, f1s_weighted = [], [], []
    conf_matrices = []
    
    cat_cols = ["type"]

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        target_means = {col: y_tr.groupby(X_tr[col]).mean() for col in cat_cols}
        
        X_tr_processed = X_tr.copy()
        X_te_processed = X_te.copy()

        for col in cat_cols:
            X_tr_processed[col] = X_tr_processed[col].map(target_means[col])
            X_te_processed[col] = X_te_processed[col].map(target_means[col]).fillna(y_tr.mean())

        model.fit(X_tr_processed, y_tr)
        y_pred = model.predict(X_te_processed)

        precisions_macro.append(precision_score(y_te, y_pred, average='macro'))
        recalls_macro.append(recall_score(y_te, y_pred, average='macro'))
        f1s_macro.append(f1_score(y_te, y_pred, average='macro'))

        precisions_weighted.append(precision_score(y_te, y_pred, average='weighted'))
        recalls_weighted.append(recall_score(y_te, y_pred, average='weighted'))
        f1s_weighted.append(f1_score(y_te, y_pred, average='weighted'))

        conf_matrices.append(confusion_matrix(y_te, y_pred))

    print(f"Precision macro: {np.mean(precisions_macro):.4f}")
    print(f"Recall macro   : {np.mean(recalls_macro):.4f}")
    print(f"F1-score macro : {np.mean(f1s_macro):.4f}")
    print(f"Precision weighted: {np.mean(precisions_weighted):.4f}")
    print(f"Recall weighted   : {np.mean(recalls_weighted):.4f}")
    print(f"F1-score weighted : {np.mean(f1s_weighted):.4f}")
    print("Matriz de Confusão média (soma de folds):")
    print(np.sum(conf_matrices, axis=0))
    print("\n" + "-"*50 + "\n")

# Função para CV de modelo LINEAR
def cross_validate_linear_model(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    precisions_macro, recalls_macro, f1s_macro = [], [], []
    precisions_weighted, recalls_weighted, f1s_weighted = [], [], []
    conf_matrices = []
    
    cat_cols = ["type"]

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        X_tr_processed = pd.get_dummies(X_tr, columns=cat_cols, drop_first=True)
        X_te_processed = pd.get_dummies(X_te, columns=cat_cols, drop_first=True)
        
        X_te_processed = X_te_processed.reindex(columns=X_tr_processed.columns, fill_value=0)

        model.fit(X_tr_processed, y_tr)
        y_pred = model.predict(X_te_processed)

        precisions_macro.append(precision_score(y_te, y_pred, average='macro'))
        recalls_macro.append(recall_score(y_te, y_pred, average='macro'))
        f1s_macro.append(f1_score(y_te, y_pred, average='macro'))

        precisions_weighted.append(precision_score(y_te, y_pred, average='weighted'))
        recalls_weighted.append(recall_score(y_te, y_pred, average='weighted'))
        f1s_weighted.append(f1_score(y_te, y_pred, average='weighted'))

        conf_matrices.append(confusion_matrix(y_te, y_pred))

    print(f"Precision macro: {np.mean(precisions_macro):.4f}")
    print(f"Recall macro   : {np.mean(recalls_macro):.4f}")
    print(f"F1-score macro : {np.mean(f1s_macro):.4f}")
    print(f"Precision weighted: {np.mean(precisions_weighted):.4f}")
    print(f"Recall weighted   : {np.mean(recalls_weighted):.4f}")
    print(f"F1-score weighted : {np.mean(f1s_weighted):.4f}")
    print("Matriz de Confusão média (soma de folds):")
    print(np.sum(conf_matrices, axis=0))
    print("\n" + "-"*50 + "\n")

# Executando modelos
print("\n--- Treinando modelos baseados em árvore ---")
models_tree = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(class_weight='balanced', random_state=42),
        random_state=42
    ),
    "XGBoost": XGBClassifier(scale_pos_weight=(y==0).sum()/(y==1).sum(), use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42)
}

for name, model in models_tree.items():
    print(f"### {name} ###")
    cross_validate_tree_model(model, X, y)

print("\n--- Treinando modelo linear ---")
lgr = LogisticRegression(class_weight='balanced', random_state=42, solver='liblinear')
print(f"### Regressão Logística ###")
cross_validate_linear_model(lgr, X, y)

# Salvando modelo com melhor peformace
print("\n--- Treinando e salvando o modelo de melhor desempenho (XGBoost) ---")

best_model = XGBClassifier(
    scale_pos_weight=(y==0).sum()/(y==1).sum(), 
    use_label_encoder=False, 
    eval_metric='logloss', 
    random_state=42
)

X_processed = X.copy()
cat_cols = ["type"]

target_means = {col: y.groupby(X_processed[col]).mean() for col in cat_cols}
for col in cat_cols:
    X_processed[col] = X_processed[col].map(target_means[col])

print("Treinando o modelo final...")
best_model.fit(X_processed, y)
print("Treinamento concluído.")

model_path = os.path.join(project_dir, 'models', 'xgboost_fraud_model.joblib')

print(f"Salvando o modelo em: {model_path}")
joblib.dump(best_model, model_path)
print("Modelo salvo com sucesso.")

target_means_path = os.path.join(project_dir, 'models', 'target_means.joblib')
joblib.dump(target_means, target_means_path)
print("Mapeamento de médias salvas com sucesso.")