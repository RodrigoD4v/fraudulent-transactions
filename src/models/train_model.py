import joblib
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")
from src.visualization import plot_multiple_confusion_matrices, plot_multiple_feature_importances, plot_roc_comparison, plot_precision_recall_comparison

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
data_path = os.path.join(project_dir, 'data', 'processed', 'fraud_features.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"O arquivo não foi encontrado: {data_path}")

df = pd.read_csv(data_path)

X = df.drop(columns=['isFraud'])
y = df['isFraud']

# Cria a pasta para salvar os gráficos, se não existir
plots_dir = os.path.join(project_dir, 'reports', 'figures')
os.makedirs(plots_dir, exist_ok=True)

# Função de Validação Cruzada Unificada
def cross_validate_model(model, X, y, model_name, n_splits=5):
    """
    Executa a validação cruzada e coleta os dados para plotagem.
    Retorna os dados necessários para os gráficos.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    precisions_macro, recalls_macro, f1s_macro = [], [], []
    precisions_weighted, recalls_weighted, f1s_weighted = [], [], []
    
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        
        try:
            y_proba = model.predict_proba(X_te)[:, 1]
        except AttributeError:
            y_proba = None
        
        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred)
        if y_proba is not None:
            all_y_proba.extend(y_proba)

        precisions_macro.append(precision_score(y_te, y_pred, average='macro', zero_division=0))
        recalls_macro.append(recall_score(y_te, y_pred, average='macro', zero_division=0))
        f1s_macro.append(f1_score(y_te, y_pred, average='macro', zero_division=0))
        
        precisions_weighted.append(precision_score(y_te, y_pred, average='weighted'))
        recalls_weighted.append(recall_score(y_te, y_pred, average='weighted'))
        f1s_weighted.append(f1_score(y_te, y_pred, average='weighted'))

    print(f"Precision macro: {np.mean(precisions_macro):.4f}")
    print(f"Recall macro   : {np.mean(recalls_macro):.4f}")
    print(f"F1-score macro : {np.mean(f1s_macro):.4f}")
    print(f"Precision weighted: {np.mean(precisions_weighted):.4f}")
    print(f"Recall weighted   : {np.mean(recalls_weighted):.4f}")
    print(f"F1-score weighted : {np.mean(f1s_weighted):.4f}")
    print("\n" + "-"*50 + "\n")
    
    # Retorna todos os dados para plotagem
    return np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba)

# Dicionários para armazenar os resultados para os gráficos de comparação
all_roc_results = {}
all_pr_results = {}
all_cm_results = {}
all_fi_results = {}

# Executando modelos
print("\nAvaliando e Gerando Gráficos para os Modelos")
models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(class_weight='balanced', random_state=42),
        random_state=42
    ),
    "XGBoost": XGBClassifier(scale_pos_weight=(y==0).sum()/(y==1).sum(), use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42),
    "Regressão Logística": LogisticRegression(class_weight='balanced', random_state=42, solver='liblinear')
}

for name, model in models.items():
    print(f"### {name} ###")
    y_true, y_pred, y_proba = cross_validate_model(model, X, y, name)
    
    all_cm_results[name] = (y_true, y_pred)
    
    if y_proba.size > 0:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        all_roc_results[name] = (fpr, tpr, roc_auc)

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        all_pr_results[name] = (precision, recall, avg_precision)
    
    # Lógica para coletar a importância das features para diferentes tipos de modelos
    if hasattr(model, 'feature_importances_'):
        all_fi_results[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        all_fi_results[name] = np.abs(model.coef_)[0]
    else:
        all_fi_results[name] = None
        
print("Todos os modelos avaliados. Gerando gráficos finais.")

plot_multiple_confusion_matrices(all_cm_results, class_names=['Não Fraude', 'Fraude'],
                                 save_path=os.path.join(plots_dir, 'all_confusion_matrices.png'))

plot_multiple_feature_importances(all_fi_results, X.columns, top_n=10,
                                  save_path=os.path.join(plots_dir, 'all_feature_importances.png'))

plot_roc_comparison(all_roc_results, save_path=os.path.join(plots_dir, 'roc_comparison.png'))

plot_precision_recall_comparison(all_pr_results, save_path=os.path.join(plots_dir, 'pr_comparison.png'))

# Salvando modelo com melhor performance
print("\nTreinando e salvando o modelo de melhor desempenho (XGBoost)")
best_model = XGBClassifier(
    scale_pos_weight=(y==0).sum()/(y==1).sum(), 
    use_label_encoder=False, 
    eval_metric='logloss', 
    random_state=42
)
print("Treinando o modelo final...")
best_model.fit(X, y)
print("Treinamento concluído.")
model_path = os.path.join(project_dir, 'models', 'xgboost_fraud_model.joblib')
print(f"Salvando o modelo em: {model_path}")
joblib.dump(best_model, model_path)
print("Modelo salvo com sucesso.")