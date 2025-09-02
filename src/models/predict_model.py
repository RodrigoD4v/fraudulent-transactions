import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from src.features import (
    criando_novas_features, 
    removendo_features_redundantes_e_ineficientes
)
# Importa as funções de visualização do seu módulo
from src.visualization.visualize import (
    plot_multiple_confusion_matrices, 
    plot_multiple_feature_importances, 
    plot_roc_comparison, 
    plot_precision_recall_comparison
)

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
model_path = os.path.join(project_dir, 'models', 'xgboost_fraud_model.joblib')
raw_data_path = os.path.join(project_dir, 'data', 'raw', 'Fraud.csv') 
reports_dir = os.path.join(project_dir, 'reports', 'figures')

# Cria o diretório de relatórios se não existir
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

try:
    print("Carregando o modelo...")
    model = joblib.load(model_path)
    print("Modelo carregado com sucesso.")
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Erro: Arquivo do modelo não encontrado. Verifique se 'train_model.py' foi executado "
        f"e salvou o modelo em {os.path.join(project_dir, 'models')}."
    ) from e

def preprocess_for_prediction(df_new):
    """
    Aplica o pipeline completo de pré-processamento e engenharia de features.
    Não é necessário codificação, pois a feature 'type' foi removida.
    """
    df_processed = df_new.copy()
    
    # Aplica as funções de feature engineering
    df_processed = criando_novas_features(df_processed)
    df_processed = removendo_features_redundantes_e_ineficientes(df_processed)
    
    return df_processed

# Lógica Principal para Previsão e Avaliação em Dados
print(f"Carregando e processando o dataset completo do caminho: {raw_data_path}")

all_true_labels = []
all_predictions = []
all_probabilities = []

try:
    chunk_size = 500000 
    
    for chunk in pd.read_csv(raw_data_path, chunksize=chunk_size):
        if 'isFraud' not in chunk.columns:
            print("Aviso: A coluna 'isFraud' não foi encontrada no dataset. A avaliação de métricas não será possível.")
            break

        true_labels = chunk['isFraud']
        chunk_to_predict = chunk.drop('isFraud', axis=1)

        processed_chunk = preprocess_for_prediction(chunk_to_predict)
        
        predictions = model.predict(processed_chunk)
        probabilities = model.predict_proba(processed_chunk)[:, 1]

        all_true_labels.extend(true_labels)
        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)

    print("\nPrevisão e acumulação de resultados concluídas em todo o dataset.")

    y_true = np.array(all_true_labels)
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probabilities)
    
    feature_names = processed_chunk.columns.tolist()

    # Cálculo e Exibição das Métricas
    print("\nMétricas de Avaliação do Modelo no Dataset Completo")
    
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall    (Macro): {recall:.4f}")
    print(f"F1-score  (Macro): {f1:.4f}")
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nMatriz de Confusão:")
    print(conf_matrix)
    print("(A ordem das classes é: [Não Fraude, Fraude])")

    # Geração de Gráficos de Visualização para o  modelo
    print("\nGerando Gráficos de Visualização do Modelo")
    
    # Matriz de Confusão
    plot_multiple_confusion_matrices(
        {'XGBoost': (y_true, y_pred)},
        class_names=['Não Fraude', 'Fraude'],
        save_path=os.path.join(reports_dir, 'predict_confusion_matrix.png')
    )
    
    # Importância de Features
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)[0]
    else:
        importances = None
    
    plot_multiple_feature_importances(
        {'XGBoost': importances},
        feature_names=feature_names,
        top_n=15,
        save_path=os.path.join(reports_dir, 'predict_feature_importances.png')
    )
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plot_roc_comparison(
        {'XGBoost': (fpr, tpr, roc_auc)},
        save_path=os.path.join(reports_dir, 'predict_roc_curve.png')
    )
    
    # Curva Precision-Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    plot_precision_recall_comparison(
        {'XGBoost': (precision_vals, recall_vals, avg_precision)},
        save_path=os.path.join(reports_dir, 'predict_precision_recall_curve.png')
    )

except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em {raw_data_path}. Verifique o caminho.")
except Exception as e:
    print(f"Ocorreu um erro durante o processamento: {e}")