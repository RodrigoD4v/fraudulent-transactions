import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_multiple_confusion_matrices(model_results, class_names, save_path=None):
    """
    Plota as matrizes de confusão de múltiplos modelos em um único grid.
    O número de colunas é ajustado automaticamente.
    model_results é um dicionário no formato {'Nome do Modelo': (y_true, y_pred)}
    """
    n_models = len(model_results)
    
    # Define o número de colunas baseado na quantidade de modelos
    if n_models <= 3:
        n_cols = n_models
    else:
        n_cols = 3  # Mantém no máximo 3 colunas para gráficos maiores

    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Aumenta a legibilidade em caso de um único modelo
    if n_models == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, (model_name, (y_true, y_pred)) in enumerate(model_results.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f'Matriz de Confusão: {model_name}')
        axes[i].set_xlabel('Previsão')
        axes[i].set_ylabel('Verdadeiro')

    # Desativa os subplots não utilizados
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_multiple_feature_importances(model_importances, feature_names, top_n=10, save_path=None):
    """
    Plota a importância das features de múltiplos modelos em um único grid.
    O número de colunas é ajustado automaticamente.
    model_importances é um dicionário no formato {'Nome do Modelo': importances}
    """
    n_models = len(model_importances)
    
    if n_models <= 3:
        n_cols = n_models
    else:
        n_cols = 3

    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    if n_models == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, (model_name, importances) in enumerate(model_importances.items()):
        if importances is None:
            axes[i].axis('off')
            continue

        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        sorted_indices = np.argsort(importances)[::-1]
        top_indices = sorted_indices[:top_n]
        
        sns.barplot(x=importances[top_indices], y=np.array(feature_names)[top_indices],
                    ax=axes[i], palette='viridis')
        
        axes[i].set_title(f'Importância: {model_name}')
        axes[i].set_xlabel('Importância')
        axes[i].set_ylabel('Features')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_comparison(model_results, save_path=None):
    """
    Plota as curvas ROC de múltiplos modelos em um único gráfico.
    model_results é um dicionário no formato {'Nome do Modelo': (fpr, tpr, roc_auc)}
    """
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    
    for model_name, (fpr, tpr, roc_auc) in model_results.items():
        plt.plot(fpr, tpr, label=f'{model_name} (Área = {roc_auc:.2f})')
        
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    
    if len(model_results) == 1:
        plt.title(f'Curva ROC: {list(model_results.keys())[0]}')
    else:
        plt.title('Comparação de Curvas ROC de Múltiplos Modelos')
    
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_precision_recall_comparison(model_results, save_path=None):
    """
    Plota as curvas Precision-Recall de múltiplos modelos em um único gráfico.
    model_results é um dicionário no formato {'Nome do Modelo': (precision, recall, avg_precision)}
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, (precision, recall, avg_precision) in model_results.items():
        plt.plot(recall, precision, label=f'{model_name} (Área = {avg_precision:.2f})')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    if len(model_results) == 1:
        plt.title(f'Curva Precision-Recall: {list(model_results.keys())[0]}')
    else:
        plt.title('Comparação de Curvas Precision-Recall de Múltiplos Modelos')
    
    plt.legend(loc="lower left")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()