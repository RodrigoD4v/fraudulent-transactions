# Resumo das Métricas do Modelo XGBoost

Este arquivo resume as métricas de desempenho do modelo XGBoost, avaliado no dataset completo de 6.4 milhões de transações.

---

## Métricas de Desempenho
- **Precision (Macro):** 0.6748
- **Recall (Macro):** 0.9196
- **F1-score (Macro):** 0.7465
---

## Análise da Matriz de Confusão
A matriz de confusão abaixo mostra a performance do modelo na identificação de fraudes em larga escala.

|              | Previsão: Não Fraude | Previsão: Fraude |
|--------------|----------------------|------------------|
| **Verdadeiro: Não Fraude** | 6.341.565            | 12.842           |
| **Verdadeiro: Fraude** | 1.304                | 6.909            |

- **Falsos Negativos:** 1.304 fraudes não foram detectadas, resultando em uma taxa de 15,9%.
- **Falsos Positivos:** 12.842 transações normais foram classificadas erroneamente como fraude, resultando em uma taxa de 0,20%.

---

## Curvas de Desempenho
- **Área sob a Curva ROC (AUC):** 0.99
- **Área sob a Curva Precision-Recall (AP):** 0.80

A consistência desses resultados com os da validação cruzada indica que o modelo generaliza bem e não sofre de superajuste.