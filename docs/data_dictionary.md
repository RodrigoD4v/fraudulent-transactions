# Dicionário de Dados

Este documento descreve as features presentes no dataset de transações financeiras utilizado no projeto.

| Feature               | Descrição                                                                            |
|-----------------------|--------------------------------------------------------------------------------------|
| `step`                | Mapeia uma unidade de tempo no mundo real, onde cada `step` equivale a uma hora.       |
| `type`                | O tipo de transação (e.g., 'CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'). Este campo não foi usado no modelo final. |
| `amount`              | O valor monetário da transação.                                                        |
| `nameOrig`            | O ID do cliente que iniciou a transação. Este campo não foi usado no modelo final.                                             |
| `oldbalanceOrg`       | O saldo da conta do cliente que originou a transação antes da operação.              |
| `newbalanceOrig`      | O novo saldo da conta do cliente que originou a transação após a operação. Este campo não foi usado no modelo final.          |
| `nameDest`            | O ID do destinatário da transação. Este campo não foi usado no modelo final.                                                 |
| `oldbalanceDest`      | O saldo da conta do destinatário antes da transação.                                 |
| `newbalanceDest`      | O novo saldo da conta do destinatário após a transação. Este campo não foi usado no modelo final.                              |
| `isFraud`             | **(Rótulo)** `1` para transações fraudulentas, `0` caso contrário.                  |
| `isFlaggedFraud`      | **(Rótulo)** `1` se o sistema original marcou a transação como suspeita. Este campo não foi usado para o treinamento. |
| `hour`                | **(Feature Criada)** A hora do dia da transação, variando de 0 (meia-noite) a 23.  |