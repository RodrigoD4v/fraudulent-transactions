import pandas as pd
import os

def removendo_features_redundantes_e_ineficientes(df):
  """
  Remove as colunas com alta multicolinearidade e as ineficazes.
  """
  print('Removendo colunas desnecessárias...')
  df = df.drop(['newbalanceOrig', 'newbalanceDest', 'isFlaggedFraud', 'nameOrig', 'nameDest'], axis = 1)
  return df

def criando_novas_features(df):
  """
  Cria novas features a partir das colunas existentes.
  """
  print('Criando novas features...')
    
  # 1. Feature de 'hora' do dia
  df['hour'] = df['step'] % 24
    
  # 2. Feature de 'diferença de saldo na origem'
  df['balance_diff_orig'] = (df['oldbalanceOrg'] - df['newbalanceOrig']) - df['amount']
    
  # 3. Feature para identificar se a conta de destino é um comerciante
  df['is_merchant_dest'] = df['nameDest'].str.startswith('M').astype(int)
    
  return df
  
def target_encoding(df, target_col='isFraud', categorical_cols=['type']):
  """
  Aplica Target Encoding nas colunas categóricas.
  Para modelos baseados em árvores.
  """
  print('Aplicando Target Encoding...')
  
  df_encoded = df.copy()
  
  for col in categorical_cols:
     # Calcula a média da variável alvo para cada categoria
      mean_by_category = df_encoded.groupby(col)[target_col].mean()
      # Mapeia os valores para a nova feature
      df_encoded[f'{col}_target_encoded'] = df_encoded[col].map(mean_by_category)
      # Remove a coluna original
      df_encoded = df_encoded.drop(col, axis=1)
  return df_encoded

def one_hot_encoding(df, categorical_cols=['type']):
  """
  Aplica One-Hot Encoding nas colunas categóricas.
  Para modelos lineares como Regressão Logística.
  """
  print('Aplicando One-Hot Encoding...')
  
  df_encoded = df.copy()
  
  # Usa pd.get_dummies para converter categorias em colunas binárias
  df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
  return df_encoded

def main():
  """
  Pipeline principal de engenharia de features.
  """
  print('Iniciando a engenharia de features...')
  data_dir = os.path.join('data', 'interim')
  file_path = os.path.join(data_dir, 'Fraud_sample.parquet')
  
  try:
    df = pd.read_parquet(file_path)
  except FileNotFoundError:
    print(f"Erro: Arquivo {file_path} não encontrado. Verifique o caminho.")
    return

# --- Pipeline para modelos insensíveis à escala (Árvores, Boosting) ---
  df_tree_based = df.copy()
    
  df_tree_based = criando_novas_features(df_tree_based)
  df_tree_based = removendo_features_redundantes_e_ineficientes(df_tree_based)
  df_tree_based = target_encoding(df_tree_based, target_col='isFraud', categorical_cols=['type'])
    
  processed_dir_tree = os.path.join('data', 'processed', 'tree_based')
  os.makedirs(processed_dir_tree, exist_ok=True)
  df_tree_based.to_csv(os.path.join(processed_dir_tree, 'processed_tree_based.csv'), index=False)
  print(f"Dataset para modelos de árvore salvo em {processed_dir_tree}")

  # --- Pipeline para modelos sensíveis à escala (Regressão Logística) ---
  df_linear_based = df.copy()
    
  df_linear_based = criando_novas_features(df_linear_based)
  df_linear_based = removendo_features_redundantes_e_ineficientes(df_linear_based)
  df_linear_based = one_hot_encoding(df_linear_based, categorical_cols=['type'])
    
  processed_dir_linear = os.path.join('data', 'processed', 'linear_based')
  os.makedirs(processed_dir_linear, exist_ok=True)
  df_linear_based.to_csv(os.path.join(processed_dir_linear, 'processed_linear_based.csv'), index=False)
  print(f"Dataset para modelos lineares salvo em {processed_dir_linear}")
    
if __name__ == '__main__':
    main()