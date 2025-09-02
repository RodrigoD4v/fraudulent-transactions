import pandas as pd
import os

def removendo_features_redundantes_e_ineficientes(df):
    print('Removendo colunas desnecessárias...')
    df = df.drop(['newbalanceOrig', 'newbalanceDest', 
                  'isFlaggedFraud', 'nameOrig', 'nameDest', 'type'], axis=1)
    return df

def criando_novas_features(df):
    print('Criando novas features...')
    df['hour'] = df['step'] % 24
    return df

def main():
    print('Iniciando a engenharia de features...')
    data_dir = os.path.join('data', 'interim')
    file_path = os.path.join(data_dir, 'Fraud_sample.parquet')

    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return

    df = criando_novas_features(df)
    df = removendo_features_redundantes_e_ineficientes(df)

    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(os.path.join(processed_dir, 'fraud_features.csv'), index=False)
    print(f"Dataset processado salvo em {processed_dir}")

if __name__ == '__main__':
    main()
