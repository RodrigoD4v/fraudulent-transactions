.PHONY: clean setup_dirs create_environment test_environment requirements data features train predict all lint help

#################################################################################
# GLOBALS
#################################################################################

PROJECT_NAME = fraudulent-transactions

# Comando Python adequado
PYTHON_INTERPRETER = python

# Caminho do Python do venv
ifeq ($(OS),Windows_NT)
    VENV_PYTHON = venv/Scripts/python.exe
else
    VENV_PYTHON = venv/bin/python
endif


#################################################################################
# COMMANDS
#################################################################################

## Criar pastas necessárias
setup_dirs:
	@mkdir -p data/raw data/interim data/processed

## Criar ambiente Python (venv)
create_environment:
	@echo ">>> CRIANDO VIRTUALENV..."
	@$(PYTHON_INTERPRETER) -m ensurepip --default-pip || echo ">>> AVISO: PIP NAO ENCONTRADO, TENTANDO INSTALAR MANUALMENTE..."
	@if [ ! -d "venv" ]; then $(PYTHON_INTERPRETER) -m venv venv; fi
	@echo ">>> VIRTUAL ENV CRIADO. USANDO $(VENV_PYTHON) PARA INSTALAR DEPENDENCIAS."

## Testar ambiente Python
test_environment: create_environment
	@echo ">>> TESTANDO VERSAO DO PYTHON NA VENV..."
	@$(VENV_PYTHON) test_environment.py || ( \
		echo ">>> ERRO: PYTHON DA VENV NAO E 3.9.19, DELETANDO VENV..."; \
		rm -rf venv; \
		exit 1; \
	)

## Instalar dependências Python e pacote local
requirements: test_environment
	@echo ">>> INSTALANDO DEPENDENCIAS COM $(VENV_PYTHON)..."
	@$(VENV_PYTHON) -m pip install -U pip setuptools wheel
	@$(VENV_PYTHON) -m pip install -r requirements.txt
	@$(VENV_PYTHON) -m pip install -e .

## Processar dados
data: setup_dirs requirements
	@if [ ! -f "data/raw/Fraud.csv" ]; then \
		echo ">>> ERRO: COLOQUE O ARQUIVO Fraud.csv EM data/raw/ ANTES DE RODAR O MAKEFILE'."; \
		exit 1; \
	fi
	@$(VENV_PYTHON) src/data/make_dataset.py data/raw/Fraud.csv data/interim/Fraud_sample.parquet --nrows 500000

## Construir features
features: data
	@$(VENV_PYTHON) src/features/build_features.py

## Treinar modelo
train: features
	@$(VENV_PYTHON) src/models/train_model.py

## Gerar previsoes
predict: train
	@$(VENV_PYTHON) src/models/predict_model.py

## Rodar pipeline completo
all: setup_dirs create_environment test_environment requirements data features train predict

## Apagar arquivos compilados Python
clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete

## Lint no codigo
lint:
	@flake8 src

#################################################################################
# Self Documenting Commands
#################################################################################

.DEFAULT_GOAL := help

help:
	@echo "$$(tput bold)COMANDOS DISPONIVEIS:$$(tput sgr0)"
	@grep -E '^##' Makefile | sed -e 's/## //'
