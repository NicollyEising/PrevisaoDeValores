# PrevisaoDeValores

## Descrição
Este projeto implementa um sistema de previsão diária do índice Bovespa (^BVSP) utilizando três abordagens de Machine Learning:
  - Rede LSTM (Long Short-Term Memory)
  - XGBoost (Extreme Gradient Boosting)
  - Ensemble com RandomForest em meta-features

## Trello
[Acesse o quadro do Trello aqui](https://trello.com/b/VjS7WFgy/previsaodevalores)

## Instruções de uso

1. **Rodar o backend**  
   No diretório raiz do projeto, execute:  
   ```bash
   uvicorn api:app --reload

2. **Rodar o frontend**  
   Navegue até o diretório do frontend e inicie a aplicação:  
   ```bash
   cd frontend
   npm start
