{
  "asp": {
    "versao": "1.0",
    "descricao": "Modelo ASP treinado",
    "data_criacao": "Jan 26 2026 11:02:47"
  },
  "arquitetura": {
    "nome": "ASP-MLP",
    "camadas": 3,
    "neuronios": [8, 95, 1],
    "ativacao": "sigmoid",
    "parametros_totais": 951
  },
  "treino": {
    "taxa_aprendizado": 0.001000,
    "inicializacao": "Xavier"
  },
  "arquivos": {
    "pesos": "pesos.asp",
    "config": "config.asp"
  },
  "camadas_detalhadas": [
    {
      "id": 0,
      "tipo": "densa",
      "entrada": 8,
      "saida": 95,
      "pesos": 760,
      "biases": 95
    },
    {
      "id": 1,
      "tipo": "densa",
      "entrada": 95,
      "saida": 1,
      "pesos": 95,
      "biases": 1
    }
  ]
}
