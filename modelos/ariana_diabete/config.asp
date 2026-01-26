{
  "asp": {
    "versao": "1.0",
    "descricao": "Modelo ASP treinado",
    "data_criacao": "Jan 26 2026 11:02:47"
  },
  "arquitetura": {
    "nome": "ASP-MLP",
    "camadas": 5,
    "neuronios": [8, 64, 32, 16, 1],
    "ativacao": "relu",
    "parametros_totais": 3201
  },
  "treino": {
    "taxa_aprendizado": 0.010000,
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
      "saida": 64,
      "pesos": 512,
      "biases": 64
    },
    {
      "id": 1,
      "tipo": "densa",
      "entrada": 64,
      "saida": 32,
      "pesos": 2048,
      "biases": 32
    },
    {
      "id": 2,
      "tipo": "densa",
      "entrada": 32,
      "saida": 16,
      "pesos": 512,
      "biases": 16
    },
    {
      "id": 3,
      "tipo": "densa",
      "entrada": 16,
      "saida": 1,
      "pesos": 16,
      "biases": 1
    }
  ]
}
