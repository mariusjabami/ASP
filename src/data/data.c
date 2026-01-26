#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"

Dataset carregar_csv(const char* caminho, int tem_cabecalho) {
    FILE* f = fopen(caminho, "r");
    Dataset dados = {0};
    
    if (!f) {
        printf("ERRO: Não abriu %s\n", caminho);
        return dados;
    }
    
    char linha[1024];
    int capacidade = 100;
    dados.X = malloc(capacidade * sizeof(float*));
    dados.y = malloc(capacidade * sizeof(float*));
    
    int primeira_linha = 1;
    
    while (fgets(linha, sizeof(linha), f)) {
        // Pula cabeçalho
        if (tem_cabecalho && primeira_linha) {
            primeira_linha = 0;
            continue;
        }
        
        // Remove newline
        linha[strcspn(linha, "\n")] = 0;
        
        if (dados.samples >= capacidade) {
            capacidade *= 2;
            dados.X = realloc(dados.X, capacidade * sizeof(float*));
            dados.y = realloc(dados.y, capacidade * sizeof(float*));
        }
        
        // Primeira linha: conta colunas
        if (dados.features == 0) {
            char* token = strtok(linha, ",");
            while (token) {
                dados.features++;
                token = strtok(NULL, ",");
            }
            dados.outputs = 1;
            dados.features--; // última coluna é y
            continue; // volta a ler esta linha como dados
        }
        
        // Lê dados
        dados.X[dados.samples] = malloc(dados.features * sizeof(float));
        dados.y[dados.samples] = malloc(dados.outputs * sizeof(float));
        
        char* token = strtok(linha, ",");
        for (int i = 0; i < dados.features && token; i++) {
            dados.X[dados.samples][i] = atof(token);
            token = strtok(NULL, ",");
        }
        
        if (token) {
            dados.y[dados.samples][0] = atof(token);
        }
        
        dados.samples++;
    }
    
    fclose(f);
    
    if (dados.samples > 0) {
        printf("CSV: %d amostras, %d features\n", dados.samples, dados.features);
    } else {
        printf("AVISO: CSV vazio\n");
        free(dados.X);
        free(dados.y);
        dados.X = NULL;
        dados.y = NULL;
    }
    
    return dados;
}

void liberar_dataset(Dataset* dados) {  // NOME CORRETO
    if (!dados || !dados->X) return;
    
    for (int i = 0; i < dados->samples; i++) {
        free(dados->X[i]);
        free(dados->y[i]);
    }
    
    free(dados->X);
    free(dados->y);
    
    dados->X = NULL;
    dados->y = NULL;
    dados->samples = 0;
    dados->features = 0;
    dados->outputs = 0;
}
