#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include "asp.h"
#include "../math/math.h"

// ==================== UTILS ====================
static float ativar(float x, const char* tipo) {
    if (strcmp(tipo, "sigmoid") == 0) return asp_sigmoid(x);
    if (strcmp(tipo, "degrau") == 0) return asp_degrau(x);
    if (strcmp(tipo, "relu") == 0) return asp_relu(x);
    return x;
}

static float derivada(float x, const char* tipo) {
    if (strcmp(tipo, "sigmoid") == 0) return asp_sigmoid_derivada(x);
    if (strcmp(tipo, "relu") == 0) return (x > 0) ? 1.0f : 0.0f;
    return 1.0f;
}

// ==================== CRIAR ====================
ASP_Rede* asp_criar(int camadas, int neuronios[], const char* ativacao) {
    srand(time(NULL));
    
    ASP_Rede* rede = malloc(sizeof(ASP_Rede));
    rede->camadas = camadas;
    rede->taxa = 0.01f;
    strcpy(rede->ativacao, ativacao);
    
    rede->neuronios = malloc(camadas * sizeof(int));
    memcpy(rede->neuronios, neuronios, camadas * sizeof(int));
    
    rede->pesos = malloc((camadas-1) * sizeof(float*));
    rede->biases = malloc((camadas-1) * sizeof(float*));
    
    for (int c = 0; c < camadas-1; c++) {
        int in = neuronios[c];
        int out = neuronios[c+1];
        
        rede->pesos[c] = malloc(in * out * sizeof(float));
        rede->biases[c] = malloc(out * sizeof(float));
        
        float limite = sqrtf(2.0f / (in + out));
        for (int i = 0; i < in * out; i++)
            rede->pesos[c][i] = asp_peso_aleatorio() * limite;
        
        for (int i = 0; i < out; i++)
            rede->biases[c][i] = 0.0f;
    }
    
    return rede;
}

// ==================== PREVER ====================
float* asp_prever(ASP_Rede* rede, float entrada[]) {
    float* atual = malloc(rede->neuronios[0] * sizeof(float));
    memcpy(atual, entrada, rede->neuronios[0] * sizeof(float));
    
    for (int c = 0; c < rede->camadas-1; c++) {
        int in = rede->neuronios[c];
        int out = rede->neuronios[c+1];
        float* proximo = malloc(out * sizeof(float));
        
        for (int n = 0; n < out; n++) {
            float soma = rede->biases[c][n];
            for (int i = 0; i < in; i++)
                soma += atual[i] * rede->pesos[c][i * out + n];
            
            // Hidden layers → ReLU, Output → ativação escolhida
            if (c < rede->camadas-2)
                proximo[n] = ativar(soma, "relu");
            else
                proximo[n] = ativar(soma, rede->ativacao);
        }
        
        free(atual);
        atual = proximo;
    }
    
    return atual;
}

// ==================== TREINAR (BACKPROP) ====================
void asp_treinar(ASP_Rede* rede, float** X, float** y, int amostras, int epocas) {
    printf("Treino: %d épocas com %d amostras\n", epocas, amostras);
    
    for (int ep = 0; ep < epocas; ep++) {
        float erro_total = 0;
        
        for (int a = 0; a < amostras; a++) {
            // === FORWARD ===
            float* ativacoes[rede->camadas];
            ativacoes[0] = malloc(rede->neuronios[0] * sizeof(float));
            memcpy(ativacoes[0], X[a], rede->neuronios[0] * sizeof(float));
            
            float* z[rede->camadas-1];
            
            for (int c = 0; c < rede->camadas-1; c++) {
                int in = rede->neuronios[c];
                int out = rede->neuronios[c+1];
                
                z[c] = malloc(out * sizeof(float));
                ativacoes[c+1] = malloc(out * sizeof(float));
                
                for (int n = 0; n < out; n++) {
                    z[c][n] = rede->biases[c][n];
                    for (int i = 0; i < in; i++)
                        z[c][n] += ativacoes[c][i] * rede->pesos[c][i * out + n];
                    
                    if (c < rede->camadas-2)
                        ativacoes[c+1][n] = ativar(z[c][n], "relu");
                    else
                        ativacoes[c+1][n] = ativar(z[c][n], rede->ativacao);
                }
            }
            
            // === ERRO ===
            int ultima = rede->camadas-1;
            for (int i = 0; i < rede->neuronios[ultima]; i++) {
                float erro = y[a][i] - ativacoes[ultima][i];
                erro_total += erro * erro;
            }
            
            // === BACKPROP ===
            float* deltas[rede->camadas-1];
            
            // Última camada
            int c = rede->camadas-2;
            deltas[c] = malloc(rede->neuronios[c+1] * sizeof(float));
            for (int n = 0; n < rede->neuronios[c+1]; n++) {
                float saida = ativacoes[c+1][n];
                if (strcmp(rede->ativacao, "sigmoid") == 0)
                    deltas[c][n] = saida - y[a][n]; // BCE
                else
                    deltas[c][n] = (y[a][n] - saida) * derivada(z[c][n], rede->ativacao); // MSE ou outra
            }
            
            // Camadas ocultas
            for (c = rede->camadas-3; c >= 0; c--) {
                deltas[c] = malloc(rede->neuronios[c+1] * sizeof(float));
                for (int n = 0; n < rede->neuronios[c+1]; n++) {
                    float soma = 0;
                    for (int k = 0; k < rede->neuronios[c+2]; k++)
                        soma += deltas[c+1][k] * rede->pesos[c+1][n * rede->neuronios[c+2] + k];
                    
                    deltas[c][n] = soma * derivada(z[c][n], "relu");
                }
            }
            
            // === UPDATE PESOS & BIASES ===
            for (c = 0; c < rede->camadas-1; c++) {
                int in = rede->neuronios[c];
                int out = rede->neuronios[c+1];
                
                for (int n = 0; n < out; n++)
                    rede->biases[c][n] += rede->taxa * deltas[c][n];
                
                for (int i = 0; i < in; i++)
                    for (int n = 0; n < out; n++)
                        rede->pesos[c][i * out + n] += rede->taxa * deltas[c][n] * ativacoes[c][i];
                
                free(deltas[c]);
            }
            
            // Cleanup
            for (c = 0; c < rede->camadas; c++) free(ativacoes[c]);
            for (c = 0; c < rede->camadas-1; c++) free(z[c]);
        }
        
        if ((ep+1) % (epocas/10) == 0)
            printf("Época %d: MSE = %.6f\n", ep+1, erro_total/amostras);
    }
}

// ==================== SALVAR ====================
int asp_salvar(ASP_Rede* rede, const char* caminho_pasta) {
    if (!rede || !caminho_pasta) return 0;
    mkdir(caminho_pasta, 0755);
    
    char pesos_path[512];
    sprintf(pesos_path, "%s/pesos.asp", caminho_pasta);
    
    FILE* f = fopen(pesos_path, "wb");
    if (!f) return 0;
    
    fwrite(&rede->camadas, sizeof(int), 1, f);
    fwrite(rede->neuronios, sizeof(int), rede->camadas, f);
    fwrite(&rede->taxa, sizeof(float), 1, f);
    fwrite(rede->ativacao, sizeof(char), 20, f);
    
    for (int c = 0; c < rede->camadas-1; c++) {
        int in = rede->neuronios[c];
        int out = rede->neuronios[c+1];
        fwrite(rede->pesos[c], sizeof(float), in * out, f);
        fwrite(rede->biases[c], sizeof(float), out, f);
    }
    fclose(f);
    
    char config_path[512];
    sprintf(config_path, "%s/config.asp", caminho_pasta);
    FILE* config = fopen(config_path, "w");
    if (!config) return 0;
    
    int total_params = 0;
    for (int c = 0; c < rede->camadas-1; c++)
        total_params += rede->neuronios[c] * rede->neuronios[c+1] + rede->neuronios[c+1];
    
    fprintf(config,
        "{\n"
        "  \"asp\": {\"versao\": \"1.0\",\"descricao\": \"Modelo ASP treinado\",\"data_criacao\": \"%s %s\"},\n"
        "  \"arquitetura\": {\"nome\": \"ASP-MLP\",\"camadas\": %d,\"neuronios\": [", 
        __DATE__, __TIME__, rede->camadas
    );
    for (int i = 0; i < rede->camadas; i++) {
        fprintf(config, "%d", rede->neuronios[i]);
        if (i < rede->camadas-1) fprintf(config, ", ");
    }
    fprintf(config, "],\"ativacao\": \"%s\",\"parametros_totais\": %d},\n", rede->ativacao, total_params);
    
    fprintf(config,
        "  \"treino\": {\"taxa_aprendizado\": %.6f,\"inicializacao\": \"Xavier\"},\n"
        "  \"arquivos\": {\"pesos\": \"pesos.asp\",\"config\": \"config.asp\"},\n"
        "  \"camadas_detalhadas\": [\n", rede->taxa
    );
    for (int c = 0; c < rede->camadas-1; c++) {
        fprintf(config,
            "    {\"id\": %d,\"tipo\": \"densa\",\"entrada\": %d,\"saida\": %d,\"pesos\": %d,\"biases\": %d}%s\n",
            c, rede->neuronios[c], rede->neuronios[c+1],
            rede->neuronios[c]*rede->neuronios[c+1], rede->neuronios[c+1],
            (c < rede->camadas-2) ? "," : ""
        );
    }
    fprintf(config, "  ]\n}\n");
    fclose(config);
    
    printf("  ├── pesos.asp (binário)\n");
    printf("  └── config.asp (JSON)\n");
    
    return 1;
}

// ==================== CARREGAR ====================
ASP_Rede* asp_carregar(const char* caminho_pasta) {
    char pesos_path[512];
    sprintf(pesos_path, "%s/pesos.asp", caminho_pasta);
    
    FILE* f = fopen(pesos_path, "rb");
    if (!f) { printf("ERRO: Não encontrou %s\n", pesos_path); return NULL; }
    
    int camadas;
    fread(&camadas, sizeof(int), 1, f);
    
    int* neuronios = malloc(camadas * sizeof(int));
    fread(neuronios, sizeof(int), camadas, f);
    
    float taxa;
    fread(&taxa, sizeof(float), 1, f);
    
    char ativacao[20];
    fread(ativacao, sizeof(char), 20, f);
    
    ASP_Rede* rede = asp_criar(camadas, neuronios, ativacao);
    rede->taxa = taxa;
    
    for (int c = 0; c < camadas-1; c++) {
        int in = neuronios[c];
        int out = neuronios[c+1];
        fread(rede->pesos[c], sizeof(float), in * out, f);
        fread(rede->biases[c], sizeof(float), out, f);
    }
    
    free(neuronios);
    fclose(f);
    
    char config_path[512];
    sprintf(config_path, "%s/config.asp", caminho_pasta);
    FILE* config = fopen(config_path, "r");
    if (config) { printf("✅ Modelo carregado de %s/\n", caminho_pasta); fclose(config); }
    
    return rede;
}

// ==================== DEBUG ====================
void asp_info(ASP_Rede* rede) {
    printf("=== REDE ASP ===\n");
    printf("Camadas: %d [", rede->camadas);
    for (int i = 0; i < rede->camadas; i++) {
        printf("%d", rede->neuronios[i]);
        if (i < rede->camadas-1) printf("->");
    }
    printf("]\nAtivação saída: %s\nTaxa: %.4f\n", rede->ativacao, rede->taxa);
    
    int params = 0;
    for (int c = 0; c < rede->camadas-1; c++)
        params += rede->neuronios[c] * rede->neuronios[c+1] + rede->neuronios[c+1];
    printf("Parâmetros: %d\n", params);
}

void asp_liberar(ASP_Rede* rede) {
    for (int c = 0; c < rede->camadas-1; c++) {
        free(rede->pesos[c]);
        free(rede->biases[c]);
    }
    free(rede->pesos);
    free(rede->biases);
    free(rede->neuronios);
    free(rede);
}