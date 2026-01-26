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
    if (strcmp(tipo, "degrau") == 0) return 1.0f;
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
            
            if (c < rede->camadas-2)
                proximo[n] = ativar(soma, rede->ativacao);
            else
                proximo[n] = soma; // Última camada linear
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
                        ativacoes[c+1][n] = ativar(z[c][n], rede->ativacao);
                    else
                        ativacoes[c+1][n] = z[c][n];
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
                deltas[c][n] = (y[a][n] - saida); // * derivada(z[c][n]) para MSE
            }
            
            // Camadas ocultas
            for (c = rede->camadas-3; c >= 0; c--) {
                deltas[c] = malloc(rede->neuronios[c+1] * sizeof(float));
                for (int n = 0; n < rede->neuronios[c+1]; n++) {
                    float soma = 0;
                    for (int k = 0; k < rede->neuronios[c+2]; k++)
                        soma += deltas[c+1][k] * rede->pesos[c+1][n * rede->neuronios[c+2] + k];
                    
                    deltas[c][n] = soma * derivada(z[c][n], rede->ativacao);
                }
            }
            
            // === UPDATE ===
            for (c = 0; c < rede->camadas-1; c++) {
                int in = rede->neuronios[c];
                int out = rede->neuronios[c+1];
                
                // Update biases
                for (int n = 0; n < out; n++)
                    rede->biases[c][n] += rede->taxa * deltas[c][n];
                
                // Update pesos
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

// ==================== SALVAR (COM pesos.asp E config.asp) ====================
int asp_salvar(ASP_Rede* rede, const char* caminho_pasta) {
    if (!rede || !caminho_pasta) return 0;
    
    // Cria pasta se não existir
    mkdir(caminho_pasta, 0755);
    
    // 1. SALVA pesos.asp (BINÁRIO)
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
    
    // 2. GERA config.asp (JSON LEGÍVEL)
    char config_path[512];
    sprintf(config_path, "%s/config.asp", caminho_pasta);
    
    FILE* config = fopen(config_path, "w");
    if (!config) return 0;
    
    // Calcula parâmetros
    int total_params = 0;
    for (int c = 0; c < rede->camadas-1; c++) {
        total_params += rede->neuronios[c] * rede->neuronios[c+1];
        total_params += rede->neuronios[c+1];
    }
    
    // Escreve config.asp
    fprintf(config, "{\n");
    fprintf(config, "  \"asp\": {\n");
    fprintf(config, "    \"versao\": \"1.0\",\n");
    fprintf(config, "    \"descricao\": \"Modelo ASP treinado\",\n");
    fprintf(config, "    \"data_criacao\": \"%s %s\"\n", __DATE__, __TIME__);
    fprintf(config, "  },\n");
    
    fprintf(config, "  \"arquitetura\": {\n");
    fprintf(config, "    \"nome\": \"ASP-MLP\",\n");
    fprintf(config, "    \"camadas\": %d,\n", rede->camadas);
    fprintf(config, "    \"neuronios\": [");
    for (int i = 0; i < rede->camadas; i++) {
        fprintf(config, "%d", rede->neuronios[i]);
        if (i < rede->camadas-1) fprintf(config, ", ");
    }
    fprintf(config, "],\n");
    fprintf(config, "    \"ativacao\": \"%s\",\n", rede->ativacao);
    fprintf(config, "    \"parametros_totais\": %d\n", total_params);
    fprintf(config, "  },\n");
    
    fprintf(config, "  \"treino\": {\n");
    fprintf(config, "    \"taxa_aprendizado\": %.6f,\n", rede->taxa);
    fprintf(config, "    \"inicializacao\": \"Xavier\"\n");
    fprintf(config, "  },\n");
    
    fprintf(config, "  \"arquivos\": {\n");
    fprintf(config, "    \"pesos\": \"pesos.asp\",\n");
    fprintf(config, "    \"config\": \"config.asp\"\n");
    fprintf(config, "  },\n");
    
    fprintf(config, "  \"camadas_detalhadas\": [\n");
    for (int c = 0; c < rede->camadas-1; c++) {
        fprintf(config, "    {\n");
        fprintf(config, "      \"id\": %d,\n", c);
        fprintf(config, "      \"tipo\": \"densa\",\n");
        fprintf(config, "      \"entrada\": %d,\n", rede->neuronios[c]);
        fprintf(config, "      \"saida\": %d,\n", rede->neuronios[c+1]);
        fprintf(config, "      \"pesos\": %d,\n", rede->neuronios[c] * rede->neuronios[c+1]);
        fprintf(config, "      \"biases\": %d\n", rede->neuronios[c+1]);
        fprintf(config, "    }%s\n", (c < rede->camadas-2) ? "," : "");
    }
    fprintf(config, "  ]\n");
    
    fprintf(config, "}\n");
    
    fclose(config);
    
    printf("  ├── pesos.asp (binário, %d bytes)\n", 
           (int)(sizeof(int) + rede->camadas*sizeof(int) + sizeof(float) + 20 + 
                 total_params * sizeof(float)));
    printf("  └── config.asp (JSON, %d bytes)\n", total_params * 2);
    
    return 1;
}

// ==================== CARREGAR ====================
ASP_Rede* asp_carregar(const char* caminho_pasta) {
    char pesos_path[512];
    sprintf(pesos_path, "%s/pesos.asp", caminho_pasta);
    
    FILE* f = fopen(pesos_path, "rb");
    if (!f) {
        printf("ERRO: Não encontrou %s\n", pesos_path);
        return NULL;
    }
    
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
    
    // Verifica se config.asp existe
    char config_path[512];
    sprintf(config_path, "%s/config.asp", caminho_pasta);
    FILE* config = fopen(config_path, "r");
    if (config) {
        printf("✅ Modelo carregado de %s/\n", caminho_pasta);
        fclose(config);
    }
    
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
    printf("]\nAtivação: %s\nTaxa: %.4f\n", rede->ativacao, rede->taxa);
    
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
