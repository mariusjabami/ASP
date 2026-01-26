#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "src/core/asp.h"

// ==================== DECLARAÇÕES ====================
void mostrar_ajuda();
float* parse_entrada(const char* str, int* tamanho);
void mostrar_config(const char* caminho_modelo);

// ==================== FUNÇÃO PRINCIPAL ====================
int main(int argc, char** argv) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║            ASP - INFERÊNCIA                      ║\n");
    printf("║            Sistema de Previsão                   ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");
    
    // Variáveis de configuração
    char caminho_modelo[256] = "";
    char entrada_str[1024] = "";
    char arquivo_entrada[256] = "";
    int modo_batch = 0;
    
    // Parse argumentos
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) {
            strncpy(caminho_modelo, argv[++i], 255);
        }
        else if (strcmp(argv[i], "--input") == 0 && i+1 < argc) {
            strncpy(entrada_str, argv[++i], 1023);
        }
        else if (strcmp(argv[i], "--file") == 0 && i+1 < argc) {
            strncpy(arquivo_entrada, argv[++i], 255);
            modo_batch = 1;
        }
        else if (strcmp(argv[i], "--help") == 0) {
            mostrar_ajuda();
            return 0;
        }
        else if (strcmp(argv[i], "--info") == 0 && i+1 < argc) {
            mostrar_config(argv[++i]);
            return 0;
        }
    }
    
    // Validação
    if (caminho_modelo[0] == '\0') {
        printf("❌ ERRO: Especifique um modelo com --model\n");
        mostrar_ajuda();
        return 1;
    }
    
    if (!modo_batch && entrada_str[0] == '\0') {
        printf("❌ ERRO: Especifique entrada com --input ou --file\n");
        mostrar_ajuda();
        return 1;
    }
    
    // ==================== CARREGA MODELO ====================
    printf("📂 Carregando modelo: %s\n", caminho_modelo);
    ASP_Rede* rede = asp_carregar(caminho_modelo);
    
    if (!rede) {
        printf("❌ ERRO: Não foi possível carregar o modelo\n");
        printf("   Verifique se a pasta contém pesos.asp e config.asp\n");
        return 1;
    }
    
    printf("   ✅ Modelo carregado com sucesso!\n");
    asp_info(rede);
    
    // ==================== MODO SINGLE (--input) ====================
    if (!modo_batch) {
        printf("\n🔮 Previsão Single:\n");
        printf("   Entrada: %s\n", entrada_str);
        
        // Parse entrada
        int tamanho_entrada;
        float* entrada = parse_entrada(entrada_str, &tamanho_entrada);
        
        if (!entrada) {
            printf("❌ ERRO: Entrada inválida\n");
            asp_liberar(rede);
            return 1;
        }
        
        // Verifica compatibilidade
        if (tamanho_entrada != rede->neuronios[0]) {
            printf("❌ ERRO: Entrada tem %d valores, mas modelo espera %d\n",
                   tamanho_entrada, rede->neuronios[0]);
            free(entrada);
            asp_liberar(rede);
            return 1;
        }
        
        // Faz previsão
        float* resultado = asp_prever(rede, entrada);
        
        // Mostra resultado
        printf("\n✅ RESULTADO:\n");
        printf("   ");
        for (int i = 0; i < rede->camadas; i++) {
            printf("%d", rede->neuronios[i]);
            if (i < rede->camadas-1) printf("→");
        }
        printf(" = ");
        
        if (rede->neuronios[rede->camadas-1] == 1) {
            // Saída única (ex: probabilidade)
            printf("%.6f", resultado[0]);
            
            // Interpretação para classificação binária
            if (strcmp(rede->ativacao, "sigmoid") == 0) {
                printf(" (%.1f%% chance)", resultado[0] * 100);
            } else if (strcmp(rede->ativacao, "degrau") == 0) {
                printf(" → %s", resultado[0] > 0.5 ? "CLASSE 1" : "CLASSE 0");
            }
        } else {
            // Múltiplas saídas
            printf("[");
            for (int i = 0; i < rede->neuronios[rede->camadas-1]; i++) {
                printf("%.4f", resultado[i]);
                if (i < rede->neuronios[rede->camadas-1]-1) printf(", ");
            }
            printf("]");
        }
        printf("\n");
        
        // Cleanup
        free(entrada);
        free(resultado);
    }
    
    // ==================== MODO BATCH (--file) ====================
    else {
        printf("\n📊 Modo Batch: %s\n", arquivo_entrada);
        
        FILE* f = fopen(arquivo_entrada, "r");
        if (!f) {
            printf("❌ ERRO: Não foi possível abrir %s\n", arquivo_entrada);
            asp_liberar(rede);
            return 1;
        }
        
        char linha[1024];
        int linha_num = 0;
        int acertos = 0, total = 0;
        
        printf("\n📈 RESULTADOS:\n");
        printf("   LINHA | ENTRADA → SAÍDA\n");
        printf("   --------\n");
        
        while (fgets(linha, sizeof(linha), f)) {
            linha_num++;
            
            // Remove newline
            linha[strcspn(linha, "\n")] = 0;
            
            // Pula linhas vazias
            if (strlen(linha) == 0) continue;
            
            // Parse entrada (último valor é o target real, se existir)
            char* ultima_virgula = strrchr(linha, ',');
            int tem_target = (ultima_virgula != NULL);
            
            char entrada_para_parse[1024];
            float target_real = 0;
            
            if (tem_target) {
                // Separa entrada do target
                *ultima_virgula = '\0';
                strcpy(entrada_para_parse, linha);
                target_real = atof(ultima_virgula + 1);
            } else {
                strcpy(entrada_para_parse, linha);
            }
            
            // Parse e previsão
            int tamanho_entrada;
            float* entrada = parse_entrada(entrada_para_parse, &tamanho_entrada);
            
            if (!entrada || tamanho_entrada != rede->neuronios[0]) {
                printf("   %4d | ERRO: entrada inválida\n", linha_num);
                if (entrada) free(entrada);
                continue;
            }
            
            float* resultado = asp_prever(rede, entrada);
            
            // Mostra resultado
            printf("   %4d | ", linha_num);
            
            // Entrada resumida
            if (tamanho_entrada <= 5) {
                for (int i = 0; i < tamanho_entrada; i++) {
                    printf("%.2f", entrada[i]);
                    if (i < tamanho_entrada-1) printf(",");
                }
            } else {
                printf("[%d valores]", tamanho_entrada);
            }
            
            printf(" → ");
            
            if (rede->neuronios[rede->camadas-1] == 1) {
                printf("%.4f", resultado[0]);
                
                if (tem_target) {
                    // Verifica acerto (para classificação binária)
                    int pred_classe = (resultado[0] > 0.5) ? 1 : 0;
                    int real_classe = (target_real > 0.5) ? 1 : 0;
                    
                    if (pred_classe == real_classe) {
                        printf(" ✅");
                        acertos++;
                    } else {
                        printf(" ❌ (real: %.0f)", target_real);
                    }
                    total++;
                }
            } else {
                printf("[");
                for (int i = 0; i < rede->neuronios[rede->camadas-1]; i++) {
                    printf("%.2f", resultado[i]);
                    if (i < rede->neuronios[rede->camadas-1]-1) printf(",");
                }
                printf("]");
            }
            
            printf("\n");
            
            free(entrada);
            free(resultado);
        }
        
        fclose(f);
        
        if (total > 0) {
            printf("\n📊 ESTATÍSTICAS:\n");
            printf("   Acurácia: %d/%d = %.1f%%\n", acertos, total, (float)acertos/total*100);
        }
    }
    
    // ==================== FINALIZA ====================
    asp_liberar(rede);
    
    printf("\n══════════════════════════════════════════════════\n");
    printf("✅ INFERÊNCIA CONCLUÍDA!\n");
    printf("══════════════════════════════════════════════════\n\n");
    
    return 0;
}

// ==================== FUNÇÕES AUXILIARES ====================

// Parse string como "1,2,3,4" para array float
float* parse_entrada(const char* str, int* tamanho) {
    // Conta números
    *tamanho = 1;
    for (int i = 0; str[i]; i++) {
        if (str[i] == ',') (*tamanho)++;
    }
    
    float* array = malloc(*tamanho * sizeof(float));
    if (!array) return NULL;
    
    // Faz cópia para strtok
    char* copia = malloc(strlen(str) + 1);
    strcpy(copia, str);
    
    char* token = strtok(copia, ",");
    int idx = 0;
    
    while (token && idx < *tamanho) {
        // Remove espaços
        while (isspace(*token)) token++;
        
        array[idx++] = atof(token);
        token = strtok(NULL, ",");
    }
    
    free(copia);
    return array;
}

// Mostra informações do modelo
void mostrar_config(const char* caminho_modelo) {
    char config_path[512];
    sprintf(config_path, "%s/config.asp", caminho_modelo);
    
    FILE* f = fopen(config_path, "r");
    if (!f) {
        printf("❌ Não encontrou config.asp em %s\n", caminho_modelo);
        return;
    }
    
    printf("\n📄 CONFIGURAÇÃO DO MODELO: %s\n", caminho_modelo);
    printf("══════════════════════════════════════════════════\n");
    
    char linha[256];
    while (fgets(linha, sizeof(linha), f)) {
        printf("%s", linha);
    }
    
    fclose(f);
    printf("══════════════════════════════════════════════════\n");
}

// Ajuda
void mostrar_ajuda() {
    printf("USO: asp_predict --model PASTA_MODELO [OPÇÕES]\n");
    printf("\nOPÇÕES:\n");
    printf("  --model PASTA    Pasta do modelo treinado (OBRIGATÓRIO)\n");
    printf("  --input VALORES  Entrada para previsão (ex: \"1,2,3,4\")\n");
    printf("  --file ARQUIVO   Arquivo com múltiplas entradas (uma por linha)\n");
    printf("  --info PASTA     Mostra configuração do modelo sem prever\n");
    printf("  --help           Mostra esta ajuda\n");
    printf("\nEXEMPLOS:\n");
    printf("  asp_predict --model modelos/diabetes_model --input \"6,148,72,35,0,33.6,0.627,50\"\n");
    printf("  asp_predict --model modelos/meu_modelo --file dados_teste.csv\n");
    printf("  asp_predict --model modelos/diabetes_model --info\n");
}
