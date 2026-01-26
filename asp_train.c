#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include "src/core/asp.h"
#include "src/data/data.h"

// ==================== ESTRUTURAS ====================
typedef struct {
    char csv[256];
    char nome_modelo[256];
    int camadas;
    int* neuronios;
    int epocas;
    float taxa;
    char ativacao[20];
    int tem_cabecalho;
} ConfigTreino;

// ==================== FUNÇÕES CLI ====================
ConfigTreino parse_args(int argc, char** argv) {
    ConfigTreino cfg = {0};
    
    // Valores padrão
    strcpy(cfg.ativacao, "sigmoid");
    cfg.epocas = 1000;
    cfg.taxa = 0.01f;
    cfg.tem_cabecalho = 1;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0 && i+1 < argc) {
            strncpy(cfg.csv, argv[++i], 255);
        }
        else if (strcmp(argv[i], "--nome") == 0 && i+1 < argc) {
            strncpy(cfg.nome_modelo, argv[++i], 255);
        }
        else if (strcmp(argv[i], "--layers") == 0 && i+1 < argc) {
            char* token = strtok(argv[++i], ",");
            int temp[10];
            cfg.camadas = 0;
            
            while (token && cfg.camadas < 10) {
                temp[cfg.camadas++] = atoi(token);
                token = strtok(NULL, ",");
            }
            
            cfg.neuronios = malloc(cfg.camadas * sizeof(int));
            memcpy(cfg.neuronios, temp, cfg.camadas * sizeof(int));
        }
        else if (strcmp(argv[i], "--epochs") == 0 && i+1 < argc) {
            cfg.epocas = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) {
            cfg.taxa = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--activation") == 0 && i+1 < argc) {
            strncpy(cfg.ativacao, argv[++i], 19);
        }
        else if (strcmp(argv[i], "--no-header") == 0) {
            cfg.tem_cabecalho = 0;
        }
        else if (strcmp(argv[i], "--help") == 0) {
            printf("USO: asp_train --csv dados.csv --layers 8,4,1 --nome meu_modelo\n");
            printf("\nOPÇÕES:\n");
            printf("  --csv CAMINHO          Arquivo CSV com dados (OBRIGATÓRIO)\n");
            printf("  --layers n,n,n         Arquitetura da rede (ex: 8,4,1)\n");
            printf("  --nome NOME            Nome do modelo (padrão: data+hora)\n");
            printf("  --epochs N             Épocas de treino (padrão: 1000)\n");
            printf("  --lr VALOR             Taxa de aprendizado (padrão: 0.01)\n");
            printf("  --activation NOME      sigmoid|degrau|relu (padrão: sigmoid)\n");
            printf("  --no-header            CSV não tem linha de cabeçalho\n");
            printf("  --help                 Mostra esta ajuda\n");
            exit(0);
        }
    }
    
    // Nome padrão se não especificado
    if (cfg.nome_modelo[0] == '\0') {
        time_t t = time(NULL);
        struct tm tm = *localtime(&t);
        sprintf(cfg.nome_modelo, "modelo_%04d%02d%02d_%02d%02d%02d",
                tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec);
    }
    
    return cfg;
}

// ==================== GERAR LOG ====================
void gerar_log_treino(const char* caminho_pasta, ConfigTreino* cfg, Dataset* dados, float erro_final) {
    char log_path[512];
    sprintf(log_path, "%s/treino.log", caminho_pasta);
    
    FILE* log = fopen(log_path, "w");
    if (!log) return;
    
    fprintf(log, "=== LOG DE TREINO ASP ===\n");
    fprintf(log, "Data: %s %s\n", __DATE__, __TIME__);
    fprintf(log, "Comando: ");
    for (int i = 0; i < 10 && i < 10; i++) fprintf(log, "%s ", "asp_train"); // placeholder
    
    fprintf(log, "\n\nCONFIGURAÇÃO:\n");
    fprintf(log, "  Modelo: %s\n", cfg->nome_modelo);
    fprintf(log, "  CSV: %s\n", cfg->csv);
    fprintf(log, "  Amostras: %d\n", dados->samples);
    fprintf(log, "  Features: %d\n", dados->features);
    fprintf(log, "  Saídas: %d\n", dados->outputs);
    fprintf(log, "  Tem cabeçalho: %s\n", cfg->tem_cabecalho ? "sim" : "não");
    
    if (cfg->camadas > 0) {
        fprintf(log, "  Arquitetura: [");
        for (int i = 0; i < cfg->camadas; i++) {
            fprintf(log, "%d", cfg->neuronios[i]);
            if (i < cfg->camadas-1) fprintf(log, "->");
        }
        fprintf(log, "]\n");
    }
    
    fprintf(log, "  Ativação: %s\n", cfg->ativacao);
    fprintf(log, "  Épocas: %d\n", cfg->epocas);
    fprintf(log, "  Taxa aprendizado: %.4f\n", cfg->taxa);
    
    fprintf(log, "\nRESULTADOS:\n");
    fprintf(log, "  Erro final (MSE): %.6f\n", erro_final);
    fprintf(log, "  Pasta modelo: %s/\n", caminho_pasta);
    
    fprintf(log, "\nARQUIVOS GERADOS:\n");
    fprintf(log, "  config.asp - Configuração completa da rede\n");
    fprintf(log, "  pesos.asp - Pesos treinados (binário)\n");
    fprintf(log, "  treino.log - Este arquivo\n");
    
    fprintf(log, "\n=== FIM DO LOG ===\n");
    
    fclose(log);
}

// ==================== FUNÇÃO PRINCIPAL ====================
int main(int argc, char** argv) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║            ASP - ADAPTIVE SYSTEM PHILOSOPHY      ║\n");
    printf("║               Sistema de Treino MLP              ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");
    
    // 1. Parse argumentos
    ConfigTreino config = parse_args(argc, argv);
    
    if (config.csv[0] == '\0') {
        printf("❌ ERRO: Especifique um arquivo CSV com --csv\n");
        printf("Use --help para ver opções\n");
        return 1;
    }
    
    printf("📁 Configuração:\n");
    printf("   CSV: %s\n", config.csv);
    printf("   Modelo: %s\n", config.nome_modelo);
    printf("   Épocas: %d\n", config.epocas);
    printf("   Ativação: %s\n\n", config.ativacao);
    
    // 2. Carrega dados
    printf("📊 Carregando dados...\n");
    Dataset dados = carregar_csv(config.csv, config.tem_cabecalho);
    
    if (dados.samples == 0) {
        printf("❌ ERRO: Nenhum dado carregado de %s\n", config.csv);
        return 1;
    }
    
    printf("   ✅ %d amostras carregadas\n", dados.samples);
    printf("   ✅ %d features, %d saída(s)\n", dados.features, dados.outputs);
    
    // 3. Define arquitetura (automática se não especificada)
    int* arquitetura = NULL;
    int num_camadas = 0;
    
    if (config.camadas > 0) {
        // Usa arquitetura especificada pelo usuário
        arquitetura = config.neuronios;
        num_camadas = config.camadas;
        
        // Verifica compatibilidade
        if (arquitetura[0] != dados.features) {
            printf("❌ ERRO: Primeira camada deve ter %d neurônios (tem %d)\n",
                   dados.features, arquitetura[0]);
            liberar_dataset(&dados);
            free(config.neuronios);
            return 1;
        }
        
        if (arquitetura[num_camadas-1] != dados.outputs) {
            printf("❌ ERRO: Última camada deve ter %d neurônios (tem %d)\n",
                   dados.outputs, arquitetura[num_camadas-1]);
            liberar_dataset(&dados);
            free(config.neuronios);
            return 1;
        }
    } else {
        // Arquitetura automática: [features, features/2, outputs]
        num_camadas = 3;
        arquitetura = malloc(3 * sizeof(int));
        arquitetura[0] = dados.features;
        arquitetura[1] = dados.features / 2;
        if (arquitetura[1] < 2) arquitetura[1] = 2; // Mínimo 2 neurônios
        arquitetura[2] = dados.outputs;
        
        printf("   🔧 Arquitetura automática: [%d, %d, %d]\n",
               arquitetura[0], arquitetura[1], arquitetura[2]);
    }
    
    // 4. Cria rede
    printf("\n🧠 Criando rede neural...\n");
    ASP_Rede* rede = asp_criar(num_camadas, arquitetura, config.ativacao);
    rede->taxa = config.taxa;
    
    asp_info(rede);
    
    // 5. Treina
    printf("\n🔥 Iniciando treino...\n");
    printf("   Progresso:\n");
    
    clock_t inicio = clock();
    asp_treinar(rede, dados.X, dados.y, dados.samples, config.epocas);
    clock_t fim = clock();
    
    double tempo = (double)(fim - inicio) / CLOCKS_PER_SEC;
    printf("\n   ⏱️  Tempo de treino: %.2f segundos\n", tempo);
    
    // 6. Cria pasta do modelo
    char caminho_pasta[512];
    sprintf(caminho_pasta, "modelos/%s", config.nome_modelo);
    mkdir("modelos", 0755);
    mkdir(caminho_pasta, 0755);
    
    // 7. Salva modelo (gera pesos.asp e config.asp)
    printf("\n💾 Salvando modelo...\n");
    if (asp_salvar(rede, caminho_pasta)) {
        printf("   ✅ Pasta: %s/\n", caminho_pasta);
        printf("   ├── pesos.asp (pesos treinados)\n");
        printf("   ├── config.asp (configuração JSON)\n");
    } else {
        printf("❌ ERRO ao salvar modelo\n");
    }
    
    // 8. Teste rápido com primeira amostra
    printf("\n🧪 Teste rápido:\n");
    float* predicao = asp_prever(rede, dados.X[0]);
    printf("   Primeira amostra -> Previsto: %.4f, Real: %.4f\n",
           predicao[0], dados.y[0][0]);
    free(predicao);
    
    // 9. Gera log
    gerar_log_treino(caminho_pasta, &config, &dados, 0.0f); // erro_final placeholder
    
    // 10. Cleanup
    liberar_dataset(&dados);
    asp_liberar(rede);
    
    if (config.camadas == 0) {
        free(arquitetura);
    } else {
        free(config.neuronios);
    }
    
    printf("\n══════════════════════════════════════════════════\n");
    printf("✅ TREINO CONCLUÍDO COM SUCESSO!\n");
    printf("📂 Modelo salvo em: modelos/%s/\n", config.nome_modelo);
    printf("══════════════════════════════════════════════════\n\n");
    
    return 0;
}
