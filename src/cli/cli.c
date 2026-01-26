#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cli.h"

ConfigCLI parse_args(int argc, char** argv) {
    ConfigCLI config = {0};
    strcpy(config.ativacao, "sigmoid");
    strcpy(config.saida, "modelo_asp");
    config.epocas = 1000;
    config.taxa = 0.01;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0 && i+1 < argc)
            strcpy(config.csv, argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i+1 < argc)
            strcpy(config.saida, argv[++i]);
        else if (strcmp(argv[i], "--layers") == 0 && i+1 < argc) {
            char* token = strtok(argv[++i], ",");
            int temp[10];
            config.camadas = 0;
            while (token && config.camadas < 10) {
                temp[config.camadas++] = atoi(token);
                token = strtok(NULL, ",");
            }
            config.neuronios = malloc(config.camadas * sizeof(int));
            memcpy(config.neuronios, temp, config.camadas * sizeof(int));
        }
        else if (strcmp(argv[i], "--epochs") == 0 && i+1 < argc)
            config.epocas = atoi(argv[++i]);
        else if (strcmp(argv[i], "--activation") == 0 && i+1 < argc)
            strcpy(config.ativacao, argv[++i]);
        else if (strcmp(argv[i], "--help") == 0)
            mostrar_ajuda();
    }
    
    return config;
}

void mostrar_ajuda() {
    printf("USO: asp_train --csv dados.csv --layers 8,4,1\n");
    printf("OPÇÕES:\n");
    printf("  --csv caminho       Arquivo CSV com dados\n");
    printf("  --layers n,n,n      Arquitetura (ex: 8,4,1)\n");
    printf("  --out nome          Nome do modelo de saída\n");
    printf("  --epochs N          Épocas de treino (padrão: 1000)\n");
    printf("  --activation nome   sigmoid|degrau|relu (padrão: sigmoid)\n");
    exit(0);
}
