#ifndef ASP_CLI_H
#define ASP_CLI_H

typedef struct {
    char csv[256];
    char saida[256];
    int camadas;
    int* neuronios;
    int epocas;
    float taxa;
    char ativacao[20];
} ConfigCLI;

ConfigCLI parse_args(int argc, char** argv);
void mostrar_ajuda();

#endif
