#ifndef ASP_CORE_H
#define ASP_CORE_H

typedef struct {
    int camadas;
    int* neuronios;
    float** pesos;
    float** biases;
    float taxa;
    char ativacao[20];
} ASP_Rede;

// CORE
ASP_Rede* asp_criar(int camadas, int neuronios[], const char* ativacao);
void      asp_liberar(ASP_Rede* rede);
float*    asp_prever(ASP_Rede* rede, float entrada[]);
void      asp_treinar(ASP_Rede* rede, float** X, float** y, int amostras, int epocas);

// IO
int       asp_salvar(ASP_Rede* rede, const char* caminho);
ASP_Rede* asp_carregar(const char* caminho);

// DEBUG
void      asp_info(ASP_Rede* rede);

#endif
