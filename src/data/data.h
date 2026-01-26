#ifndef ASP_DATA_H
#define ASP_DATA_H

typedef struct {
    float** X;
    float** y;
    int samples;
    int features;
    int outputs;
} Dataset;

Dataset carregar_csv(const char* caminho, int tem_cabecalho);
void liberar_dataset(Dataset* dados);

#endif
