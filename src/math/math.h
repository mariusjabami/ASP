#ifndef ASP_MATH_H
#define ASP_MATH_H

#include <math.h>
float asp_sigmoid(float x);
float asp_sigmoid_derivada(float x);
float asp_degrau(float x);
float asp_relu(float x);

float asp_mse(float previsto, float real);
float asp_peso_aleatorio();

void asp_matriz_mult(float* saida, float* a, float* b, int linhas_a, int cols_a, int cols_b);

#endif
