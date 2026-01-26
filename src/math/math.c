#include <math.h>
#include <stdlib.h>
#include "math.h"


float asp_sigmoid(float x){
	return 1.0f / (1.0f + expf(-x));
}
float asp_sigmoid_derivada(float x){
	float s = asp_sigmoid(x);
	return s * (1.0f - s);
}
float asp_degrau(float x){
	return (x >= 0) ? 1.0f : 0.0f;
}
float asp_relu(float x){
	return (x > 0) ? x : 0.0f;
}

float asp_mse(float previsto, float real){
	float diff = previsto - real;
	return diff * diff;
}
float asp_peso_aleatorio(){
	return ((float)rand() / RAND_MAX) - 0.5f;
}

void asp_matriz_mult(float* saida, float* a, float* b, int linhas_a, int cols_a, int cols_b){
	for (int i = 0; i < linhas_a; i++) {
		for (int j = 0; j < cols_b; j++) {
			float soma = 0.0f;
			for (int k = 0; k < cols_a; k++) {
				soma += a[i + cols_a + k] * b[k * cols_b + j];
			}
			saida[i * cols_b + j] = soma;
		
		}
	
	}

}


