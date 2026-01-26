CC = gcc
CFLAGS = -std=c99 -Wall -O2 -I./src
TARGET_TRAIN = asp_train
TARGET_PREDICT = asp_predict

SRC_CORE = src/core/asp.c
SRC_MATH = src/math/math.c
SRC_DATA = src/data/data.c

all: $(TARGET_TRAIN) $(TARGET_PREDICT)

# Programa de treino
$(TARGET_TRAIN): asp_train.c $(SRC_CORE) $(SRC_MATH) $(SRC_DATA)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Programa de inferência
$(TARGET_PREDICT): asp_predict.c $(SRC_CORE) $(SRC_MATH)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Limpeza
clean:
	rm -f $(TARGET_TRAIN) $(TARGET_PREDICT) core.* vgcore.*

# Limpeza total
clean-all: clean
	rm -rf modelos/*

# Teste rápido
test: $(TARGET_TRAIN) $(TARGET_PREDICT)
	@echo "=== TESTE COMPLETO DO ASP ==="
	@echo "1. Treinando modelo pequeno..."
	@./asp_train --csv teste_dados.csv --layers 8,4,1 --epochs 5 --nome teste_final --no-header 2>/dev/null || true
	@echo "2. Fazendo inferência..."
	@./asp_predict --model modelos/teste_final --input "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8" 2>/dev/null || true
	@echo "=== FIM DO TESTE ==="

.PHONY: all clean clean-all test
