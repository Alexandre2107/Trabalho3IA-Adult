train_data <- read.csv(file = "train_data.csv")
test_data <- read.csv(file = "test_data.csv")

# ============================================================================== #

#Algoritmo Baseline (Classe Majoritária)
train_data$income_Baseline <- 0

true_negatives <- sum(train_data$income_Baseline == 0 & train_data$income == 0)
false_positives <- sum(train_data$income_Baseline == 0 & train_data$income == 1)
false_negatives <- 0
true_positives <- 0


cat("Matriz de Confusão:\n")
cat("           Previsto 0   Previsto 1\n")
cat("Real 0:    ", true_negatives, "        ", false_positives, "\n")
cat("Real 1:    ", false_negatives, "        ", true_positives, "\n\n")

accuracy_baseline <- (true_negatives + true_positives) / length(train_data$income)
cat("Acurácia:", accuracy_baseline, "\n")

if (true_positives + false_positives == 0) {
  cat("Precisão: Não definida (nenhuma previsão positiva)\n")
} else {
  precision_baseline <- true_positives / (true_positives + false_positives)
  cat("Precisão:", precision_baseline, "\n")
}

train_data$income_Baseline <- remove()

# ============================================================================== #

# Algoritmo K-NN
if (!require(class)) install.packages("class")
library(class)

target <- train_data$income
K <- 5
folds <- cut(seq(1, nrow(train_data)), breaks=K, labels=FALSE)

accuracy_list <- c()
precision_list <- c()

# K-Fold Cross-Validation manual
for (i in 1:K) {
  test_indexes <- which(folds == i, arr.ind = TRUE)
  teste <- train_data[test_indexes, ]
  treino <- train_data[-test_indexes, ]
  
  # Definir os atributos preditores e a variável target
  train_features <- treino[, -ncol(treino)]
  train_labels <- treino$income
  test_features <- teste[, -ncol(teste)]
  test_labels <- teste$income
  
  # Aplicar o algoritmo K-NN (usando K = 5 como exemplo)
  knn_pred <- knn(train = train_features, test = test_features, cl = train_labels, k = 5)  
  
  
  accuracy <- sum(knn_pred == test_labels) / length(test_labels)
  accuracy_list <- c(accuracy_list, accuracy)
  
  true_positives <- sum(knn_pred == 1 & test_labels == 1)
  false_positives <- sum(knn_pred == 1 & test_labels == 0)
  if (true_positives + false_positives == 0) {
    precision <- 0
  } else {
    precision <- true_positives / (true_positives + false_positives)
    precision_list <- c(precision_list, precision)
  }
  
  
  cat("Matriz de Confusão para o fold", i, ":\n")
  cat("           Previsto 0   Previsto 1\n")
  cat("Real 0:    ", sum(knn_pred == 0 & test_labels == 0), "        ", sum(knn_pred == 1 & test_labels == 0), "\n")
  cat("Real 1:    ", sum(knn_pred == 0 & test_labels == 1), "        ", sum(knn_pred == 1 & test_labels == 1), "\n\n")
}

# Média das acurácias dos K folds
mean_accuracy_knn <- mean(accuracy_list)
print(paste("Acurácia média da validação cruzada:", mean_accuracy_knn))

# Média das precisões dos K folds
mean_precision_knn <- mean(precision_list)
print(paste("Precisão média da validação cruzada:", mean_precision_knn))

# ============================================================================== #

# Algoritmo arvore de decisão

if (!require(rpart)) install.packages("rpart")
library(rpart)

# Inicializar variáveis para guardar resultados
accuracy_list <- c()
precision_list <- c()

for (i in 1:K) {
  # Separar os dados em treinamento e teste
  test_indexes <- which(folds == i, arr.ind = TRUE)
  teste <- train_data[test_indexes, ]
  treino <- train_data[-test_indexes, ]
  
  # Definir os atributos preditores e a variável target
  train_features <- treino[, -ncol(treino)]
  train_labels <- treino$income
  test_features <- teste[, -ncol(teste)]
  test_labels <- teste$income
  
  # Aplicar o algoritmo Árvore de Decisão
  decision_tree <- rpart(income ~ ., data = treino, method = "class")
  rpart_pred <- predict(decision_tree, test_features, type = "class")
  
  accuracy <- sum(rpart_pred == test_labels) / length(test_labels)
  accuracy_list <- c(accuracy_list, accuracy)
  
  true_positives <- sum(rpart_pred == 1 & test_labels == 1)
  false_positives <- sum(rpart_pred == 1 & test_labels == 0)
  if (true_positives + false_positives == 0) {
    precision <- 0
  } else {
    precision <- true_positives / (true_positives + false_positives)
    precision_list <- c(precision_list, precision)
  }
  
  cat("Matriz de Confusão para o fold", i, ":\n")
  cat("           Previsto 0   Previsto 1\n")
  cat("Real 0:    ", sum(rpart_pred == 0 & test_labels == 0), "        ", sum(rpart_pred == 1 & test_labels == 0), "\n")
  cat("Real 1:    ", sum(rpart_pred == 0 & test_labels == 1), "        ", sum(rpart_pred == 1 & test_labels == 1), "\n\n")
}

# Média das acurácias dos K folds
mean_accuracy_DecTree <- mean(accuracy_list)
print(paste("Acurácia média da validação cruzada:", mean_accuracy_DecTree))

# Média das precisões dos K folds
mean_precision_DecTree <- mean(precision_list)
print(paste("Precisão média da validação cruzada:", mean_precision_DecTree))

# ============================================================================== #

#Algoritmo de Rede Neural (MLP)
if (!require(nnet)) install.packages("nnet")
library(nnet)
train_data$income <- as.factor(train_data$income)

accuracy_list <- c()
precision_list <- c()

for (i in 1:K) {
  # Separar os dados em treinamento e teste
  test_indexes <- which(folds == i, arr.ind = TRUE)
  teste <- train_data[test_indexes, ]
  treino <- train_data[-test_indexes, ]
  
  # Definir os atributos preditores e a variável target
  train_features <- treino[, -ncol(treino)]
  train_labels <- treino$income
  test_features <- teste[, -ncol(teste)]
  test_labels <- teste$income
  
  # Aplicar o algoritmo Rede Neural
  mlp_model <- nnet(income ~ ., data = treino, size = 5, maxit = 1000)
  mlp_pred <- predict(mlp_model, test_features, type = "class")
  
  
  accuracy <- sum(mlp_pred == test_labels) / length(test_labels)
  accuracy_list <- c(accuracy_list, accuracy)
  true_positives <- sum(mlp_pred == 1 & test_labels == 1)
  false_positives <- sum(mlp_pred == 1 & test_labels == 0)
  if (true_positives + false_positives == 0) {
    precision <- 0
  } else {
    precision <- true_positives / (true_positives + false_positives)
    precision_list <- c(precision_list, precision)
  }
  
  cat("Matriz de Confusão para o fold", i, ":\n")
  cat("           Previsto 0   Previsto 1\n")
  cat("Real 0:    ", sum(mlp_pred == 0 & test_labels == 0), "        ", sum(mlp_pred == 1 & test_labels == 0), "\n")
  cat("Real 1:    ", sum(mlp_pred == 0 & test_labels == 1), "        ", sum(mlp_pred == 1 & test_labels == 1), "\n\n")
}


mean_accuracy_mlp <- mean(accuracy_list)
print(paste("Acurácia média da validação cruzada:", mean_accuracy_mlp))


mean_precision_mlp <- mean(precision_list)
print(paste("Precisão média da validação cruzada:", mean_precision_mlp))

# ============================================================================== #


# Comparação dos resultados
cat("Acurácia média da validação cruzada para o algoritmo Baseline:", accuracy_baseline, "\n")
cat("Precisão média da validação cruzada para o algoritmo Baseline: Não definida (nenhuma previsão positiva)\n\n")

cat("Acurácia média da validação cruzada para o algoritmo K-NN:", mean_accuracy_knn, "\n")
cat("Precisão média da validação cruzada para o algoritmo K-NN:", mean_precision_knn, "\n\n")

cat("Acurácia média da validação cruzada para o algoritmo Árvore de Decisão:", mean_accuracy_DecTree, "\n")
cat("Precisão média da validação cruzada para o algoritmo Árvore de Decisão:", mean_precision_DecTree, "\n\n")

cat("Acurácia média da validação cruzada para o algoritmo Rede Neural:", mean_accuracy_mlp, "\n")
cat("Precisão média da validação cruzada para o algoritmo Rede Neural:", mean_precision_mlp, "\n\n")

# ============================================================================== #

# Gráfico barplot (Precisão e Acurácia) e traçar a linha do baseline em um barplot
# Definir os valores de acurácia média para cada algoritmo
algoritmos <- c("Baseline", "K-NN", "Árvore de Decisão", "Rede Neural (MLP)")
acuracias <- c(accuracy_baseline, mean_accuracy_knn, mean_accuracy_DecTree, mean_accuracy_mlp )
baseline_acuracia <- accuracy_baseline

barplot(acuracias, names.arg = algoritmos, col = "lightblue", ylim = c(0, 1),
        main = "Comparação de Acurácia por Algoritmo", ylab = "Acurácia Média")

abline(h = baseline_acuracia, col = "red", lty = 2, lwd = 2)

text(x = 1, y = baseline_acuracia + 0.02, labels = paste("Baseline =", baseline_acuracia), col = "red")

algoritmos <- c("Baseline", "K-NN", "Árvore de Decisão", "Rede Neural (MLP)")
precisoes <- c(0, mean_precision_knn, mean_precision_DecTree, mean_precision_mlp)
baseline_precisao <- 0

barplot(precisoes, names.arg = algoritmos, col = "lightgreen", ylim = c(0, 1),
        main = "Comparação de Precisão por Algoritmo", ylab = "Precisão Média")

abline(h = baseline_precisao, col = "red", lty = 2, lwd = 2)

text(x = 1, y = baseline_precisao + 0.02, labels = paste("Baseline =", baseline_precisao), col = "red")

# ============================================================================== #

# Aplicar o algoritmo Árvore de Decisão na base de teste  
decision_tree <- rpart(income ~ ., data = train_data, method = "class")
rpart_pred <- predict(decision_tree, test_features, type = "class")

true_negatives <- sum(rpart_pred == 0 & test_labels == 0)
false_positives <- sum(rpart_pred == 1 & test_labels == 0)
false_negatives <- sum(rpart_pred == 0 & test_labels == 1)
true_positives <- sum(rpart_pred == 1 & test_labels == 1)

cat("Matriz de Confusão para a base de teste:\n")
cat("           Previsto 0   Previsto 1\n")
cat("Real 0:    ", true_negatives, "        ", false_positives, "\n")
cat("Real 1:    ", false_negatives, "        ", true_positives, "\n\n")


accuracy_rpart_test <- (true_negatives + true_positives) / length(test_labels)
cat("Acurácia na base de teste:", accuracy_rpart_test, "\n")

if (true_positives + false_positives == 0) {
  cat("Precisão na base de teste: Não definida (nenhuma previsão positiva)\n")
} else {
  precision_rpart_test <- true_positives / (true_positives + false_positives)
  cat("Precisão na base de teste:", precision_rpart_test, "\n")
}

#calular recall
recall <- true_positives / (true_positives + false_negatives)
cat("Recall na base de teste:", recall, "\n")


# ============================================================================== #