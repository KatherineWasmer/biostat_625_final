# Load necessary libraries
install.packages("xgboost")
install.packages("caret")
install.packages("ggplot2")
install.packages("pROC")
install.packages("dplyr") 
library(dplyr)
library(xgboost)
library(caret)
library(ggplot2)
library(pROC)

# 1. Load and Clean Data
load_and_clean_data <- function(file_path) {
  data <- read.csv(file_path)
  data$id <- NULL  # Remove ID column
  data <- na.omit(data)  # Remove missing values
  return(data)
}

# 2. Preprocess Data
preprocess_data <- function(data) {
  data$BP.Category <- dplyr::recode(as.character(data$BP.Category),
                                    "Normal" = 1,
                                    "Elevated" = 2,
                                    "High Blood Pressure Stage 1" = 3,
                                    "High Blood Pressure Stage 2" = 4,
                                    "Hypertensive Crisis" = 5)
  data$gender <- dplyr::recode(data$gender, "M" = 0, "F" = 1)
  data$cardio <- as.factor(data$cardio)
  return(data)
}



# 3. Split Data into Training and Testing
split_data <- function(data, split_ratio = 0.8, seed = 123) {
  set.seed(seed)
  train_index <- createDataPartition(data$cardio, p = split_ratio, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  return(list(train = train_data, test = test_data))
}

# 4. Train XGBoost Model with Cross-Validation
train_xgb_model <- function(train_data) {
  train_control <- trainControl(method = "cv", number = 5, search = "random")
  tune_grid <- expand.grid(
    nrounds = c(50, 100),      
    max_depth = c(3, 6),          
    eta = c(0.01, 0.1),        
    gamma = c(0, 1),              
    colsample_bytree = c(0.5,0.8), 
    min_child_weight = c(1,3,5),   
    subsample = c(0.5, 0.8)        
  )
  set.seed(123)
  xgb_tuned <- train(
    cardio ~ .,
    data = train_data,
    method = "xgbTree",
    trControl = train_control,
    tuneGrid = tune_grid
  )
  return(xgb_tuned)
}

# 5. Train Final XGBoost Model
train_final_xgb <- function(train_matrix, best_params) {
  training_time <- system.time({
    model <- xgboost(
      params = list(
        booster = "gbtree",
        objective = "binary:logistic",
        eta = best_params$eta,
        max_depth = best_params$max_depth,
        gamma = best_params$gamma,
        colsample_bytree = best_params$colsample_bytree,
        min_child_weight = best_params$min_child_weight,
        subsample = best_params$subsample
      ),
      data = train_matrix,
      nrounds = best_params$nrounds,
      verbose = 0
    )
  })
  return(list(model = model, training_time = training_time))
}

# 6. Evaluate Model
evaluate_model <- function(model, test_data) {
  test_matrix <- xgb.DMatrix(data = as.matrix(test_data[, -which(names(test_data) == "cardio")]),
                             label = as.numeric(as.character(test_data$cardio)))
  pred_probs <- predict(model, test_matrix)
  pred_labels <- ifelse(pred_probs > 0.5, 1, 0)
  
  # Confusion Matrix
  conf_matrix <- confusionMatrix(factor(pred_labels), factor(test_data$cardio))
  print(conf_matrix)
  
  
  # ROC-AUC Curve
  roc_curve <- roc(test_data$cardio, pred_probs)
  print(auc(roc_curve))
  plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for XGBoost Model")
}


# 7. Plot Feature Importance
plot_feature_importance <- function(model, train_data) {
  importance_matrix <- xgb.importance(feature_names = colnames(train_data[, -which(names(train_data) == "cardio")]),
                                      model = model)
  print(importance_matrix)
  xgb.plot.importance(importance_matrix, main = "XGBoost Feature Importance")
}

