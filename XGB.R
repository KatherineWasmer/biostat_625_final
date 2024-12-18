# Load necessary libraries
install_packages_and_libraries <- function() {
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
}

# Load and clean data
load_and_clean_data <- function(file_path) {
  clean_data <- read.csv(file_path)
  clean_data$id <- NULL
  clean_data <- na.omit(clean_data)
  return(clean_data)
}

# Preprocess data
preprocess_data <- function(data) {
  data$BP.Category <- dplyr::recode(data$BP.Category,
                                    "Normal" = 1,
                                    "Elevated" = 2,
                                    "High Blood Pressure Stage 1" = 3,
                                    "High Blood Pressure Stage 2" = 4,
                                    "Hypertensive Crisis" = 5)
  data$gender <- dplyr::recode(data$gender,
                               "M" = 0,
                               "F" = 1)
  return(data)
}

# Ensure binary labels
fix_binary_labels <- function(data, target_col) {
  data[[target_col]] <- as.numeric(as.character(data[[target_col]]))
  if (max(data[[target_col]]) > 1) {
    data[[target_col]] <- data[[target_col]] - 1  # Shift to 0 and 1
  }
  return(data)
}

# Split data into training and testing sets
split_data <- function(data, target_col, train_ratio = 0.8, seed = 123) {
  set.seed(seed)
  train_index <- createDataPartition(data[[target_col]], p = train_ratio, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  # Fix labels to ensure 0/1
  train_data <- fix_binary_labels(train_data, target_col)
  test_data <- fix_binary_labels(test_data, target_col)
  
  return(list(train = train_data, test = test_data))
}

# Prepare data for XGBoost
prepare_dmatrix <- function(data, target_col) {
  dmatrix <- xgb.DMatrix(data = as.matrix(data[, -which(names(data) == target_col)]),
                         label = data[[target_col]])
  return(dmatrix)
}

# Perform hyperparameter tuning
tune_xgboost <- function(train_data, target_col, tune_grid, cv_folds = 5, seed = 123) {
  train_control <- trainControl(method = "cv", number = cv_folds, search = "grid")
  set.seed(seed)
  tuning_time <- system.time({
    model <- train(
      as.formula(paste(target_col, "~ .")),
      data = train_data,
      method = "xgbTree",
      trControl = train_control,
      tuneGrid = tune_grid
    )
  })
  return(list(model = model, tuning_time = tuning_time["elapsed"]))
}

# Train final XGBoost model
train_final_xgboost <- function(train_matrix, best_params, nrounds) {
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
      nrounds = nrounds,
      verbose = 0
    )
  })
  return(list(model = model, training_time = training_time["elapsed"]))
}

# Evaluate model performance
evaluate_model <- function(model, test_matrix, true_labels) {
  pred_probs <- predict(model, test_matrix)
  pred_labels <- ifelse(pred_probs > 0.5, 1, 0)
  conf_matrix <- confusionMatrix(factor(pred_labels), factor(true_labels))
  roc_curve <- roc(true_labels, pred_probs)
  auc_score <- auc(roc_curve)
  return(list(conf_matrix = conf_matrix, auc_score = auc_score, roc_curve = roc_curve))
}

# Plot feature importance
plot_feature_importance <- function(model, feature_names) {
  importance_matrix <- xgb.importance(feature_names = feature_names, model = model)
  print(importance_matrix)
  xgb.plot.importance(importance_matrix, main = "XGBoost Feature Importance")
}


