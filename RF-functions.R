
###data preprocessing
preprocess<-function(data){
  data<-na.omit(data)
  data$cardio<-as.factor(data$cardio)
  data$BP.Category <- data$BP.Category %>% recode(
    "Normal" = 1,
    "Elevated" = 2,
    "High Blood Pressure Stage 1" = 3,
    "High Blood Pressure Stage 2" = 4,
    "Hypertensive Crisis" = 5,
  )
  
  data$gender <- data$gender %>% recode(
    "M" = 0,
    "F" = 1
  )
  return(data)
}

###data preprocessing and division
split_data<-function(data){
  data$cardio <- as.factor(data$cardio) 
  set.seed(123)
  trainIndex <- sample(1:nrow(data), size = 0.8 * nrow(data))
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]
  invisible(list(trainData,testData))
}

###fit the model
tune_train_RF <- function(trainData) {
  start_time <- Sys.time()
  
  #set the search scope
  param_grid <- expand.grid(
    mtry = c(3, 4, 5),
    num.trees = c(100, 200, 300),
    min.node.size = c(1, 3, 5)
  )
  
  #cross-validation coefficients
  set.seed(123)
  folds <- createFolds(trainData$cardio, k = 10, list = TRUE, returnTrain = TRUE)
  
  results <- data.frame(
    mtry = integer(),
    num.trees = integer(),
    min.node.size = integer(),
    Accuracy = numeric()
  )
  
  #finding the best parameters, using accuracy as criteria
  for (i in 1:nrow(param_grid)) {
    mtry_val <- param_grid$mtry[i]
    num_trees_val <- param_grid$num.trees[i]
    min_node_val <- param_grid$min.node.size[i]
    
    accuracy_list <- c()
    
    for (fold in folds) {
      train_fold <- trainData[fold, ]
      valid_fold <- trainData[-fold, ]
      
      model <- ranger(
        cardio ~ .,
        data = train_fold,
        mtry = mtry_val,
        num.trees = num_trees_val,
        min.node.size = min_node_val,
        probability = TRUE
      )
      
      predictions <- predict(model, data = valid_fold)$predictions
      pred_labels <- ifelse(predictions[, 2] > 0.5, "1", "0")

      acc <- mean(pred_labels == valid_fold$cardio)
      accuracy_list <- c(accuracy_list, acc)
    }
    
    #compute average accuracy and finding the best one
    avg_acc <- mean(accuracy_list)
    
    results <- rbind(results, data.frame(
      mtry = mtry_val,
      num.trees = num_trees_val,
      min.node.size = min_node_val,
      Accuracy = avg_acc
    ))
  }
  best_params <- results %>% filter(Accuracy == max(Accuracy))
  print(best_params)
  
  #using the best parameter to fit a best model
  best_rf_model <- ranger(
    cardio ~ .,
    data = trainData,
    mtry = best_params$mtry,
    num.trees = best_params$num.trees,
    min.node.size = best_params$min.node.size,
    probability = TRUE,
    importance = "impurity"
  )
  
  end_time <- Sys.time()
  run_time <- end_time - start_time
  
  return(list(best_rf_model = best_rf_model, best_params = best_params, run_time = run_time))
}

###compute the evaluaiton metrics
evaluate_RF <- function(testData, best_rf_model) {
  
  test_pred_prob <- predict(best_rf_model, data = testData)$predictions
  test_pred_prob_yes <- test_pred_prob[, "1"]
  
  test_pred_labels <- ifelse(test_pred_prob_yes > 0.5, "1", "0")
  
  #confusion matrix
  conf_matrix <- confusionMatrix(factor(test_pred_labels, levels = c("0", "1")), testData$cardio)
  
  #roc and auc
  roc_curve <- roc(testData$cardio, test_pred_prob_yes)
  auc_value <- auc(roc_curve)
  
  #accuracy,recall and precision
  accuracy <- conf_matrix$overall["Accuracy"]
  recall <- conf_matrix$byClass["Sensitivity"]
  precision <- conf_matrix$byClass["Pos Pred Value"]

  result <- list(
    confusion_matrix = conf_matrix,
    accuracy = accuracy,
    recall = recall,
    precision = precision,
    auc = auc_value,
    roc = roc_curve
  )
  return(result)
}

