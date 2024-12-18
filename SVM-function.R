library(VIM)
library(e1071)
library(dplyr)
library(caret)

#Preprocessing part
preprocess <- function(data) {
  data<-na.omit(data)
  data <- data[, -1]
  data <- data %>%
    mutate(BP.Category = na_if(BP.Category, ""))
  data <- data %>%
    mutate(
      BP.Category = dplyr::recode(BP.Category,
                                  "Normal" = 1,
                                  "Elevated" = 2,
                                  "High Blood Pressure Stage 1" = 3,
                                  "High Blood Pressure Stage 2" = 4,
                                  "Hypertensive Crisis" = 5),
      gender = dplyr::recode(gender,
                             "M" = 0,
                             "F" = 1)
    )
  return(data)
}

split_data<-function(data){
  set.seed(123)
  train_size <- floor(0.8 * nrow(data))
  train_index<-sample(seq_len(nrow(data)), size = train_size)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  return(list(train_data,test_data))
}

svm_process<-function(train_data){
  start_time<-Sys.time()

  model <- svm(cardio ~ ., data = train_data, kernel = "radial", cost=1, gamma=1)
  predictions <- predict(model, test_data)
  binary_predictions <- ifelse(predictions > 0.5, 1, 0)
  confusion_matrix <- table(Predicted = binary_predictions, Actual = test_data$cardio)

  end_time<-Sys.time()
  run_time<-end_time-start_time

  return(list(model,predictions,binary_predictions,confusion_matrix,run_time))
}


evaluate_model <- function(confusion_matrix,predictions,actual) {

  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  precision <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
  recall <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])

  roc_curve <- roc(actual, predictions)
  auc_value <- auc(roc_curve)
  return(list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    auc = auc_value,
    roc = roc_curve
  ))
}
