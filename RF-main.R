library(ranger)
library(caret)
library(dplyr)
library(pROC)
library(randomForest)

#loading data
original_data <- read.csv("cleaned_data_v2.csv", header = TRUE)
data <- original_data[ , !(names(original_data) %in% c("id"))] #remove id column

#preprocess
data <- preprocess(data)

#split the dataset
train_test <- split_data(data)
trainData <- train_test[[1]]
testData <- train_test[[2]]

#fit the model
rf_model<-tune_train_RF(trainData)
best_rf_model<-rf_model[[1]]

#compute evaluation metrics
metrics<-evaluate_RF(testData,best_rf_model)
roc_curve<-result[[6]]

plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Best Random Forest Model")
abline(a = 0, b = 1, lty = 2, col = "gray")
