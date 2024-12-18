###Complete process
data<-read.csv(file="cleaned_data_v2.csv",header=T)
data<-preprocess(data)
train_test<-split_data(data)
train_data<-train_test[[1]]
test_data<-train_test[[2]]

###Parameter and kernal adjustment
model_linear <- svm(cardio ~ ., data = train_data, kernel = "linear")
model_grid <- train(cardio ~ ., data = train_data,
                    method = "svmRadial",
                    trControl = trainControl(method = "cv", number = 10),
                    tuneGrid = expand.grid(C = c(0.1,1,10), sigma = c(0.01,0.1,1)))
#Get the best parameter, c=1, gamma=1

svm_model<-svm_process(train_data)
model<-svm_model[[1]]
predictions<-svm_model[[2]]
binary_predictions<-svm_model[[3]]
confusion_matrix<-svm_model[[4]]
run_time<-svm_model[[5]]
actual<-test_data$cardio

metrics<-evaluate_model(confusion_matrix,predictions,actual)
roc_curve<-metrics[[5]]

plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Best SVM Model")
abline(a = 0, b = 1, lty = 2, col = "gray")

