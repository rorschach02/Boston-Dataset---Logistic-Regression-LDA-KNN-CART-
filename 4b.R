### Clear the environment 
rm(list = ls())


### First we will set the directory of the R script 
setwd("C:/Users/anike/Desktop/Sem 1/EAS 506 Statistical Data Mining/Homework/Homework 4")


## Loading all the libraries 
library(ISLR)
library(corrplot)
library(MASS)
library(klaR)
library(leaps)
library(lattice)
library(ggplot2)
library(corrplot)
library(car)
library(caret)
library(class)
#install.packages("rpart")
library(rpart)

# loading the dataset 

data('Boston')
median_crim <- median(Boston$crim)

crim <- ifelse(Boston$crim > median_crim,1,0) 
# If value is above median than value will be 1 or else it will be zero 
Boston <- subset(Boston, select = c(2:14))
Boston <- cbind(crim , Boston)
head(Boston,2)

#Normalize the dataset 

normalize <- function(x) {
  (x -min(x)) / (max(x) - min(x))
  
}

Boston_Norm <- as.data.frame(lapply(Boston[2:14], normalize))
Boston_Norm <- cbind(crim, Boston_Norm)
head(Boston_Norm , 3)

#Splitting in training and testing dataset: 80 - 20 split 
set.seed(1)
sample_size <- floor(0.80 * nrow(Boston))
set.seed(1)
train_indexes <- sample(seq_len(nrow(Boston)), size = sample_size)
training_data <- Boston[train_indexes ,]
testing_data <- Boston[-train_indexes ,]

# Logistic Regression 

logistic_reg_model <- glm(crim~.,data = training_data, family = binomial)

logistic_reg_model_sum <- summary(logistic_reg_model)

#Fitting Logistic Regression Model 
#Predicting Results
logistic_reg_pred_model = predict(logistic_reg_model, newdata = testing_data, type="response")
pred_values = rep(0, length(testing_data$crim))
pred_values[logistic_reg_pred_model > 0.5] = 1

table(Predicted= pred_values , Survived = testing_data$crim)


# Error: 
mean(testing_data$crim != pred_values)


# LDA 
#Performing LDA
lda.model <- lda(crim~.,data = training_data)
lda.model

#Predicting results.
pred.lda.model = predict(lda.model, newdata = testing_data)
table(Predicted=pred.lda.model$class, Survived=testing_data$crim)
test_pred_y <- pred.lda.model$class
# Error: 
mean(testing_data$crim != test_pred_y)


#KNN
# Performing KNN with k= 5
x_train <- subset(training_data , select = -c(1))
x_test <- subset(testing_data , select = -c(1))

set.seed(123)
testing_knn <- knn(x_train , x_test , training_data$crim , k=5)
confusion_matrix_knn <- table(testing_knn , testing_data$crim)
confusion_matrix_knn1<- confusionMatrix(confusion_matrix_knn)


# Accuracy :
round(confusion_matrix_knn1$overall[1]*100 , digits = 2)


# CART : classification using decision trees

model.control <- rpart.control(minsplit = 10, xval = 10 , cp = 0)
fit.training.data <- rpart(crim~. , data = training_data , method = "class" , control = model.control)

#Plotting the tree
x11()
plot(fit.training.data, uniform = T , compress =  T)
text(fit.training.data, cex = 0.5)

#Predicting the testing data 
predict_tree <- predict(fit.training.data, testing_data, type = "class")
table(Predicted=predict_tree, Survived=testing_data$crim)

#Error Rate:
mean(testing_data$crim != predict_tree)
