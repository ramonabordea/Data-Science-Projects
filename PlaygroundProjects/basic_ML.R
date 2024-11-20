# Load necessary libraries
library(carData)
library(ggplot2)
library(randomForest)
library(e1071)
library(kernlab)

# Load the cars dataset
data(cars)

data_frame <- read.csv("https://raw.githubusercontent.com/kwartler/Vienna_24/refs/heads/main/Fall_2024/PostModuleAssignment/basic/newCars.csv")

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(cars$Price, p = 0.7, list = FALSE)
trainData <- cars[trainIndex,]
testData <- cars[-trainIndex, ]

# Train a linear regression model
lm_model <- train(Mileage ~ Cylinder, data = trainData, method = "lm")

# Train a random forest model
rf_model <- train(Mileage ~ Cylinder, data = trainData, method = "rf")

# Train a support vector machine model
svm_model <- train(Mileage ~ Cylinder, data = trainData, method = "svmRadial")

# Make predictions on the test set
lm_predictions <- predict(lm_model, testData)
rf_predictions <- predict(rf_model, testData)
svm_predictions <- predict(svm_model, testData)

# Compute performance metrics
lm_rmse <- RMSE(lm_predictions, testData$Mileage)
rf_rmse <- RMSE(rf_predictions, testData$Mileage)
svm_rmse <- RMSE(svm_predictions, testData$Mileage)

# Print the performance metrics
cat("Linear Regression RMSE:", lm_rmse, "\n")
cat("Random Forest RMSE:", rf_rmse, "\n")
cat("SVM RMSE:", svm_rmse, "\n")

# Plot the predictions
ggplot(testData, aes(x = Mileage, y = Cylinder)) +
    geom_line(aes(y = lm_predictions), color = "blue", linetype = "dashed") +
    geom_line(aes(y = rf_predictions), color = "green", linetype = "dashed") +
    geom_line(aes(y = svm_predictions), color = "red", linetype = "dashed") +
    labs(title = "Model Predictions vs Actual Data",
             x = "Cylinderance",
             y = "Mileage") +
    theme_minimal()