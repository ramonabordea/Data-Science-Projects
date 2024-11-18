if (!require("xgboost")) install.packages("xgboost")
library(xgboost)
library(ggplot2)
library(ggthemes)
library(randomForest)
library(rpart)
library(caret)
library(ranger)
library(vtreat)
library(dplyr)
library(vcd)
library(rpart)
library(rpart.plot)
library(pROC)
library(reshape2)



cars <- read.csv(paste0('https://raw.githubusercontent.com/kwartler/Vienna_24/',
                        'refs/heads/main/Fall_2024/PostModuleAssignment/',
                        'basic/newCars.csv'))
summary(cars)


colnames(cars)

# Transform mileageRatingCity to numeric
cars$mileageRatingCity <- as.numeric(gsub("[^0-9.]", "", cars$mileageRatingCity))

# Do the same for mileageRatingHighway for consistency
cars$mileageRatingHighway <- as.numeric(gsub("[^0-9.]", "", cars$mileageRatingHighway))

# Handle NA values in numeric columns
numeric_cols <- sapply(cars, is.numeric)
for(col in names(cars)[numeric_cols]) {
  # Replace NA with median for numeric columns
  cars[[col]][is.na(cars[[col]])] <- median(cars[[col]], na.rm = TRUE)
}

# Handle NA values in categorical columns
categorical_cols <- sapply(cars, is.factor) | sapply(cars, is.character)
for(col in names(cars)[categorical_cols]) {
  # Replace NA with mode (most frequent category)
  mode_val <- names(sort(table(cars[[col]]), decreasing = TRUE))[1]
  cars[[col]][is.na(cars[[col]])] <- mode_val
}

# Verify no NA values remain
sum(is.na(cars))

# create functions to test statistical significance
test_categorical <- function(data, var, target, threshold = 0.05) {
  # Chi-square test
  cont_table <- table(data[[var]], data[[target]])
  test_result <- chisq.test(cont_table)
  return(test_result$p.value < threshold)
}

test_numerical <- function(data, var, target, threshold = 0.05) {
  # ANOVA test
  formula <- as.formula(paste(var, "~", target))
  test_result <- summary(aov(formula, data = data))
  return(test_result[[1]]$`Pr(>F)`[1] < threshold)
}

# Identify categorical and numerical columns
categorical_cols <- names(cars)[sapply(cars, function(x) is.factor(x) | is.character(x))]
numerical_cols <- names(cars)[sapply(cars, is.numeric)]

# Remove target variable from the lists
categorical_cols <- setdiff(categorical_cols, "priceClassification")

# Test each variable and store results
relevant_categorical <- sapply(categorical_cols, function(x) {
  tryCatch({
    test_categorical(cars, x, "priceClassification")
  }, error = function(e) FALSE)
})

relevant_numerical <- sapply(numerical_cols, function(x) {
  tryCatch({
    test_numerical(cars, x, "priceClassification")
  }, error = function(e) FALSE)
})

# Get names of relevant columns
relevant_cat_cols <- names(relevant_categorical)[relevant_categorical]
relevant_num_cols <- names(relevant_numerical)[relevant_numerical]

# Create new dataset with only relevant columns
relevant_cols <- c(relevant_cat_cols, relevant_num_cols, "priceClassification")
cars_relevant <- cars[, relevant_cols]

# Print removed and kept columns
cat("Removed columns:\n")
print(setdiff(names(cars), relevant_cols))
cat("\nKept columns:\n")
print(relevant_cols)

# First, create train/test split
set.seed(123) # for reproducibility
train_idx <- createDataPartition(cars_relevant$priceClassification, p = 0.7, list = FALSE)
train_data <- cars_relevant[train_idx, ]
test_data <- cars_relevant[-train_idx, ]

targetVar       <- names(cars_relevant)[24]
informativeVars <- names(cars_relevant)[1:23]

# Create treatment plan
str(train_data)

# 2. Make sure informativeVars doesn't include the target variable
informativeVars <- setdiff(names(train_data), "priceClassification")

# 3. Check the target variable
print("Target variable values:")
print(table(train_data$priceClassification, useNA = "ifany"))

# 4. Clean and prepare the target variable
train_data$priceClassification <- factor(train_data$priceClassification, 
                                         levels = unique(train_data$priceClassification))


# Create cross-frame experiment instead of treatment plan to avoid over fitting
cross_frame_experiment <- mkCrossFrameCExperiment(
  dframe = train_data,
  varlist = informativeVars,
  outcomename = "priceClassification",
  outcometarget = levels(train_data$priceClassification)[1],  # Explicitly set target level
  verbose = FALSE
)

# Get the treated training data from the cross frame
train_treated <- cross_frame_experiment$crossFrame

# Apply treatment plan to test data
test_treated <- prepare(
  treatmentplan = cross_frame_experiment$treatments,
  dframe = test_data,
  varRestriction = cross_frame_experiment$treatments$scoreFrame$varName
)


#------------------------------------------------------------------------------------
# Train a model (example using Random Forest)

modelRF <- randomForest(priceClassification ~ ., 
                      data = train_treated,
                      ntree = 500)
print(modelRF)

# Get probability predictions for finding "best deals"
prob_predictions <- predict(modelRF, test_treated, type = "prob")

# Create confusion matrix
#conf_matrix <- confusionMatrix(prob_predictions, test_data$priceClassification)

# Add probabilities to test data first
test_data$great_price_prob <- prob_predictions[, "Great Price"]
test_data$excellent_price_prob <- prob_predictions[, "Excellent Price"]

# Create binary indicators for each class
test_data$is_excellent <- ifelse(test_data$priceClassification == "Excellent Price", 1, 0)
test_data$is_great <- ifelse(test_data$priceClassification == "Great Price", 1, 0)

# Add probability predictions
test_data$excellent_price_prob <- prob_predictions[, "Excellent Price"]
test_data$great_price_prob <- prob_predictions[, "Great Price"]

# Calculate accuracy for Excellent Price predictions (using 0.5 as threshold)
excellent_predictions <- ifelse(test_data$excellent_price_prob > 0.5, 1, 0)
excellent_accuracy <- mean(excellent_predictions == test_data$is_excellent)

# Calculate accuracy for Great Price predictions
great_predictions <- ifelse(test_data$great_price_prob > 0.5, 1, 0)
great_accuracy <- mean(great_predictions == test_data$is_great)

# Print results
print("Model Performance Metrics:")
print(paste("Excellent Price Accuracy:", round(excellent_accuracy, 3)))
print(paste("Great Price Accuracy:", round(great_accuracy, 3)))


# ROC curves and AUC
roc_excellent <- roc(test_data$is_excellent, test_data$excellent_price_prob)
roc_great <- roc(test_data$is_great, test_data$great_price_prob)

print(paste("Excellent Price AUC:", round(auc(roc_excellent), 3)))
print(paste("Great Price AUC:", round(auc(roc_great), 3)))

# Visualize ROC curves
plot(roc_excellent, main="ROC Curves", col="blue")
lines(roc_great, col="red")
legend("bottomright", legend=c("Excellent Price", "Great Price"), 
       col=c("blue", "red"), lwd=2)

# Create correlation plot

# For Excellent Price
ggplot(test_data, aes(x=factor(is_excellent), y=excellent_price_prob)) +
  geom_boxplot() +
  labs(title="Excellent Price: Actual vs Predicted",
       x="Actual (1=Excellent Price)", 
       y="Predicted Probability") +
  theme_minimal()

# For Great Price
ggplot(test_data, aes(x=factor(is_great), y=great_price_prob)) +
  geom_boxplot() +
  labs(title="Great Price: Actual vs Predicted",
       x="Actual (1=Great Price)", 
       y="Predicted Probability") +
  theme_minimal()


# Calculate deal score
test_data$deal_score <- test_data$great_price_prob + test_data$excellent_price_prob

# Find top 100 best deals
top_deals <- test_data[order(test_data$deal_score, decreasing = TRUE), ][1:100, ]

# Print top 10 deals
print("Top 10 Best Deals:")
head(top_deals[, c("listPrice", "priceClassification", "deal_score", 
                   "great_price_prob", "excellent_price_prob")], 10)

# Save results
write.csv(top_deals, "top_100_deals.csv", row.names = FALSE)

# Print summary statistics of the top deals
summary_stats <- summary(top_deals[, c("deal_score", "listPrice")])
print("\nSummary Statistics of Top Deals:")
print(summary_stats)

# Visualize distribution of deal scores in top 100
ggplot(top_deals, aes(x = deal_score)) +
  geom_histogram(bins = 20) +
  theme_minimal() +
  ggtitle("Distribution of Deal Scores in Top 100")


# Train decision tree model
tree_model <- rpart(priceClassification ~ ., 
                   data = train_treated,
                   method = "class")

# Get probability predictions for decision tree
tree_prob_predictions <- predict(tree_model, test_treated, type = "prob")

# Create comparison dataframe
test_data$is_excellent <- ifelse(test_data$priceClassification == "Excellent Price", 1, 0)
test_data$is_great <- ifelse(test_data$priceClassification == "Great Price", 1, 0)

# Add probability predictions for both models
test_data$rf_excellent_prob <- prob_predictions[, "Excellent Price"]
test_data$rf_great_prob <- prob_predictions[, "Great Price"]
test_data$tree_excellent_prob <- tree_prob_predictions[, "Excellent Price"]
test_data$tree_great_prob <- tree_prob_predictions[, "Great Price"]

# Calculate predictions using 0.5 threshold
test_data$rf_excellent_pred <- ifelse(test_data$rf_excellent_prob > 0.5, 1, 0)
test_data$rf_great_pred <- ifelse(test_data$rf_great_prob > 0.5, 1, 0)
test_data$tree_excellent_pred <- ifelse(test_data$tree_excellent_prob > 0.5, 1, 0)
test_data$tree_great_pred <- ifelse(test_data$tree_great_prob > 0.5, 1, 0)

# Function to calculate all metrics
calc_all_metrics <- function(actual, predicted, probs) {
  accuracy <- mean(actual == predicted)
  roc_curve <- roc(actual, probs)
  auc_score <- auc(roc_curve)
  
  true_pos <- sum(actual == 1 & predicted == 1)
  false_pos <- sum(actual == 0 & predicted == 1)
  false_neg <- sum(actual == 1 & predicted == 0)
  
  precision <- true_pos / (true_pos + false_pos)
  recall <- true_pos / (true_pos + false_neg)
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  return(c(accuracy=accuracy, auc=auc_score, 
           precision=precision, recall=recall, f1=f1))
}

# Calculate metrics for both models
rf_excellent_metrics <- calc_all_metrics(test_data$is_excellent, 
                                       test_data$rf_excellent_pred,
                                       test_data$rf_excellent_prob)
rf_great_metrics <- calc_all_metrics(test_data$is_great,
                                   test_data$rf_great_pred,
                                   test_data$rf_great_prob)
tree_excellent_metrics <- calc_all_metrics(test_data$is_excellent,
                                         test_data$tree_excellent_pred,
                                         test_data$tree_excellent_prob)
tree_great_metrics <- calc_all_metrics(test_data$is_great,
                                     test_data$tree_great_pred,
                                     test_data$tree_great_prob)

# Print comparison results
print("Model Performance Comparison:")
print("\nExcellent Price Predictions:")
comparison_excellent <- rbind(
  "Random Forest" = rf_excellent_metrics,
  "Decision Tree" = tree_excellent_metrics
)
print(round(comparison_excellent, 3))

print("\nGreat Price Predictions:")
comparison_great <- rbind(
  "Random Forest" = rf_great_metrics,
  "Decision Tree" = tree_great_metrics
)
print(round(comparison_great, 3))

# Visualize ROC curves
par(mfrow=c(1,2))

# Excellent Price ROC curves
plot(roc(test_data$is_excellent, test_data$rf_excellent_prob), 
     main="ROC Curves - Excellent Price",
     col="blue")
lines(roc(test_data$is_excellent, test_data$tree_excellent_prob), 
      col="red")
legend("bottomright", 
       legend=c("Random Forest", "Decision Tree"), 
       col=c("blue", "red"), 
       lwd=2)

# Great Price ROC curves
plot(roc(test_data$is_great, test_data$rf_great_prob), 
     main="ROC Curves - Great Price",
     col="blue")
lines(roc(test_data$is_great, test_data$tree_great_prob), 
      col="red")
legend("bottomright", 
       legend=c("Random Forest", "Decision Tree"), 
       col=c("blue", "red"), 
       lwd=2)

# Visualize decision tree
rpart.plot(tree_model, 
          main="Decision Tree Model",
          extra=101,
          under=TRUE,
          box.palette="RdYlGn")

# Prepare data for plotting
prob_comparison <- data.frame(
  Actual = c(test_data$is_excellent, test_data$is_excellent),
  Probability = c(test_data$rf_excellent_prob, test_data$tree_excellent_prob),
  Model = c(rep("Random Forest", nrow(test_data)), 
           rep("Decision Tree", nrow(test_data)))
)

# Create boxplot
ggplot(prob_comparison, aes(x=factor(Actual), y=Probability, fill=Model)) +
  geom_boxplot() +
  facet_wrap(~Model) +
  labs(title="Probability Distribution Comparison - Excellent Price",
       x="Actual Class (1=Excellent Price)",
       y="Predicted Probability") +
  theme_minimal()



# Prepare data for xgboost (convert target to numeric)
labels <- as.numeric(factor(train_treated$priceClassification)) - 1
test_labels <- as.numeric(factor(test_treated$priceClassification)) - 1

# Create DMatrix objects
dtrain <- xgb.DMatrix(as.matrix(train_treated[, !names(train_treated) %in% "priceClassification"]), 
                      label = labels)
dtest <- xgb.DMatrix(as.matrix(test_treated[, !names(test_treated) %in% "priceClassification"]), 
                     label = test_labels)

# Set parameters for xgboost
params <- list(
  objective = "multi:softprob",
  num_class = length(unique(labels)),
  eta = 0.3,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8
)

# Train xgboost model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,
  verbose = 0
)

# Get probability predictions
xgb_prob_predictions <- predict(xgb_model, dtest)
xgb_prob_predictions <- matrix(xgb_prob_predictions, 
                             ncol = length(unique(labels)), 
                             byrow = TRUE)
colnames(xgb_prob_predictions) <- levels(factor(train_treated$priceClassification))

# Add XGBoost probabilities to test_data
test_data$xgb_excellent_prob <- xgb_prob_predictions[, "Excellent Price"]
test_data$xgb_great_prob <- xgb_prob_predictions[, "Great Price"]

# Calculate XGBoost predictions using 0.5 threshold
test_data$xgb_excellent_pred <- ifelse(test_data$xgb_excellent_prob > 0.5, 1, 0)
test_data$xgb_great_pred <- ifelse(test_data$xgb_great_prob > 0.5, 1, 0)

# Calculate metrics for XGBoost
xgb_excellent_metrics <- calc_all_metrics(test_data$is_excellent,
                                        test_data$xgb_excellent_pred,
                                        test_data$xgb_excellent_prob)
xgb_great_metrics <- calc_all_metrics(test_data$is_great,
                                     test_data$xgb_great_pred,
                                     test_data$xgb_great_prob)

# Print updated comparison results
print("Model Performance Comparison:")
print("\nExcellent Price Predictions:")
comparison_excellent <- rbind(
  "Random Forest" = rf_excellent_metrics,
  "Decision Tree" = tree_excellent_metrics,
  "XGBoost" = xgb_excellent_metrics
)
print(round(comparison_excellent, 3))

print("\nGreat Price Predictions:")
comparison_great <- rbind(
  "Random Forest" = rf_great_metrics,
  "Decision Tree" = tree_great_metrics,
  "XGBoost" = xgb_great_metrics
)
print(round(comparison_great, 3))

# Variable importance for XGBoost
importance_matrix <- xgb.importance(feature_names = colnames(train_treated[, !names(train_treated) %in% "priceClassification"]), 
                                  model = xgb_model)
print("XGBoost Variable Importance:")
print(head(importance_matrix, 10))

# Plot variable importance
xgb.plot.importance(importance_matrix[1:10,])

# Updated ROC curves with XGBoost
par(mfrow=c(1,2))

# Excellent Price ROC curves
plot(roc(test_data$is_excellent, test_data$rf_excellent_prob),
     main="ROC Curves - Excellent Price",
     col="blue")
lines(roc(test_data$is_excellent, test_data$tree_excellent_prob),
      col="red")
lines(roc(test_data$is_excellent, test_data$xgb_excellent_prob),
      col="green")
legend("bottomright",
       legend=c("Random Forest", "Decision Tree", "XGBoost"),
       col=c("blue", "red", "green"),
       lwd=2)

# Great Price ROC curves
plot(roc(test_data$is_great, test_data$rf_great_prob),
     main="ROC Curves - Great Price",
     col="blue")
lines(roc(test_data$is_great, test_data$tree_great_prob),
      col="red")
lines(roc(test_data$is_great, test_data$xgb_great_prob),
      col="green")
legend("bottomright",
       legend=c("Random Forest", "Decision Tree", "XGBoost"),
       col=c("blue", "red", "green"),
       lwd=2)

# After all model comparisons and evaluations...

# Choose the best model based on metrics (let's use all models for comparison)
test_data$combined_score <- (
  # Random Forest predictions
  test_data$rf_excellent_prob * 1.0 + test_data$rf_great_prob * 0.7 +
  # Decision Tree predictions
  test_data$tree_excellent_prob * 1.0 + test_data$tree_great_prob * 0.7 +
  # XGBoost predictions
  test_data$xgb_excellent_prob * 1.0 + test_data$xgb_great_prob * 0.7
) / 3  # Average across models

# Find top 100 best deals
top_deals <- test_data[order(test_data$combined_score, decreasing = TRUE), ][1:100, ]

# Create detailed results dataframe
results <- data.frame(
  listPrice = top_deals$listPrice,
  priceClassification = top_deals$priceClassification,
  combined_score = top_deals$combined_score,
  rf_excellent_prob = top_deals$rf_excellent_prob,
  rf_great_prob = top_deals$rf_great_prob,
  tree_excellent_prob = top_deals$tree_excellent_prob,
  tree_great_prob = top_deals$tree_great_prob,
  xgb_excellent_prob = top_deals$xgb_excellent_prob,
  xgb_great_prob = top_deals$xgb_great_prob
)

# Print top 10 deals
print("Top 10 Best Deals:")
print(head(results, 10))

# Save results
write.csv(results, "top_100_deals.csv", row.names = FALSE)

# Visualize distribution of scores in top deals
ggplot(top_deals, aes(x = combined_score)) +
  geom_histogram(bins = 20, fill = "blue", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Distribution of Combined Scores in Top 100 Deals",
       x = "Combined Score",
       y = "Count")

# Compare model predictions for top deals
top_deals_long <- reshape2::melt(results[, c("rf_excellent_prob", 
                                           "tree_excellent_prob", 
                                           "xgb_excellent_prob")],
                               variable.name = "model",
                               value.name = "excellent_prob")

ggplot(top_deals_long, aes(x = model, y = excellent_prob)) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal() +
  labs(title = "Model Predictions Comparison for Top 100 Deals",
       x = "Model",
       y = "Probability of Excellent Price")

# Optional: Add price distribution visualization
ggplot(top_deals, aes(x = listPrice)) +
  geom_histogram(bins = 20, fill = "green", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Price Distribution of Top 100 Deals",
       x = "List Price",
       y = "Count")

