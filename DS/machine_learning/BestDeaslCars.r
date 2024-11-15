library(ggplot2)
library(ggthemes)
library(randomForest)
library(rpart)
library(caret)
library(ranger)
library(vtreat)
library(dplyr)
library(vcd)


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

# First, let's create functions to test statistical significance
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
numerical_cols <- setdiff(numerical_cols, "priceClassification")

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

# Print dimensions of new dataset
cat("\nOriginal dataset dimensions:", dim(cars), "\n")
cat("New dataset dimensions:", dim(cars_relevant), "\n")

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

# Train a model (example using Random Forest)
library(randomForest)
model <- randomForest(priceClassification ~ ., 
                     data = train_treated,
                     ntree = 500)

# Make predictions on test set
predictions <- predict(model, test_treated)

# Get probability predictions for finding "best deals"
prob_predictions <- predict(model, test_treated, type = "prob")

# Add probabilities to test data first
test_data$great_price_prob <- prob_predictions[, "Great Price"]
test_data$excellent_price_prob <- prob_predictions[, "Excellent Price"]

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

# Optional: Print summary statistics of the top deals
summary_stats <- summary(top_deals[, c("deal_score", "listPrice")])
print("\nSummary Statistics of Top Deals:")
print(summary_stats)