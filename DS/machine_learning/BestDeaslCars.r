install.packages("vcd")


#### Libraries
library(ggplot2)
library(ggthemes)
library(randomForest)
library(rpart)
library(caret)
library(ranger)
library(vtreat)
library(dplyr)
library(vcd)


cars <- read.csv('https://raw.githubusercontent.com/kwartler/Vienna_24/refs/heads/main/Fall_2024/PostModuleAssignment/basic/newCars.csv')
summary(cars)


colnames(cars)

# ... existing code ...

# 1. For categorical variables - Create contingency tables and chi-square tests
categorical_vars <- c("vehicleHistUseType", "vehicleHistTitleStatus", 
                      "transmissionType", "fuelType", "driveType", 
                      "interiorColor", "exteriorColor", "style", "state")

for(var in categorical_vars) {
  # Create contingency table
  cont_table <- table(cars[[var]], cars$priceClassification)
  print(paste("Contingency table for", var))
  print(cont_table)
  
  # Chi-square test
  chi_test <- chisq.test(cont_table)
  print(paste("Chi-square test for", var))
  print(chi_test)
}

# 2. For numerical variables - Box plots and ANOVA
numerical_vars <- c("mileage", "monthlyPayment", "priceClassDiffAmt", 
                    "mileageRatingCity", "mileageRatingHighway", 
                    "accidentN", "ownersN", "listPrice")

# Create box plots
for(var in numerical_vars) {
  p <- ggplot(cars, aes_string(x = "priceClassification", y = var)) +
    geom_boxplot() +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle(paste(var, "by Price Classification"))
  print(p)
}

# 3. Summary statistics by price classification
summary_stats <- cars %>%
  group_by(priceClassification) %>%
  summarise(
    avg_mileage = mean(mileage, na.rm = TRUE),
    avg_listPrice = mean(listPrice, na.rm = TRUE),
    avg_monthlyPayment = mean(monthlyPayment, na.rm = TRUE),
    avg_accidents = mean(accidentN, na.rm = TRUE),
    avg_owners = mean(ownersN, na.rm = TRUE),
    count = n()
  )
print(summary_stats)

# 4. Visualize the distribution of price classification
ggplot(cars, aes(x = priceClassification)) +
  geom_bar() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Distribution of Price Classification")

# 5. Optional: Create a mosaic plot for key categorical variables

mosaic(~ priceClassification + fuelType + driveType, data = cars,
       main = "Mosaic Plot of Price Classification by Fuel and Drive Type")

# Check the number of NA values in interiorColor
na_count <- sum(is.na(cars$interiorColor))
print(paste("Number of NA values in interiorColor:", na_count))

# See the distribution of non-NA values
interior_dist <- table(cars$interiorColor, useNA = "ifany")
print("Distribution of interior colors (including NA):")
print(interior_dist)

# Calculate percentage of NA values
na_percentage <- (na_count / nrow(cars)) * 100
print(paste("Percentage of NA values:", round(na_percentage, 2), "%"))

# Visualization of interior colors (excluding NA)
ggplot(subset(cars, !is.na(interiorColor)), 
       aes(x = reorder(interiorColor, table(interiorColor)[interiorColor]))) +
  geom_bar() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Interior Color", 
       y = "Count",
       title = "Distribution of Interior Colors (excluding NA)",
       subtitle = paste0("NA values: ", na_count, " (", round(na_percentage, 2), "%)"))

# Relationship with price classification (excluding NA)
ggplot(subset(cars, !is.na(interiorColor)), 
       aes(x = interiorColor, fill = priceClassification)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Interior Color", 
       y = "Proportion",
       title = "Interior Color vs Price Classification",
       subtitle = "Proportional distribution")

# replace NA in interiorColor with Unknown
cars$interiorColor[is.na(cars$interiorColor)] <- "Unknown"

# Count NA values in each column
na_counts <- sapply(cars, function(x) sum(is.na(x)))

# Calculate percentage of NA values
na_percentages <- (na_counts / nrow(cars)) * 100

# Combine counts and percentages into a data frame
na_summary <- data.frame(
  Column = names(na_counts),
  NA_Count = na_counts,
  NA_Percentage = round(na_percentages, 2)
)

# Sort by number of NAs in descending order
na_summary <- na_summary[order(-na_summary$NA_Count), ]

# Print only columns that have NA values
na_summary_with_na <- na_summary[na_summary$NA_Count > 0, ]
print(na_summary_with_na)

# Optional: Visualize NA counts
ggplot(na_summary_with_na, aes(x = reorder(Column, NA_Count), y = NA_Count)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Columns", 
       y = "Number of NA values",
       title = "NA Values by Column")


# Find the columns that are statistically relevant 
# 1. For Numerical Variables - ANOVA test
numerical_vars <- c("mileage", "listPrice", "monthlyPayment", 
                    "mileageRatingCity", "mileageRatingHighway",
                    "accidentN", "ownersN", "priceClassDiffAmt")

# Function to run ANOVA and return p-value
anova_test <- function(var_name) {
  formula <- as.formula(paste(var_name, "~ priceClassification"))
  model <- aov(formula, data = relevant_cars)
  p_value <- summary(model)[[1]]$"Pr(>F)"[1]
  return(p_value)
}

# Run ANOVA for each numerical variable
numerical_results <- data.frame(
  Variable = numerical_vars,
  P_Value = sapply(numerical_vars, function(x) {
    tryCatch(anova_test(x), error = function(e) NA)
  })
)

# Remove mileageRatingCity and mileageRatingHighway from cars dataset
cars <- cars %>% 
  select(-mileageRatingCity, -mileageRatingHighway, -driveType, -style)

# Verify the columns are removed
print("Updated column names:")
print(colnames(cars))

# 2. For Categorical Variables - Chi-square test
categorical_vars <- c("vehicleHistUseType", "transmissionType", 
                      "fuelType", "driveType")

# Function to run Chi-square test and return p-value
chi_square_test <- function(var_name) {
  contingency_table <- table(relevanncy_table)
  return(test_result$p.value)
}
t_cars[[var_name]], 
                             relevant_cars$priceClassification)
  test_result <- chisq.test(continge
# Run Chi-square for each categorical variable
categorical_results <- data.frame(
  Variable = categorical_vars,
  P_Value = sapply(categorical_vars, function(x) {
    tryCatch(chi_square_test(x), error = function(e) NA)
  })
)

# Remove transmissionType from cars dataset
cars <- cars %>% 
  select(-transmissionType)

# 3. Combine and format results
all_results <- rbind(numerical_results, categorical_results)
all_results$Significant <- ifelse(all_results$P_Value < 0.05, "Yes", "No")
all_results$Test_Type <- c(rep("ANOVA", length(numerical_vars)),
                           rep("Chi-square", length(categorical_vars)))
all_results$P_Value <- round(all_results$P_Value, 4)

# Sort by p-value
all_results <- all_results[order(all_results$P_Value), ]

# Print results with interpretation
print("Statistical Significance Analysis:")
print(all_results)

# Create visualization of p-values
ggplot(all_results, aes(x = reorder(Variable, -P_Value), y = P_Value)) +
  geom_bar(stat = "identity", aes(fill = Significant)) +
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Statistical Significance of Variables",
       subtitle = "Red line indicates p=0.05 significance threshold",
       x = "Variables",
       y = "P-Value")

# Calculate effect sizes for numerical variables
print("\nEffect Sizes for Numerical Variables:")
for(var in numerical_vars) {
  # Calculate eta squared
  model <- aov(as.formula(paste(var, "~ priceClassification")), 
               data = relevant_cars)
  ss_total <- sum((relevant_cars[[var]] - mean(relevant_cars[[var]], na.rm = TRUE))^2, 
                  na.rm = TRUE)
  ss_between <- sum(summary(model)[[1]]$"Sum Sq"[1])
  eta_squared <- ss_between/ss_total
  
  print(paste(var, "- Eta squared:", round(eta_squared, 4)))
}

# Print interpretation
cat("\nInterpretation:\n")
cat("1. Variables with p-value < 0.05 are statistically significant\n")
cat("2. Effect size interpretation:\n")
cat("   - Small effect: η² ≈ 0.01\n")
cat("   - Medium effect: η² ≈ 0.06\n")
cat("   - Large effect: η² ≈ 0.14\n")

