
library(vtreat)
library(ggplot2)
library(ggthemes)
library(dplyr)
library(readr)
library(corrplot)
library(randomForest)


## Obtain the data
custMktgResults <- read.csv('https://raw.githubusercontent.com/kwartler/Vienna_July24/main/PostModuleAssignment/training/CurrentCustomerMktgResults.csv', quote='"', fill=TRUE, comment.char="")
axiomTable <- read.csv('https://raw.githubusercontent.com/kwartler/Vienna_July24/main/PostModuleAssignment/training/householdAxiomData.csv')
HHCreditTable <- read.csv('https://raw.githubusercontent.com/kwartler/Vienna_July24/main/PostModuleAssignment/training/householdCreditData.csv', quote='"', fill=TRUE, comment.char="")
HHVehicleTable <- read.csv('https://raw.githubusercontent.com/kwartler/Vienna_July24/main/PostModuleAssignment/training/householdVehicleData.csv')
ProspectiveCust <- read.csv('https://raw.githubusercontent.com/kwartler/Vienna_July24/main/PostModuleAssignment/ProspectiveCustomers.csv', quote='"', fill=TRUE, comment.char="")


## Explore - visuals, summary stats, correlation, get to know the data and challenge yourself to think about the data (there is a small trick betwen the prospect and training data variables, see if you can find it)
# Display the first few rows of each dataset
head(custMktgResults)
head(axiomTable)
head(HHCreditTable)
head(HHVehicleTable)
head(ProspectiveCust)

# Get a summary of each dataset
summary(custMktgResults)
summary(axiomTable)
summary(HHCreditTable)
summary(HHVehicleTable)
summary(ProspectiveCust)

# Check for missing values
sapply(custMktgResults, function(x) sum(is.na(x)))
sapply(axiomTable, function(x) sum(is.na(x)))
sapply(HHCreditTable, function(x) sum(is.na(x)))
sapply(HHVehicleTable, function(x) sum(is.na(x)))
sapply(ProspectiveCust, function(x) sum(is.na(x)))


# Histogram of Age (from axiomTable)
ggplot(axiomTable, aes(x=Age)) + 
  geom_histogram(binwidth=5, fill="blue", color="black") + 
  theme_minimal() + 
  labs(title="Age Distribution", x="Age", y="Count")
ggsave('~/MBA/RStudioProjects/Vienna_July24/PostModuleAssignment/HistogramAge.png')

# Bar plot for Communication types (from custMktgResults)
ggplot(custMktgResults, aes(x=Communication)) + 
  geom_bar(fill="skyblue", color="black") + 
  theme_minimal() + 
  labs(title="Distribution of Communication Types", x="Communication Type", y="Count")
ggsave('~/MBA/RStudioProjects/Vienna_July24/PostModuleAssignment/Communication.png')

# Pie chart for Y_AcceptedOffer (from custMktgResults)
custMktgResults %>%
  count(Y_AcceptedOffer) %>%
  ggplot(aes(x="", y=n, fill=Y_AcceptedOffer)) +
  geom_bar(stat="identity", width=1) +
  coord_polar("y") +
  theme_minimal() +
  labs(title="Proportion of Accepted Offers")
ggsave('~/MBA/RStudioProjects/Vienna_July24/PostModuleAssignment/OfferAccepted.png')

# Scatter plot for Age vs. RecentBalance (from HHCreditTable)
ggplot(HHCreditTable %>% inner_join(axiomTable, by="HHuniqueID"), aes(x=Age, y=RecentBalance)) + 
  geom_point(color="red") + 
  theme_minimal() + 
  labs(title="Age vs. Recent Balance", x="Age", y="Recent Balance")
ggsave('~/MBA/RStudioProjects/Vienna_July24/PostModuleAssignment/ageVSbalance.png')

# Box plot for DaysPassed by Y_AcceptedOffer (from custMktgResults)
ggplot(custMktgResults, aes(x=Y_AcceptedOffer, y=DaysPassed)) + 
  geom_boxplot(fill="orange", color="black") + 
  theme_minimal() + 
  labs(title="Days Passed by Offer Acceptance", x="Offer Accepted", y="Days Passed")
ggsave('~/MBA/RStudioProjects/Vienna_July24/PostModuleAssignment/Days2Accept.png')


# Calculate correlation matrix for numeric columns in custMktgResults
corr_matrix <- cor(select_if(custMktgResults, is.numeric), use="complete.obs")
corrplot(corr_matrix, method="circle", type="upper")

# Calculate correlation matrix for numeric columns in axiomTable
corr_matrix <- cor(select_if(axiomTable, is.numeric), use="complete.obs")
corrplot(corr_matrix, method="circle", type="upper")

# Calculate correlation matrix for numeric columns in HHCreditTable
corr_matrix <- cor(select_if(HHCreditTable, is.numeric), use="complete.obs")
corrplot(corr_matrix, method="circle", type="upper")

# Calculate correlation matrix for numeric columns in ProspectiveCust
corr_matrix <- cor(select_if(ProspectiveCust, is.numeric), use="complete.obs")
corrplot(corr_matrix, method="circle", type="upper")

# Merge tables based on HHuniqueID
merged_data <- custMktgResults %>%
  inner_join(axiomTable, by="HHuniqueID") %>%
  inner_join(HHCreditTable, by="HHuniqueID") %>%
  inner_join(HHVehicleTable, by="HHuniqueID")

# Perform analysis on merged data
summary(merged_data)
head(merged_data)
head(ProspectiveCust)

#----------# Subset merged_data to keep only the columns as ProspectiveCust-----------------------------------------------------------

# Get column names of both dataframes
common_cols <- intersect(names(merged_data), names(ProspectiveCust))

# Print common columns
print(common_cols)
# Subset merged_data to keep only the columns that are in common_cols
merged_data_cleaned <- merged_data[, common_cols]


# Check for missing values in the merged dataset
sapply(merged_data_cleaned, function(x) sum(is.na(x)))


# Bar plot for Y_AcceptedOffer
ggplot(merged_data_cleaned, aes(x=Y_AcceptedOffer)) + 
  geom_bar(fill="purple", color="black") + 
  theme_minimal() + 
  labs(title="Distribution of Accepted Offers", x="Offer Accepted", y="Count")

# Select numeric columns for correlation analysis
numeric_cols <- select_if(merged_data_cleaned, is.numeric)
# Compute correlation matrix
corr_matrix <- cor(numeric_cols, use="complete.obs")
# Plot correlation matrix
corrplot(corr_matrix, method="circle", type="upper", tl.cex=0.8)


#----------# Scrub - vtreat package & possibly engineer feature----------------------------------

varlist <- setdiff(colnames(merged_data_cleaned), "Y_AcceptedOffer")

# Step 2: Create a treatment plan; 'Y_AcceptedOffer' is the target variable
treatment_plan <- designTreatmentsC(
  dframe = merged_data_cleaned,           
  varlist = varlist,              
  outcomename = "Y_AcceptedOffer",
  outcometarget = "Accepted"    
)

# Step 3: Apply the treatment plan to the data
treated_data <- prepare(treatment_plan, merged_data_cleaned)
head(treated_data)


# Bar plot to show the distribution of accepted offers, same as before so OK
ggplot(treated_data, aes(x = Y_AcceptedOffer)) +
  geom_bar(fill = "blue") +
  labs(title = "Distribution of Accepted Offers",
       x = "Accepted Offer",
       y = "Count") +
  theme_minimal()


# Convert the target variable to a factor
treated_data$Y_AcceptedOffer <- as.factor(treated_data$Y_AcceptedOffer)

# Ensure there are no missing values
treated_data <- na.omit(treated_data)
set.seed(123)  # For reproducibility
train_index <- createDataPartition(treated_data$Y_AcceptedOffer, p = 0.7, list = FALSE)
train_data <- treated_data[train_index, ]
test_data <- treated_data[-train_index, ]



#---------------------------------------Build randomForest model, output probability ---------------------------

rf_model <- randomForest(Y_AcceptedOffer ~ ., data = train_data, ntree = 500, mtry = 3, importance = TRUE)

predicted_probabilities <- predict(rf_model, newdata = test_data, type = "prob")

# Convert the probabilities for the 'Accepted' class into percentages
predicted_percentages <- predicted_probabilities[, "Accepted"] * 100

# Add the predicted percentages to the test data for inspection
test_data$Predicted_Probability <- predicted_percentages

# View the first few rows to see the predicted probabilities
head(test_data$Predicted_Probability)

# Define a threshold for converting probabilities to class labels
threshold <- 50  # If probability is greater than 50%, predict "Accepted"

# Convert probabilities to class labels based on the threshold
predicted_classes <- ifelse(predicted_percentages > threshold, "Accepted", "DidNotAccept")

# Confusion matrix based on the thresholded predictions
confusion_matrix <- confusionMatrix(as.factor(predicted_classes), test_data$Y_AcceptedOffer)
print(confusion_matrix)

saveRDS(rf_model, file = "~/MBA/RStudioProjects/Vienna_July24/PostModuleAssignment/random_forest_model.rds")


#------------------------------------- Treat ProspectiveCust with same treatment plan as merged data

# Apply the treatment plan to the ProspectiveCust data
treated_prospective_data <- prepare(treatment_plan, ProspectiveCust)

# Predict probabilities for the ProspectiveCust dataset
prospective_probabilities <- predict(rf_model, newdata = treated_prospective_data, type = "prob")

# Convert the probabilities for the 'Accepted' class into percentages
prospective_percentages <- prospective_probabilities[, "Accepted"] * 100

# Add the predicted percentages to the ProspectiveCust data for inspection
treated_prospective_data$Predicted_Probability <- prospective_percentages


# Create a histogram of predicted probabilities
ggplot(treated_prospective_data, aes(x = Predicted_Probability)) + 
  geom_histogram(binwidth = 0.05, fill = "blue", color = "black", alpha = 0.7) +
  theme_minimal() + 
  labs(title = "Distribution of Predicted Probabilities",
       x = "Predicted Probability of Accepting Offer",
       y = "Count of Prospective Customers")


#create sorted_prospective_customers containing the prospective customers ids based on predicted probability, descending
sorted_prospective_customers <- treated_prospective_data %>%
  select(dataID, Predicted_Probability) %>%
  arrange(desc(Predicted_Probability))

# Subset the first 100 customers
top_100_customers <- sorted_prospective_customers[1:100, ]

#remove Y_AcceptedOffer from Prospective Custosmers
ProspectiveCust <- ProspectiveCust %>%
  select(-Y_AcceptedOffer)

# Create a dot plot of Predicted_Probability for all customers in sorted_prospective_customers
ggplot(sorted_prospective_customers, aes(x = seq_along(Predicted_Probability), y = Predicted_Probability)) +
  geom_point(color = "green", size = 2) +
  theme_minimal() +
  labs(title = "Dot Plot of Predicted Probabilities",
       x = "Customer Index",
       y = "Predicted Probability") +
  theme(text = element_text(size = 12))


# Create a dot plot of Predicted_Probability for all customers in top_100_customers
ggplot(top_100_customers, aes(x = seq_along(Predicted_Probability), y = Predicted_Probability)) +
  geom_point(color = "green", size = 2) +
  theme_minimal() +
  labs(title = "Dot Plot of Predicted Probabilities",
       x = "Customer Index",
       y = "Predicted Probability") +
  theme(text = element_text(size = 12))



#------------------------------Extract customer profile for the top prospective customers


# Join top_100_customers with ProspectiveCust by dataID
SelectedProspectiveCust <- top_100_customers %>%
  inner_join(ProspectiveCust, by = "dataID")

# View the first few rows of the new table
head(SelectedProspectiveCust)

# Save the new table to a CSV file (optional)
write.csv(SelectedProspectiveCust, "~/MBA/RStudioProjects/Vienna_July24/PostModuleAssignment/SelectedProspectiveCust.csv", row.names = FALSE)

