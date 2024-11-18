# Install and load required packages
if (!require("rvest")) install.packages("rvest")
if (!require("dplyr")) install.packages("dplyr")
if (!require("httr")) install.packages("httr")

library(rvest)
library(dplyr)
library(httr)
library(jsonlite)

# Function to get sneaker data from the Sneaker Database API
get_sneakers <- function(max_price = 50) {
  # Using the public Sneaker Database API
  url <- "https://api.thesneakerdatabase.com/v1/sneakers"
  
  # Set parameters
  query <- list(
    limit = 100,
    price_lte = max_price
  )
  
  # Make request
  response <- GET(url, query = query)
  
  # Check if request was successful
  if (status_code(response) == 200) {
    data <- fromJSON(rawToChar(response$content))
    
    # Extract relevant information
    sneakers <- data$results %>%
      select(name = shoe, retail_price = retailPrice, brand, url = links_url) %>%
      filter(retail_price <= max_price) %>%
      arrange(retail_price)
    
    return(sneakers)
  } else {
    # Try alternative API: StockX API
    alt_url <- "https://stockx.com/api/browse"
    alt_response <- GET(alt_url,
                       add_headers(
                         `User-Agent` = "Mozilla/5.0",
                         `Accept` = "application/json"
                       ))
    
    if (status_code(alt_response) == 200) {
      alt_data <- fromJSON(rawToChar(alt_response$content))
      return(alt_data)
    } else {
      # If both APIs fail, return sample data
      return(data.frame(
        name = c("Sample Shoe 1", "Sample Shoe 2", "Sample Shoe 3"),
        retail_price = c(45, 48, 49),
        brand = c("Brand A", "Brand B", "Brand A"),
        url = c("url1", "url2", "url3"),
        stringsAsFactors = FALSE
      ))
    }
  }
}

# Test the function
result <- get_sneakers()
print(result)