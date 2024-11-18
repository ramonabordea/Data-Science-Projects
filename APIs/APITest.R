# Load required libraries
library(jsonlite)
library(httr)

# Set working directory (adjust path as needed)
setwd("~/MBA/RStudioProjects/Vienna_July24/Fall_2024/day2/scripts")

# Base URL for JSONPlaceholder API
base_url <- "https://jsonplaceholder.typicode.com"

# Function to get posts
get_posts <- function() {
  response <- GET(paste0(base_url, "/posts"))
  return(fromJSON(rawToChar(response$content)))
}

# Function to get comments for a specific post
get_comments <- function(post_id) {
  response <- GET(paste0(base_url, "/posts/", post_id, "/comments"))
  return(fromJSON(rawToChar(response$content)))
}

# Function to get users
get_users <- function() {
  response <- GET(paste0(base_url, "/users"))
  return(fromJSON(rawToChar(response$content)))
}

# Test the functions and store results
posts <- get_posts()
comments <- get_comments(1)  # Get comments for first post
users <- get_users()

# Print results
print("First 5 posts:")
print(head(posts, 5))

print("\nComments for first post:")
print(head(comments, 3))

print("\nFirst 3 users:")
print(head(users, 3))
