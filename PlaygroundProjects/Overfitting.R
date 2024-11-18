library(ggplot2)

# Create sample data
set.seed(123)
x <- 1:20
y <- x + rnorm(20, 0, 2)  # True relationship is linear with some noise
df <- data.frame(x = x, y = y)

# Create three plots to show different levels of fitting
p1 <- ggplot(df, aes(x = x, y = y)) +
  geom_point(color = "black", size = 3) +
  geom_smooth(method = "lm", color = "blue", se = FALSE) +
  ggtitle("Good Fit") +
  theme_minimal()

p2 <- ggplot(df, aes(x = x, y = y)) +
  geom_point(color = "black", size = 3) +
  geom_smooth(method = "loess", span = 0.1, color = "red", se = FALSE) +
  ggtitle("Overfit") +
  theme_minimal()

p3 <- ggplot(df, aes(x = x, y = y)) +
  geom_point(color = "black", size = 3) +
  stat_smooth(method = "lm", formula = y ~ 1, color = "green", se = FALSE) +
  ggtitle("Underfit") +
  theme_minimal()

# Display plots
print(p1)
print(p2)
print(p3)