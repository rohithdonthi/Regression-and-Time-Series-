# Load necessary package
install.packages("MASS")
library(MASS)
#install.packages("boot") 
#library(boot)

x <- read.csv("C:\\Users\\crazy\\Downloads\\CleanedData.csv")

# Clean column names
names(x) <- make.names(names(x), unique = TRUE)

data <- read.csv("C:\\Users\\crazy\\Downloads\\CleanedData.csv")
min_max_normalize <- function(a) {
  return ((a - min(a)) / (max(a) - min(a)))
}
x <- as.data.frame(lapply(data, min_max_normalize))

# Identify predictor columns (all columns except 'shares')
predictor_columns <- setdiff(names(data), "shares")

# Apply normalization to all columns except 'shares'
x <- data  # Copy the original data
x[predictor_columns] <- lapply(x[predictor_columns], min_max_normalize)


# Full Model
full.model <- lm(shares ~ ., data = x)
start.model <- lm(shares ~ 1, data = x)
scope <- list(lower = start.model, upper = full.model)
# Stepwise Model Selection based on AIC
model.aic <- stepAIC(full.model, direction = "both")
# kinda useless without scope but kept for documentation purposes. model.aicf <- stepAIC(start.model, direction = "forward")
model.aicf <- stepAIC(start.model, direction = "forward", scope = scope)
model.aicb <- stepAIC(full.model, direction = "backward")
# Alternatively, for BIC use step() function with k=log(n)
# n is the number of observations
model.bic <- step(full.model, direction = "both", k = log(nrow(x)))
# Summary of the selected models
summary(model.aic)
summary(full.model)
summary(model.aicf)
summary(model.aicb)
summary(model.bic)
summary(start.model)
aic_value <- AIC(model.aic, full.model, model.aicf, model.aicb, model.bic, start.model)
aic_value
# Function to calculate MSE
calc_mse <- function(model, data) {
  predictions <- predict(model, newdata = data)
  return(mean((data$shares - predictions)^2))
}

# Calculate MSE for each model
mse_full <- calc_mse(full.model, x)
mse_aic <- calc_mse(model.aic, x)
mse_aicf <- calc_mse(model.aicf, x)
mse_aicb <- calc_mse(model.aicb, x)
mse_bic <- calc_mse(model.bic, x)

# Output the MSE values
mse_full
mse_aic
mse_aicf
mse_aicb
mse_bic

# Fit a Poisson Regression model
glm.model <- glm(shares ~ ., family = poisson(link = "log"), data = x)
# Summary of the model
summary(glm.model)
plot(glm.model$fitted.values, residuals(glm.model, type = "deviance"))
overdispersion_test <- glm.model$deviance / glm.model$df.residual
print(overdispersion_test)
# Define the upper (full) and lower (intercept-only) models for the Poisson regression
lower.model <- glm(shares ~ 1, family = poisson(link = "log"), data = x)
upper.model <- glm(shares ~ ., family = poisson(link = "log"), data = x)
# Perform stepwise selection
sglm.model <- stepAIC(lower.model, scope = list(lower = lower.model, upper = upper.model), direction = "both")
summary(lower.model)
summary(upper.model)
summary(sglm.model)
quasi.model <- glm(shares ~ ., family = quasipoisson(link = "log"), data = x)
summary(quasi.model)
negbin.model <- glm.nb(shares ~ ., data = x)
summary(negbin.model)
# Fit the initial Negative Binomial model (full model)
negbin.full <- glm.nb(shares ~ ., data = x)
summary(negbin.full)
# Define the intercept-only model (lower model)
negbin.lower <- glm.nb(shares ~ 1, data = x)
summary(negbin.lower)
stepwise.negbin <- stepAIC(negbin.full, scope = list(lower = negbin.lower, upper = negbin.full), direction = "both")
summary(stepwise.negbin)

#install.packages("car")
library(car)
vif(glm.model)

# Predicting on the data
predicted_negbin <- predict(negbin.model, type = "response")
# Calculating MSE
mse_negbin <- mean((x$shares - predicted_negbin)^2)
# Calculating McFadden's pseudo-R2
ll_full <- logLik(negbin.model)
ll_null <- logLik(glm.nb(shares ~ 1, data = x))
pseudo_r_squared_negbin <- 1 - as.numeric(ll_full / ll_null)
# Output the MSE and pseudo-R2
mse_negbin
pseudo_r_squared_negbin

# Predicting with the Poisson model
predicted_poisson <- predict(glm.model, type = "response")
# Calculating MSE for Poisson model
mse_poisson <- mean((x$shares - predicted_poisson)^2)
# Calculating McFadden's pseudo-R2 for Poisson model
ll_full_poisson <- logLik(glm.model)
ll_null_poisson <- logLik(glm(shares ~ 1, family = poisson(link = "log"), data = x))
pseudo_r_squared_poisson <- 1 - as.numeric(ll_full_poisson / ll_null_poisson)
# Output the MSE and pseudo-R2 for Poisson
mse_poisson
pseudo_r_squared_poisson

#Everything below this point was kept for the record, but wasn't useful enough to talk about in final documentation.
#Cross Validation
# Function for manually calculating MSE
mse_calc <- function(actual, predicted) {
  return(mean((actual - predicted)^2))
}
# Setting up k-fold cross-validation
set.seed(123)
n <- nrow(x)
k <- 10
folds <- cut(seq(1, n), breaks = k, labels = FALSE)
# Initialize a vector to store MSE for each fold
mse_values <- vector(length = k)
# Perform k-fold cross-validation
for(i in 1:k){
  # Splitting data into training and testing sets
  test_indices <- which(folds == i, arr.ind = TRUE)
  train_indices <- setdiff(1:n, test_indices)
  train_data <- x[train_indices, ]
  test_data <- x[test_indices, ]
  # Fit Negative Binomial model
  model_nb <- glm.nb(shares ~ ., data = train_data)
  # Predict on test data
  predictions <- predict(model_nb, test_data, type = "response")
  # Calculate and store MSE
  mse_values[i] <- mse_calc(test_data$shares, predictions)
}
# Average MSE across all folds
mean_mse_negbin <- mean(mse_values)
mean_mse_negbin
mse_values

# Define Poisson model
poisson_model_func <- function(data, indices) {
  train_data <- data[indices, ]
  fit <- glm(shares ~ ., family = poisson(link = "log"), data = train_data)
  return(fit)
}
# Perform k-fold cross-validation
set.seed(123) # for reproducibility
cv_poisson <- cv.glm(x, poisson_model_func, K = 10) # 10-fold cross-validation
# MSE for Poisson model
mse_poisson <- cv_poisson$delta[1]
mse_poisson

# Splitting the data
set.seed(123)  # for reproducibility
train_indices <- sample(1:nrow(x), 0.7 * nrow(x))  # 70% for training
train_data <- x[train_indices, ]
test_data <- x[-train_indices, ]

# Fit the model on training data
model_train <- glm(shares ~ ., family = poisson(link = "log"), data = train_data)

# Predict on test data
predictions_test <- predict(model_train, test_data, type = "response")

# Calculate MSE on test data
mse_test <- mean((test_data$shares - predictions_test)^2)
mse_test