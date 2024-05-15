# Basic Libraries
library(readr)
library(dplyr)
library(caret)
library(plotly)
library(ggplot2)
library(scales)
library(corrplot)
library(psych)
library(tidyverse)
library(heatmaply)  
library(xgboost)
library(ROCR)
library(pROC)
library(randomForest)
################################################################################

#Import data - overview
# Data Pre-processing Libraries
wq <- water_potability
head(wq)
str(wq)
#Summary information about the number of observations and variables
dim(wq)

################################################################################
#Data Processing - Overview of data files
#Firstly, we need to know how many drinkable/ not drinkable observations - the ratio
wq_pie_data <- wq %>%
  group_by(Potability) %>%
  count() %>%
  ungroup() 
#From result, we have 1278 drinkable observations and 1998 not drinkable observations

# Calculate covered percent
wq_pie_data <- wq_pie_data %>%
  mutate(percent = round(n / sum(n) * 100))

# Draw pie chart
pie_chart <- ggplot(wq_pie_data, aes(x = "", y = n, fill = factor(Potability))) +
  geom_bar(width = 1, stat = "identity") +
  coord_polar("y") +
  theme_void() +
  labs(title = "Pie Chart for Potability") +
  geom_text(aes(label = percent_format()(percent/100),
                y = n + 0.5), 
            position = position_stack(vjust = 0.5))

# Show piechart
print(pie_chart)

#From result, we can see that not drinkable observations are much more than drinkable observations
#We conclude that means may be have a slightly skewness to the right side of model observations
#Let try to draw boxplot and distribution of a variable - hardness, also we will draw the access limit for dinking from Google search 

hist_plot <- ggplot(wq, aes(x = Hardness)) +
  geom_histogram(binwidth = 10, fill = "blue", color = "black", alpha = 0.7) +
  geom_vline(xintercept = c(300), linetype = "dashed", color = "red") +
  labs(title = "Distribution of Hardness",
       x = "Hardness (mg/L)",
       y = "Count") +
  theme_minimal()

# Create boxplot
box_plot <- plot_ly(wq, y = ~Hardness, color = ~as.factor(Potability), type = "box") %>%
  layout(
    font = list(family = "monospace"),
    title = list(text = 'Boxplot for Hardness', x = 0.5, y = 0.95, font = list(color = "darkblue", size = 20)),
    yaxis = list(title = "Hardness (mg/L)"),
    xaxis = list(title = "Potability"),
    legend = list(x = 1, y = 0.96, bordercolor = "darkgray", borderwidth = 0, tracegroupgap = 5)
  )

# Combine plots
subplot(hist_plot, box_plot, nrows = 2)

#Okay, so almost observations may be in safe standard with hardness. Let try others

hist_plot <- ggplot(wq, aes(x = ph)) +
  geom_histogram(binwidth = 0.5, fill = "blue", color = "black", alpha = 0.7) +
  geom_vline(xintercept = c(5,8.5), linetype = "dashed", color = "red") +
  labs(title = "Distribution of Hardness",
       x = "ph",
       y = "Count") +
  theme_minimal()

# Create boxplot
box_plot <- plot_ly(wq, y = ~ph, color = ~as.factor(Potability), type = "box") %>%
  layout(
    font = list(family = "monospace"),
    title = list(text = 'Boxplot for ph', x = 0.5, y = 0.95, font = list(color = "darkblue", size = 20)),
    yaxis = list(title = "ph"),
    xaxis = list(title = "Potability"),
    legend = list(x = 1, y = 0.96, bordercolor = "darkgray", borderwidth = 0, tracegroupgap = 5)
  )

# Combine plots
subplot(hist_plot, box_plot, nrows = 2)

#Okay, so now, you just do it again and again to hold all variables - JUST DRAW PLOT FOR 2 VARIABLES
#YOU NEED TO DRAW MORE
################################################################################

#Relationship between independent/ dependent variables.
#Now, you want to know the relationship between independent variables and dependent variable
#To do this, we will show the relationship on 3 sides

#Side 1: The relationship is shown through the value of mean/ max/ min/ median/ deviation
wq$Potability <- factor(wq$Potability)

#mean
wq %>% group_by(Potability) %>%
  summarize_all(~mean(., na.rm = TRUE)) %>%
  ungroup() %>%
  pivot_longer(!Potability, names_to = "features", values_to = "mean") %>%
  ggplot(aes(features, mean, fill = Potability)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~features, scales = "free")

#from result, we see that the difference is not clear so we need to use other value
#The result by mean show that almost values for Potability of drinkable/ not drinkable approximately equal
#Therefore, we dont see ay differences between them.
#We need to use other approach method - min/ max (usually min)
#min
wq %>% group_by(Potability) %>%
  summarize_all(~min(., na.rm = TRUE)) %>%
  ungroup() %>%
  pivot_longer(!Potability, names_to = "features", values_to = "min") %>%
  ggplot(aes(features, min, fill = Potability)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~features, scales = "free")

#The differences are very clearly

#Side 2: The scatter matrix plot
##Scatter matrix plot and corr heat map to see the relationship among variables detaily
# Select columns that you want to include in the Scatter Plot Matrix
selected_columns <- wq[, c("ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity")]

# Create Scatter Plot Matrix with 'psych' package
scatter_plot_matrix <- pairs.panels(selected_columns, 
                                    scale = FALSE,  # Disable variable scaling
                                    gap = 0,  # Set the gap between panels to zero
                                    ellipses = TRUE,  # Show ellipses for bivariate normal distributions
                                    col = "blue",  # Set the color of points
                                    bg = "lightblue",  # Set the background color of points
                                    pch = 16,  # Set the type of points
                                    main = "Custom Scatter Plot Matrix",  # Set the main title
                                    cex.main = 1.2,  # Set the size of the main title
                                    cex.sub = 0.8,  # Set the size of sub-titles
                                    font.sub = 2,  # Set the font style of sub-titles
                                    labels = c("pH", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic Carbon", "Trihalomethanes", "Turbidity"),  # Set custom labels for variables
                                    cex.labels = 0.8,  # Set the size of variable labels
)

# Show the Scatter Plot Matrix
print(scatter_plot_matrix)

#Side 3: Use the corr heat map
# Calculate correlation matrix
wq_corr <- wq[, sapply(wq, is.numeric)]
correlation_matrix <- cor(wq_corr, use = "pairwise.complete.obs")
correlation_matrix

# Create Correlation Heatmap using heatmaply
heatmap <- heatmaply(correlation_matrix,
                     col = viridis(100),
                     notecol = "white",
                     margins = c(50,50),
                     xlab = "Variables",
                     ylab = "Variables",
                     main = "Correlation Heatmap",
                     fontsize_row = 8,  # Set font size for row labels
                     fontsize_col = 8   # Set font size for column labels
)

# Show the Correlation Heatmap
print(heatmap)

################################################################################
#Handling missing value
#Now, we need to know how many missing data values

colSums(is.na(wq))
wq %>% group_by(Potability) %>%
  summarise_all(~sum(is.na(.))) %>% 
  mutate(Potability, sumNA = rowSums(.[-1]))
#As we can see, missing values are distributed in 3 variables - ph, chloramines and trihalomethannes
#May be need something likes barchart to see more detaily

missing_data_counts <- wq %>%
  summarise(across(everything(), ~sum(is.na(.))))

# Convert data to "tall" value for drawing chart
missing_data_counts_tall <- tidyr::gather(missing_data_counts, key = "Variable", value = "MissingCount")

# Draw bar chart
bar_chart <- ggplot(missing_data_counts_tall, aes(x = reorder(Variable, -MissingCount), y = MissingCount, fill = Variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Missing Data Counts by Variable", x = "Variable", y = "Missing Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Set3")  # Chọn màu sắc từ bảng màu Set3

# show bar chart
print(bar_chart)

#We also try to summarize the table for variables about mean, min, max, median and deviation
summary_stats_wq <- summary(wq)
summary_stats_wq
#Replace mean value
check <- sapply(wq, is.numeric)
check
# Assuming the last column is Potability (dependent variable)
dependent_var <- ncol(wq)
# Replace NA values with column means
wq_no_na <- wq %>%
  mutate(across(-dependent_var, ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Calculate summary statistics
summary_stats <- summary(wq_no_na)
summary_stats

################################################################################
#Now, before building models, we need to treat the ratio of outliers
#Check outliers for variables
# Name of variables
columns_to_check <- c("ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity")

# Check outliers
check_outliers <- function(column_name) {
  outliers <- boxplot.stats(wq_no_na[[column_name]])$out
  index_out <- which(wq_no_na[[column_name]] %in% c(outliers))
  cat("Outliers for", column_name, ":", length(index_out), "out of", length(wq_no_na[[column_name]]), "|| the ratio is:", length(index_out)/length(wq_no_na[[column_name]]), "\n")
}

# Apply for selected column
sapply(columns_to_check, check_outliers)
#As we can see, there has a border to split the ratio of outliers into 2 parts: greater than 2% and smaller than 2%
#So we wil concentrate on 3 variables: ph ~ 4,3%; Hardness ~ 2,5% and Sulfate ~ 8,05%

#You need to do with all variable - except the Potability
#Now, you know it is really unreal when some observations has high degree of ph but still to be treated as drinkable water
#So obviously, we should setup the limit for values of variables 
#set up the floor and cell value for ph
# Use quantile method
# Use quantile method for ph, Hardness và Sulfate
apply_quantile_method <- function(column_name) {
  q25 <- quantile(wq_no_na[[column_name]], 0.25)
  q75 <- quantile(wq_no_na[[column_name]], 0.75)
  
  # Set up flooring and capping
  floor_threshold <- q25 - 1.5 * IQR(wq_no_na[[column_name]])
  cap_threshold <- q75 + 1.5 * IQR(wq_no_na[[column_name]])
  
  # Implement flooring and capping
  wq_no_na[[column_name]][wq_no_na[[column_name]] < floor_threshold] <- floor_threshold
  wq_no_na[[column_name]][wq_no_na[[column_name]] > cap_threshold] <- cap_threshold
  
  # Return processed data
  return(wq_no_na[[column_name]])
}

# List of variables are used quantile method
quantile_variables <- c("ph", "Hardness", "Sulfate")

# Use for variables in quantile_variables
for (variable in quantile_variables) {
  apply_quantile_method(variable)
}

#Do again for sure
wq_no_na[quantile_variables] <- lapply(quantile_variables, apply_quantile_method)

#Check outliers again
outliers <- boxplot.stats(wq_no_na$ph)$out 
index_out <- which(wq_no_na$ph %in% c(outliers)) 
length(index_out)/length(wq_no_na$ph)

#Use drop method for others - this method is used for others has ratio of outliers <2%
#Warning: Shouldn't use-  because they can cause over-fitting problem
# List to process
#variables_to_process <- c("Solids", "Chloramines", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity")

# Loop
#for (variable in variables_to_process) {
#  outliers <- boxplot.stats(wq_no_na[[variable]])$out
#  wq_no_na[[variable]][wq_no_na[[variable]] %in% outliers] <- NA
#}

# Remove NA
# wq_no_na <- na.omit(wq_no_na)
# Because the ratio is too small so we should hold them, don't do anything

########################Check outliers again by box plot
# List variables to draw box plot
variables_to_plot <- c("ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity")

# Tạo box plot cho mỗi biến
box_plots <- lapply(variables_to_plot, function(variable) {
  plot_ly(wq_no_na, y = as.formula(paste0("~", variable)), color = ~as.factor(Potability), type = "box") %>%
    layout(
      font = list(family = "monospace"),
      title = list(text = paste("Boxplot for", variable), x = 0.5, y = 0.95, font = list(color = "darkblue", size = 20)),
      yaxis = list(title = variable),
      xaxis = list(title = "Potability"),
      legend = list(x = 1, y = 0.96, bordercolor = "darkgray", borderwidth = 0, tracegroupgap = 5)
    )
})

# Hiển thị các boxplot
box_plots


#PCA method - because the number of independent variables of our data is so large - 9 variables
#So we will use PCA - method helps us to reduce the dimension and the complexity of our data

wq_sample <-wq_no_na

wq_no_na <- as.data.frame(wq_no_na)

# Applying PCA and data viz 
x1 <- preProcess(wq_no_na %>% select(-Potability), method = "pca", pcaComp = 2)
x2 <- predict(x1, wq_no_na %>% select(-Potability))
colors <- c("#51C4D3", "#74C365")  # Set your desired colors here

x2 %>% mutate(
  Potability = as.factor(wq_no_na$Potability)  # Ensure Potability is a factor
) %>%
  ggplot(aes(PC1, PC2, color = Potability)) +
  geom_point() +
  scale_color_manual(values = colors)

################################################################################
#Build Model
#Part 1: Logistic Regression
wq_filter <- wq_no_na

set.seed(2021)
trainIndex <- createDataPartition(wq_filter$Potability, p = .7, 
                                  list = FALSE, 
                                  times = 1)

wq_filterTrain <- wq_filter[ trainIndex,]
wq_filterTest  <- wq_filter[-trainIndex,]

train_control <- trainControl(method="cv", number=10)

# train the model
logit_model <- train(Potability ~ ., data= wq_filterTrain, trControl=train_control, family= "binomial", method="glm")

summary(logit_model)
pred <- predict(logit_model, wq_filterTest, type = "raw")
#Conclusion 1: Very bad model - as we can see, many standard shows NA value

confusionMatrix(data = wq_filterTest$Potability, pred)

#Part 2: SVM model
svm_radial <- train(Potability ~ ., data= wq_filterTrain, trControl=train_control,  method = "svmRadial", preProcess = c("center","scale"), tuneLength = 10)
summary(svm_radial)
#"svmRadial" "svmLinear"
pred_svm_radial <- predict(svm_radial, wq_filterTest)

confusionMatrix(data = wq_filterTest$Potability, pred_svm_radial)
#Conclusion 2: Quite good model - accuracy is approximately 67,92% - moderate-high accuracy
#The kapp is 25,23% - not bad 
#But may be will happens over-fitting problems
#Part 3: Random Forest
# Convert Potability to a binary factor in the original data
wq_no_na$Potability <- as.factor(wq_no_na$Potability)

# Split the data into training and testing sets for Random Forest
set.seed(2021)
trainIndex_rf <- createDataPartition(wq_no_na$Potability, p = 0.7, list = FALSE)
wq_train_rf <- wq_no_na[trainIndex_rf, ]
wq_test_rf <- wq_no_na[-trainIndex_rf, ]

# Build the Random Forest model
rf_model <- randomForest(Potability ~ ., data = wq_train_rf, ntree = 450, mtry = 3)
summary(rf_model)
# Make predictions on the test set
rf_pred <- predict(rf_model, newdata = wq_test_rf)

# Make predictions on the test set with probabilities
rf_pred_probs <- as.numeric(predict(rf_model, newdata = wq_test_rf, type = "response"))

# Create a ROC curve object
roc_curve <- roc(wq_test_rf$Potability, rf_pred_probs)

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for Random Forest Model", col = "blue", lwd = 2)

# Add AUC value to the plot
auc_value <- auc(roc_curve)
legend("bottomright", legend = paste("AUC =", round(auc_value, 10)), col = "blue", lwd = 2)

# Print AUC value
print(paste("AUC for Random Forest Model:", auc_value))

# Evaluate the Random Forest model
conf_matrix_rf <- confusionMatrix(rf_pred, wq_test_rf$Potability)
print("Confusion Matrix for Random Forest Model:")
print(conf_matrix_rf)

################################################################################
###The last model: XGBoost Model
# Define the features and the target variable for tuning
wq_xgb <- wq_no_na
set.seed(2021)
trainIndex_xgb <- createDataPartition(wq_xgb$Potability, p = 0.7, list = FALSE)
wq_train_xgb <- wq_xgb[trainIndex_xgb, ]
wq_test_xgb <- wq_xgb[-trainIndex_xgb, ]

# Convert Potability to a binary factor
wq_train_xgb$Potability <- as.factor(wq_train_xgb$Potability)
wq_test_xgb$Potability <- as.factor(wq_test_xgb$Potability)
features_tune <- setdiff(names(wq_train_xgb), "Potability")

# Create a matrix for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(wq_train_xgb[, features_tune]),
                      label = as.numeric(wq_train_xgb$Potability) - 1)

# Define a grid of hyperparameters to search
hyperparameter_grid <- expand.grid(
  nrounds = c(50, 100, 150),          # Number of boosting rounds
  max_depth = c(5, 10, 15),           # Maximum depth of a tree
  eta = c(0.01, 0.05, 0.1),           # Learning rate
  subsample = c(0.7, 0.8, 0.9),       # Subsample ratio of the training data
  colsample_bytree = c(0.7, 0.8, 0.9) # Subsample ratio of columns when constructing each tree
)

# Perform grid search with cross-validation
xgb_tune <- xgb.cv(
  params = list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    booster = "gbtree"
  ),
  data = dtrain,
  nrounds = 200,               # Maximum number of boosting rounds
  nfold = 5,                   # Number of folds for cross-validation
  verbose = 0,                 # Disable printing progress
  early_stopping_rounds = 10,  # Stop if performance hasn't improved in 10 rounds
  maximize = FALSE,            # Minimize logloss
  metrics = "logloss",         # Evaluation metric
  print_every_n = 10,          # Print every 10 rounds
  verbose_eval = TRUE,         # Print logloss metric
  search_spaces = hyperparameter_grid
)

# Get the best hyperparameters
best_params <- xgb_tune$best_parameters
print("Best Hyperparameters:")
print(best_params)

# Train the final XGBoost model with the best hyperparameters
xgb_model_final <- xgboost(
  data = dtrain,
  params = c(best_params, list(objective = "binary:logistic", eval_metric = "logloss")),
  nrounds = xgb_tune$best_iteration
)

summary(xgb_model_final)

# Make predictions on the test set
xgb_pred_probs_final <- predict(xgb_model_final, xgb.DMatrix(data = as.matrix(wq_test_xgb[, features_tune])))
xgb_pred_final <- ifelse(xgb_pred_probs_final > 0.5, 1, 0)

# Evaluate the final model
xgb_roc_final <- prediction(xgb_pred_probs_final, wq_test_xgb$Potability)
xgb_perf_final <- performance(xgb_roc_final, "tpr", "fpr")
auc_value_final <- performance(xgb_roc_final, "auc")@y.values[[1]]
print(paste("Final AUC:", auc_value_final))

# Plot ROC curve for the final model
plot(xgb_perf_final, main = "Final ROC Curve for XGBoost Model", col = "blue", lwd = 2)

# Display the confusion matrix for the final model
conf_matrix_xgb_final <- confusionMatrix(as.factor(xgb_pred_final), as.factor(wq_test_xgb$Potability))
print("Final Confusion Matrix for XGBoost Model:")
print(conf_matrix_xgb_final)
##################################################################################