# Using data Telco-Customer-Churn

# Importing datasets

our.data = read.csv("Telco-Customer-Churn.csv")
str(our.data)

# setup Factors
# SeniorCitizen

table(our.data$SeniorCitizen)
our.data$SeniorCitizen = as.factor(our.data$SeniorCitizen)

# Check the missing data

count_na = function(x){
  na_varname = c()
  na_count = c()
  na_perc = c()
  
  for (i in 1:ncol(x)) {
    if (sum(is.na(x[i])) > 0) {
      var_name = colnames(x[i])
      count = sum(is.na(x[i]))
      perc = round(count/nrow(x) * 100, digits = 2)
      
      na_varname = c(na_varname, var_name)
      na_count = c(na_count, count)
      na_perc = c(na_perc, perc)
    }
    
    
  }
  missing_table = data.frame(na_varname, na_count, na_perc)
  return (missing_table)
}

na_table = count_na(x = our.data)
na_table

# TotalCharges have 11 missing values

# First, we don't need customer Id so dropping that column
our.data = our.data[,-1]

# Change some factor variable
# MultipleLines
our.factor.MultipleLines = c("NoPhoneService", "No", "Yes")

our.data$MultipleLines = factor(our.data$MultipleLines, levels = c("No phone service", "No", "Yes"), 
               labels = our.factor.MultipleLines)

# InternetService
our.factor.InternetService = c( "DSL", "FiberOptic","No")
our.data$InternetService = factor(our.data$InternetService, levels = c("DSL", "Fiber optic", "No"), 
                                labels = our.factor.InternetService)

# Contract
our.factor.Contract = c( "MonthToMonth", "OneYear", "TwoYear")
our.data$Contract = factor(our.data$Contract, levels = c("Month-to-month", "OneYear", "TwoYear"), 
                                  labels = our.factor.Contract)

# PaymentMethod
our.factor.PaymentMethod = c("BankTransferA","CreditCardA", 
                             "ElectronicCheck", "MailedCheck" )

our.data$PaymentMethod = factor(our.data$PaymentMethod, 
                                levels = c("Bank transfer (automatic)","Credit card (automatic)", 
                                           "Electronic check", "Mailed check" ), 
                           labels = our.factor.PaymentMethod)

# OnlineSecurity
our.factor.OnlineSecurity = c("No","NoInternetService", "Yes" )

our.data$OnlineSecurity = factor(our.data$OnlineSecurity, 
                                levels = c("No","No internet service", "Yes" ), 
                                labels = our.factor.OnlineSecurity)

# OnlineBackup
our.factor.OnlineBackup = c("No","NoInternetService", "Yes" )

our.data$OnlineBackup = factor(our.data$OnlineBackup, 
                                 levels = c("No","No internet service", "Yes" ), 
                                 labels = our.factor.OnlineBackup)

# DeviceProtection
our.factor.DeviceProtection = c("No","NoInternetService", "Yes" )

our.data$DeviceProtection = factor(our.data$DeviceProtection, 
                               levels = c("No","No internet service", "Yes" ), 
                               labels = our.factor.DeviceProtection)

# TechSupport
our.factor.TechSupport = c("No","NoInternetService", "Yes" )

our.data$TechSupport = factor(our.data$TechSupport, 
                                   levels = c("No","No internet service", "Yes" ), 
                                   labels = our.factor.TechSupport)

# StreamingTV
our.factor.StreamingTV = c("No","NoInternetService", "Yes" )

our.data$StreamingTV = factor(our.data$StreamingTV, 
                              levels = c("No","No internet service", "Yes" ), 
                              labels = our.factor.StreamingTV)

# StreamingMovies
our.factor.StreamingMovies = c("No","NoInternetService", "Yes" )

our.data$StreamingMovies = factor(our.data$StreamingMovies, 
                              levels = c("No","No internet service", "Yes" ), 
                              labels = our.factor.StreamingMovies)


# Imputing data using CARET

library(caret)

dummy.vars = dummyVars(~., 
                       data = our.data)
our.data.dummy = predict(dummy.vars, our.data)

# Now, Impute!
pre.process = preProcess(our.data.dummy, method = "bagImpute")

imputed.data = predict(pre.process, our.data.dummy)

our.data$TotalCharges = imputed.data[, 46]


# Sampling the data into training and test set
indexs = createDataPartition(our.data$Churn, times = 1, 
                             p = 0.7, list = FALSE)

trd = our.data[indexs, ]
tsd = our.data[-indexs, ]

# examine the proportion on churn

prop.table(table(our.data$Churn))
prop.table(table(trd$Churn))
prop.table(table(tsd$Churn))


#=============================================================================
# TRAIN MODEL
#=============================================================================

# SetUp Caret to do 5 fold cross validation repeated 2 times and uses  
# grid search for optimal model

train.control = trainControl(method = 'repeatedcv', 
                             number = 5, 
                             repeats = 2, 
                             search = "grid")

# Using doSNOW
library(doSNOW)

c1 = makeCluster(2, type = "SOCK")
registerDoSNOW(c1)

caret.cv = train(Churn ~ ., 
                 data = trd, 
                 method = "xgbTree",
                 trControl = train.control)
stopCluster(c1)

caret.cv

pred = predict(caret.cv, tsd)

# Using confusion matrix
confusionMatrix(pred, tsd$Churn)
