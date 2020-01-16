getwd()
setwd("C:/Users/jonathanhudgins/Desktop/Harvard/Courses/ML/Project")
library(data.table)
air_pollution <- data.frame(fread("air_pollution.csv"))

# packages
#install.packages("ISLR")
#install.packages("plyr")
#install.packages("dplyr")
#install.packages("readxl")
#install.packages("randomForest")
#install.packages("ggplot2")
# install.packages("mice")

# Library
library(AER)
library(tidyr)
library(dplyr)
library(class)
library(FNN)
library(ISLR)
library(tree)
library(randomForest)
library(gbm)
library(glmnet)
library(ggplot2)
library(mice)

save.image(file="myWorkspace.RData")

# Data Cleaning 
## Downsizing columns to more useful information for the model 
# air_pollution_clean <- air_pollution[c("year", "month", "day", "hour", "PM2.5", "PM10", 
#                                       "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM")]

# Remove rows with missing values 
# air_pollution_clean <- na.omit(air_pollution_clean)

# time series of PM2.5
ggplot(air_pollution, aes(x=No, y=PM2.5)) +
  geom_line() + 
  xlab("")


# Change character into factor and integer into numeric

sapply(air_pollution,class)
air_pollution <- air_pollution %>% mutate_if(is.character, as.factor)
air_pollution <- air_pollution %>% mutate_if(is.integer, as.numeric)

# Scatterplot matrix
pairs(~PM2.5+TEMP+PRES+DEWP,data=air_pollution,
      main="Simple Scatterplot Matrix")

# Check how many % NA for each variable

for (i in 1:ncol(air_pollution)) {
  percNA <- round(length(which(is.na(air_pollution[,i])))/nrow(air_pollution)*100,2)
  print(paste(colnames(air_pollution[i]), percNA))
}

md.pattern(air_pollution)  # pattern or missing values in data.

# Relatively low NAs in the data, less than 5% on average, we will impute the data excluding PM2.5
# Impute missing values with pmm method in MICE

miceMod <- mice(air_pollution[, !names(air_pollution) %in% c("No","PM2.5","station")], 
                method="pmm")  # perform mice imputation, based on pmm.
miceOutput <- complete(miceMod)  # generate the completed data.
anyNA(miceOutput)

air_pollution[, !names(air_pollution) %in% c("No","PM2.5","station")] <- 
  miceOutput

air_pollution_clean <- na.omit(air_pollution) #exclude NAs in PM2.5 for now


# Change character variables to factor variables 
# air_pollution_clean <- mutate_if(air_pollution_clean, is.character, as.factor)

# Convert wind direction variables into a Complete Set of Indicator Variables
for (i in ncol(air_pollution_clean):1) {
  if (is.factor(air_pollution_clean[,i])) {
    for (j in unique(air_pollution_clean[,i])) {
      new_col             <- paste(colnames(air_pollution_clean)[i], j, sep = "_")
      air_pollution_clean[,new_col] <- as.numeric(air_pollution_clean[,i] == j) 
    }
    air_pollution_clean       <- air_pollution_clean[,-i]     
  } else if (typeof(air_pollution_clean[,i]) == "integer") {
    air_pollution_clean[,i]   <- as.numeric(as.character(air_pollution_clean[,i]))
  } 
 }

# Mean PM2.5 
# mean(air_pollution_clean$PM2.5)

# Standard Deviation of PM2.5 
# sd(air_pollution_clean$PM2.5)

# Histogram 
# hist(air_pollution$PM2.5)

# Split Training and Test 
# set.seed(2019)
# test_obs              <- round(0.2 * nrow(air_pollution_clean))
# train_obs             <- nrow(air_pollution_clean) - test_obs
# test_train_vec        <- c(rep("test", test_obs),
#                           rep("train", train_obs))
#?sample
# test_train_vec        <- sample(test_train_vec, nrow(air_pollution_clean), replace = FALSE)
# test_data             <- air_pollution_clean[which(test_train_vec == "test"),]
# train_data            <- air_pollution_clean[which(test_train_vec == "train"),]


# We'll try diferent ML methods and compare their MSE

set.seed(2019)

fold_ids      <- rep(seq(10), 
                     ceiling(nrow(air_pollution_clean) / 10))
fold_ids      <- fold_ids[1:nrow(air_pollution_clean)]

fold_ids      <- sample(fold_ids, length(fold_ids))


# First, try measuring MSE of linear regression

CV_MSEP_mtx_linear  <- matrix(0, 
                     nrow = 1, 
                     ncol = 10)

colnames(air_pollution_clean)
for (fold in 1:10) {
  
  lmTraining <- lm(PM2.5~as.factor(year)+as.factor(month)+as.factor(day)+as.factor(hour)
                 +PM10+SO2+NO2+CO+O3+TEMP+PRES+DEWP+RAIN+WSPM+wd_NNW
                 +wd_N+wd_NW+wd_NNE+wd_ENE+wd_E+wd_NE+wd_W+wd_SSW+
                   wd_WSW+wd_SE+wd_WNW+wd_SSE+wd_ESE+wd_S+wd_SW
                 ,air_pollution_clean[which(fold_ids != fold),])

  lmPred <- predict(lmTraining, newdata = data.frame(air_pollution_clean[which(fold_ids == fold),]))
  truth <- air_pollution_clean[which(fold_ids == fold), "PM2.5"]
  CV_MSEP_mtx_linear[1,fold]  <-  mean((lmPred - truth)^2, na.rm=TRUE)
  
}

rowMeans(CV_MSEP_mtx_linear) # MSE for linear model = 900.9 -> pretty bad... (ignore warning for now, result ok)


# Second, try measuring MSE of linear regression with logtransform of PM2.5

CV_MSEP_mtx_logtransform  <- matrix(0, 
                              nrow = 1, 
                              ncol = 10)

for (fold in 1:10) {
  
  lmTraining <- lm(log(PM2.5+1)~as.factor(year)+as.factor(month)+as.factor(day)+as.factor(hour)
                   +PM10+SO2+NO2+CO+O3+TEMP+PRES+DEWP+RAIN+WSPM+wd_NNW
                   +wd_N+wd_NW+wd_NNE+wd_ENE+wd_E+wd_NE+wd_W+wd_SSW+
                     wd_WSW+wd_SE+wd_WNW+wd_SSE+wd_ESE+wd_S+wd_SW,air_pollution_clean[which(fold_ids != fold),])
  
  lmPred <- predict(lmTraining, newdata = data.frame(air_pollution_clean[which(fold_ids == fold),]))
  lmPred_exp <- exp(lmPred)-1
  truth <- air_pollution_clean[which(fold_ids == fold), "PM2.5"]
  CV_MSEP_mtx_logtransform[1,fold]  <-  mean((lmPred_exp - truth)^2, na.rm=TRUE)
  
}

rowMeans(CV_MSEP_mtx_logtransform) # MSE for linear model with logtransform...even worse forget about it

# Third, try kNN

CV_MSEP_mtx_knn  <- matrix(0, 
                                    nrow = 10, 
                                    ncol = 10)

colnames(air_pollution_clean)
data_knn <- air_pollution_clean[,!names(air_pollution) %in% c("No","station")]

data_knn <- air_pollution_clean[,!names(air_pollution_clean) %in% c("No","station","PM10","SO2","NO2","CO","O3")]

data_knn <- data_knn %>% mutate_if(is.factor,as.numeric) # transform wd into numeric
data_knn <- na.omit(data_knn)
length(which(is.na(data_knn)))

sapply(data_knn,class)

set.seed(2019)

fold_ids      <- rep(seq(10), 
                     ceiling(nrow(data_knn) / 10))
fold_ids      <- fold_ids[1:nrow(data_knn)]

fold_ids      <- sample(fold_ids, length(fold_ids))

for (k in 1:10) {
  for (fold in 1:10) {
    
    ## Train the KNN model (Note: if it throws a weird error, make sure
    ## all features are numeric variables -- not factors)
    knn_fold_model    <- knn.reg(data_knn[which(fold_ids != fold),-c(5)],
                             data_knn[which(fold_ids == fold),-c(5)],
                             data_knn[which(fold_ids != fold),5],
                             k = k)
    
    ## Measure and save error rate (% wrong)
    CV_MSEP_mtx_knn[k,fold]  <- 
      mean((knn_fold_model$pred - data_knn[which(fold_ids == fold),5])^2, na.rm=TRUE)
  }
}

rowMeans(CV_MSEP_mtx_knn) # MSE for knn model is 704.6 for k=5 => BETTER THAN linear regression!!!

# Fourth, try Lasso

lasso_air       <- cv.glmnet(x = as.matrix(air_pollution_clean
                                           [,!names(air_pollution) %in% c("No","station_Aotizhongxin", "PM2.5")]),
                             y = as.numeric(air_pollution_clean[,5]),
                             alpha = 1, standardize = T)

print(lasso_air$lambda)

## And we can see how the cross-validation error varied by lambda

print(round(lasso_air$cvm,4))
lambda_min <- lasso_air$lambda.min

# Now try to predict using lasso model
set.seed(2019)
train               <- sample(seq(nrow(air_pollution_clean)),
                              floor(nrow(air_pollution_clean) * 0.8))
train               <- sort(train)
test                <- which(!(seq(nrow(air_pollution_clean)) %in% train))


lasso_optimal      <- glmnet(x = as.matrix(air_pollution_clean[train,!names(air_pollution) %in% 
                                                                 c("No","station_Aotizhongxin", "PM2.5")]),
                             y = as.numeric(air_pollution_clean[train,5]),
                             alpha = 1, standardize = T, lambda=lambda_min)

lasso_pred <- predict(lasso_optimal, newx= as.matrix(air_pollution_clean[test,!names(air_pollution) %in% 
                                                                     c("No","station_Aotizhongxin", "PM2.5")]))

mean((lasso_pred - air_pollution_clean[test,"PM2.5"])^2, na.rm=TRUE) 

# MSE lasso is 11914, which is horrible...

# Fifth, try random forest- here we don't do CV to save computing time, RF relatively less biased

rf_PM2.5_mod    <- randomForest(formula = PM2.5 ~ .- No - station_Aotizhongxin, 
                                na.action=na.exclude, 
                                data = data.frame(air_pollution_clean[train,]), 
                                do.trace=TRUE, importance=TRUE,ntree=100)
mean(rf_PM2.5_mod$mse)

# Now we check test MSE for randomforest
rf_PM2.5_pred <- predict(rf_PM2.5_mod, 
                         newdata = data.frame(air_pollution_clean[test,]))

mean((rf_PM2.5_pred - air_pollution_clean[test,"PM2.5"])^2, na.rm=TRUE) 
# MSE for rf=324.3 so far the best model                   
                         
# Code to cross-validate optimal ntree for randomforest, not used yet

for (ntree in 1:10) {
  
  rf_PM2.5_mod    <- randomForest(formula = PM2.5 ~ .- No - station_Aotizhongxin, 
                                      na.action=na.exclude, 
                                      data = data.frame(air_pollution_clean[which(fold_ids != fold),]), 
                                      do.trace=TRUE, importance=TRUE, ntree=100)
  rf_PM2.5_pred <- predict(rf_PM2.5_mod, 
                             newdata = data.frame(air_pollution_clean[which(fold_ids == fold),]))
  
  truth <- air_pollution_clean[which(fold_ids == fold), "PM2.5"]
  
  CV_MSEP_mtx[1,fold]  <-  mean((rf_PM2.5_pred - truth)^2, na.rm=TRUE)
}

# Sixth, let's try boosting...

boost.air <- gbm(PM2.5~.-No -station_Aotizhongxin, data=data.frame(air_pollution_clean[-test,]), 
                     distribution= "gaussian", n.trees=500, interaction.depth=4)

summary(boost.air)  
boost.air_pred <- predict(boost.air , newdata=air_pollution_clean[-train ,], n.trees=100)
summary(boost.air_pred)

mean((boost.air_pred - air_pollution_clean[test,"PM2.5"])^2, na.rm=TRUE) 
#Boost MSE=607.7
