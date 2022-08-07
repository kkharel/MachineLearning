# Use the data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants
# They were asked to perform barbell lifts correctly and incorrectly in 5 different ways

# source data comes from:  http://groupware.les.inf.puc-rio.br/har

# Goal: to predict the manner in which particiapnts did the exercise

# classe variable in the training set represents the manner

# may use any of the other variable to predict with

# create a report describing how I built my model,
# how I used cross validation,
# what I think the expected out of sample error is,
# and why I made the choices you did.
# I will also use my prediction model to predict 20 different test cases.

library(caret); library(kernlab);library(randomForest)
traincsv = read.csv("C:\\Users\\kkhar\\OneDrive\\Documents\\Practical Machine Learning\\pml-training.csv")
testcsv = read.csv("C:\\Users\\kkhar\\OneDrive\\Documents\\Practical Machine Learning\\pml-testing.csv")
names(traincsv)
names(testcsv)

dim(traincsv); dim(testcsv)

# removing non-numeric variables from training set
traincsv = subset(traincsv, select = -c(X,raw_timestamp_part_1,raw_timestamp_part_2, cvtd_timestamp, new_window, num_window, user_name))

# removing non-numeric variables from testing set
testcsv = subset(testcsv, select = -c(X,raw_timestamp_part_1,raw_timestamp_part_2, cvtd_timestamp, new_window, num_window, user_name))


# creating a vector with all of the column classes from train set

columnClasses <- sapply(traincsv,class)
columnClasses
grep("classe", colnames(traincsv)) # getting the column number of classe variable


# creating a vector with all of the column classes from test set
columnClassestest = sapply(testcsv, class)
columnClassestest

# converting all data columns to numeric except classe variable in training set

for(i in 1:152){
  if(columnClasses[i] != "numeric"){
    if(columnClasses[i] == "factor"){
      traincsv <- traincsv[,ifelse(NAcount == 0, TRUE, FALSE)]
      traincsv[traincsv[,i] == "",i] <- NA
      traincsv[,i] <- as.numeric(as.character(traincsv[,i]))
    }
    traincsv[,i] <- as.numeric(traincsv[,i])
  }
}

# converting all data columns to numeric from testing set
for(i in 1:152){
  if(columnClassestest[i] != "numeric"){
    if(columnClassestest[i] == "factor"){
      testcsv <- testcsv[,ifelse(NAcount == 0, TRUE, FALSE)]
      testcsv[testcsv[,i] == "",i] <- NA
      testcsv[,i] <- as.numeric(as.character(testcsv[,i]))
    }
    testcsv[,i] <- as.numeric(testcsv[,i])
  }
}

# total number of NA values in training set
library(dplyr)
traincsv %>%
  summarise(count = sum(is.na(traincsv)))

# total number of NA values in testing set
testcsv %>%
  summarise(count = sum(is.na(testcsv)))

# taking the glimpse on the dataset
library(tidyverse)
glimpse(traincsv)
glimpse(testcsv)

# remove all rows and columns with more than 80% NA in train set
# come back to this later (dropping only columns or rows too)
#newtrain = traincsv[which(rowMeans(!is.na(traincsv)) > 0.80), which(colMeans(!is.na(traincsv)) > 0.80)]
newtrain = traincsv[, which(colMeans(!is.na(traincsv)) > 0.80)]
newtrain$classe = as.factor(newtrain$classe)

# remove all columns with more than 80% NA in test set
newtest = testcsv[, which(colMeans(!is.na(testcsv)) > 0.80)]

# checking if NA's are present in the training dataset
sum(is.na(newtrain))

# checking if NA's are present in the testing dataset
sum(is.na(newtest))

#removing nerozerovariance variables from the dataset if any of them exists

nzv = nearZeroVar(newtrain[,-53], freqCut = 95/5, uniqueCut = 10, saveMetrics = TRUE, allowParallel = TRUE)
nzv

# since none of the variables has nzv true, we are including all of the variables in our model

# looking at the histogram of data
library(Hmisc)

par(mar=c(1,1,1,1))

Hmisc::hist.data.frame(newtrain[-53])
# dividing the training set into validation set, sub-training set and testing set

x= newtrain[-53]
x = scale(x, center=TRUE, scale = TRUE)

Hmisc::hist.data.frame(x)

inBuild = createDataPartition(y = newtrain$classe, p = 0.75, list = FALSE)

validation = newtrain[-inBuild,]; buildData = newtrain[inBuild, ]
inTrain = createDataPartition(y = buildData$classe, p = 0.75, list = FALSE)

training = buildData[inTrain, ]
testing = buildData[-inTrain, ]


# standardizing the data in training
modelFit1 = train(classe~., data = training,
                 preProcess = c("center", "scale"), method = "rpart",
                 trControl = trainControl(method="cv", number=3, verboseIter=F), tuneLength = 4) # setting up control for training to use 3-fold cross validation


par(mfrow=c(1,1))

# Plotting decision tree
rpart.plot::rpart.plot(modelFit1$finalModel, cex = 0.5, type = 4, under = TRUE)

prediction1 = predict(modelFit1, testing)
cmdecisiontree <- confusionMatrix(prediction1, factor(testing$classe))
cmdecisiontree

# decision tree is not able to accurately classify the classe

# Now, let's try random forest

modelFit2 = randomForest::randomForest(classe~., data = training,
                                       preProcess = c("center","scale"),
                                       trControl = trainControl(method="cv", number=3, verboseIter=F),  # setting up control for training to use 3-fold cross validation
                                       importance = TRUE, proximity = TRUE
                                       )
modelFit2
getTree(modelFit2, k = 4)
plot(modelFit2)


#produce variable importance plot (importance of each predictor variable in the final model)
varImpPlot(modelFit2,
           type=NULL, class=NULL, scale=TRUE,
           main=deparse(substitute(modelFit2)))

# The x-axis displays the average increase in node purity of the regression trees
# based on splitting on the various predictors displayed on the y-axis.

# From the plot we can see that yaw_belt is the most important predictor variable,
# followed closely by roll_belt.


# tuning the model
# ntreeTry: The number of trees to build.
# mtryStart: The starting number of predictor variables to consider at each split.
# stepFactor: The factor to increase by until the out-of-bag estimated error stops
# improving by a certain amount.
# improve: The amount that the out-of-bag error needs to improve by to
# keep increasing the step factor

# hyper parameter tuning
tuned_model = tuneRF(
  x= training[,-53], #define predictor variables
  y=training$classe, #define response variable
  ntreeTry=500,
  stepFactor=1.5,
  improve=0.01,
)
print(tuned_model)

# We can see the number of predictors used at each split when building the trees
# on the x-axis and the out-of-bag estimated error on the y-axis

# We can see that the lowest OOB error is achieved by using 7 randomly chosen predictors
# at each split when building the trees which is default in function above.

prediction2 = predict(modelFit2, testing)
cmrandomforest = confusionMatrix(prediction2, factor(testing$classe))
cmrandomforest

# Gradient Boosted Trees

modelFit3 = train(classe~., method = "gbm", data = training,
                preProcess = c("center", "scale"),
                verbose = FALSE) #boosting with trees

plot(modelFit3)
prediction3 = predict(modelFit3, testing)

cmboostingtrees = confusionMatrix(prediction3, testing$classe)
cmboostingtrees

# Support Vector Machine

modelFit4 = train(classe~., method = "svmLinear",
                  data = training,
                  preProcess = c("center", "scale"),
                  trControl = trainControl(method="cv", number=3, verboseIter=F),  # setting up control for training to use 3-fold cross validation
                  tuneLength = 4)

prediction4 = predict(modelFit4, testing)

cmsupportvector = confusionMatrix(prediction4, factor(testing$classe))
cmsupportvector

# Ensemble Method (combining two methods that have low accuracy, model stacking)

# Stack the predictions together using random forests ("rf").
comb = data.frame(prediction1, prediction2,prediction3, prediction4, classe = testing$classe)
modelComb = randomForest(classe~., data = comb,
                         preProcess = c("center,scale"),
                         trControl = trainControl(method="cv", number=3, verboseIter=F),  # setting up control for training to use 3-fold cross validation
                         importance = TRUE, proximity = TRUE)

getTree(modelComb, k = 4)
predictionComb = predict(modelComb, testing)
cmensemble = confusionMatrix(predictionComb, comb$classe)
cmensemble
# What is the resulting accuracy on the test set?
# Is it better or worse than each of the individual predictions?

confusionMatrix(prediction1, testing$classe)$overall[1]
confusionMatrix(prediction2, testing$classe)$overall[1]
confusionMatrix(prediction3, testing$classe)$overall[1]
confusionMatrix(prediction4, testing$classe)$overall[1]
confusionMatrix(predictionComb, testing$classe)$overall[1]

# We can see that combined has much less error

# Prediction on validation data set
pred1v = predict(modelFit1, validation);
pred2v = predict(modelFit2, validation);
pred3v = predict(modelFit3, validation);
pred4v = predict(modelFit4, validation);

predvdf = data.frame(prediction1 = pred1v, prediction2 = pred2v,
                     prediction3 = pred3v, prediction4 = pred4v)

combpredv = predict(modelComb,predvdf)

# Evaluate on Validation
cmdecisiontreev = confusionMatrix(pred1v, factor(validation$classe))$overall[1]
cmrandomforestv = confusionMatrix(pred2v, factor(validation$classe))$overall[1]
cmboostingtreev = confusionMatrix(pred3v, factor(validation$classe))$overall[1]
cmsupportvectorv = confusionMatrix(pred4v, factor(validation$classe))$overall[1]
cmensemblev = confusionMatrix(combpredv, validation$classe)$overall[1]

models <- c("Tree", "RF", "GBM", "SVM", "Ensemble")
accuracy <- round(c( cmdecisiontreev, cmrandomforestv,
                     cmboostingtreev, cmsupportvectorv,
                     cmensemblev),3) #accuracy

oos_error <- 1 - accuracy #out of sample error

data.frame(accuracy = accuracy, oos_error = oos_error, row.names = models)

# Plotting the models

library(ggplot2)
par(mfrow=c(1,1))
ggplot(modelFit1) + theme_bw()
plot(modelFit2)
ggplot(modelFit3) + theme_bw()


# Prediction on testing dataset
predc = predict(modelFit2, newtest)
predc


getwd()

