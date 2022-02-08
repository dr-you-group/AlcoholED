library(dplyr)
library(caret)

nedis <- read.csv("./data/nedis_utf8.csv")

##Set seed number
seedNum = 1234 #If you don't want to set seed, then let this be NULL. Otherwise put any number

##########################################################################################################
####Helper functions####
##Normalization of the elements in the data
minmaxNormalize <- function(x) {return ((x-min(x,na.rm = T)) / (max(x,na.rm = T)-min(x,na.rm = T)))}
zNormalize <- function(x) {return ( (x-mean(x,na.rm = T))/sd(x,na.rm = T))}
##########################################################################################################

##Set proportion for the test set
testProportion = 0.3

##set the outcome
nedis$revisit <- ifelse(nedis$N_ID>1, "yes", "no")
positiveClass <- "yes"
outcomeName = "revisit" #The name of column
outcome <- nedis %>% pull(all_of(outcomeName)) %>% factor(levels = c("no", "yes"))
is.binary = T
##Set number and metric for cross validation
crossValNum = 3
crossValMetric = "ROC" #ROC only for 'binary class', possible values are "RMSE" and "Rsquared" for regression and "Accuracy" and "Kappa" for classification

##set variables
categoricalVars <- c("PTMISEXX", 
                     "PTMIIUKD", 
                     "PTMIDGKD", 
                     "PTMIARCF", 
                     "PTMIARCS",
                     "PTMIINMN", 
                     "PTMIMNSY", 
                     "PTMIRESP",
                     "MSICK_CD")
numericalVars <- c("diffMin",
                   "PAT_AGE")

##Extract only three letters for Diagnosis ICD-10 code

nedisXCat <- nedis %>% select(all_of(categoricalVars))
nedisXCat$MSICK_CD <- stringr::str_extract(nedis$MSICK_CD, "[A-Z][0-9][0-9]")

# set '-' to NA and make all variables to string
for(col in colnames(nedisXCat)){
  nedisXCat[,col] <- as.character(nedisXCat[,col] )
  index <- pull(nedisXCat, col)=="-"
  nedisXCat[index,col] <- NA
}
#One hot encoding of categorical variables
dummy <- caret::dummyVars("~ .", data = nedisXCat) 
nedisXCatD <- data.frame(predict(dummy, newdata = nedisXCat))
# nedisXCatD <- as.data.frame(lapply(nedisXCatD, as.factor)) #convert to factor
nedisXCatD [is.na(nedisXCatD)] <- 0 #replace NA to 0 (glmnet in caret cannot handle it )

#Regularization of numerical variables
nedisXNum <- nedis %>% select(all_of(numericalVars))
nedisXNumR <- as.data.frame(lapply(nedisXNum, zNormalize))

feature <- nedisXCatD
#feature <- cbind(nedisXNumR, nedisXCatD)

##split train and test
set.seed(seedNum)
#testInd<-sample.int(n= nrow(wdbc_p), size = floor(testProportion* nrow(wdbc_p)), replace = F)
testInd <- caret::createDataPartition(outcome, p = testProportion, list = F) #To split total data set while preserving the proportion of outcomes in train and test set

featureTrain <- feature[-testInd,]
featureTest  <- feature[testInd,]

outcomeTrain <- outcome[-testInd]
outcomeTest <- outcome[testInd]

#Checking distibution in origanl data and partitioned data
prop.table(table(outcomeTrain)) * 100 ##you can see the degree of imbalance
prop.table(table(outcomeTest)) * 100

##set train control in caret package
if(is.binary){
  fitControl <- caret::trainControl(method = "repeatedcv", number = crossValNum, repeats = 5,
                                    classProbs=TRUE, summaryFunction = caret::twoClassSummary)
}else{
  fitControl <- caret::trainControl(method = "repeatedcv", number = crossValNum, repeats = 5,
                                    classProbs=FALSE) 
}
#re-sampling of imbalanced dataset
fitControl$sampling <- "rose" #"smote" #"down" #"up"

#For parallelization to speed up!(only in Unix. You can use in Mac, too )
library(doMC)
library(parallel)
getDoParWorkers()
registerDoMC(cores = parallel::detectCores())
getDoParWorkers()
##########################################################################################################

####Lasso Regression####
customGrid = expand.grid(alpha = 1,
                         lambda = seq(0.001, 0.1, by =0.001))
set.seed(seedNum)

trainFit <- caret::train(x=featureTrain,y=outcomeTrain, 
                         method = "glmnet", 
                         trControl = fitControl, 
                         # family = "binomial",
                         #preProcess = c("center","scale"), 
                         metric = crossValMetric,
                         tuneGrid = customGrid) 
#Output of the fit

#check the importance of the variables
caret::varImp(trainFit)
plot(caret::varImp(trainFit))

####Performance of model####
#applying the trained model to the test set
predictClass <- predict(trainFit,newdata = featureTest)
predictProb <- predict(trainFit,newdata = featureTest,type ="prob")
#Get the confusion matrix to see accuracy value and other parameter values
caret::confusionMatrix(predictClass, outcomeTest,positive = positiveClass)
#the accuracy
mean(predictClass == outcomeTest) 

#plot ROC curve
Epi::ROC(form=outcomeTest~predictProb[,2], plot="ROC")
out= Epi::ROC(form=outcomeTest~predictProb[,2], plot="ROC")
out$AUC #0.73