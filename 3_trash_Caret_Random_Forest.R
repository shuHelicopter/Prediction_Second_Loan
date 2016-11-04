library(randomForest)

train1 <- read.csv('./frd_data/train.csv', sep = ',', header = T)
train2 <- read.csv('./frd_data/cross.csv', sep = ',', header = T)
train_frd <- rbind(train1, train2)
test_frd <- read.table('./frd_data/test.csv', sep = ',', header = T)

##########################
####  Build gbm model ####
##########################

library(caret)
###### Check the documentation for the details of those functions
###### https://cran.r-project.org/web/packages/caret/caret.pdf

library(doMC)
registerDoMC(cores = 4)


###### Setup a 5 fold cross-validation and use AMS as the metric
###### AMS_summary function defined in helper.R
###### Check details here: http://topepo.github.io/caret/training.html#control
ctrl = trainControl(method = "repeatedcv", number = 5, repeats = 3)

#labels <- ifelse(labels=='s', 1, 0)

####################################
####  Build random forest model ####
####################################
# ensure results are repeatable

set.seed(123)
rfGrid <-  expand.grid(mtry = c(2:8))

m_rf = train(x=train_frd[, -15], y=as.integer(train_frd[, 15])-1, 
             method="rf", verbose=TRUE, trControl=ctrl, 
             metric="AUC", tuneGrid = rfGrid, 
             ntree = 1000)

plot(m_rf)

############################# Test ######################################
rfPred <- predict(m_rf, newdata = test_frd[, ], type = 'prob')
submit_rf <- cbind(test_index, rfPred)
colnames(rfPred) <- c('user_id', 'probability')
write.table(submit_rf, file = './Submit/submit_rf_frd.txt'), row.names = F, col.names = T, quote = F, sep = ',')


