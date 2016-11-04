################ random forest #############
############################################
##### Train_1 #####
###### correlation map #######
train_1[is.na(train_1)] <- 0
library(corrplot)
train_1_corr <- as.data.frame(sapply(train_1[, -c(1, 21)], as.numeric))
M_train_1 <- cor(train_1_corr)
corrplot(M_train_1)
library(randomForest)
set.seed(0)

# tree = 500
oob.err1 = numeric(15)
for (mtry in 1:15) {
  fit1 = randomForest(lable ~ .-user_id -product_id, data = train_1, mtry = mtry)
  oob.err1[mtry] = fit1$err.rate[500]
  cat('We are performing iteration', mtry, '\n')
}

plot(1:15, oob.err1, pch = 16, type = 'b',
     xlab = 'Variance Cosidered at Each Split', 
     ylab = 'OOB Mean Squared Error', 
     main = '1_Random Forest OOB Error Rates\nby # of Variables')
# variane important:
varImpPlot(fit1)
oob.err1[1]

# tree = 1000
oob.err2 = numeric(15)
for (mtry in 1:15) {
  fit2 = randomForest(lable ~ .-user_id -product_id, data = train_1, mtry = mtry, ntree = 1000)
  oob.err2[mtry] = fit2$err.rate[500]
  cat('We are performing iteration', mtry, '\n')
}

plot(1:15, oob.err2, pch = 16, type = 'b',
     xlab = 'Variance Cosidered at Each Split', 
     ylab = 'OOB Mean Squared Error', 
     main = '2_Random Forest OOB Error Rates\nby # of Variables')
# variance important:
varImpPlot(fit2) #
oob.err2[1]

# tree = 5000
oob.err3 = numeric(5)
for (mtry in 1:5) {
  fit3 = randomForest(lable ~ .-user_id -product_id, data = train_1, mtry = mtry, ntree = 5000)
  oob.err3[mtry] = fit3$err.rate[500]
  cat('We are performing iteration', mtry, '\n')
}

plot(1:5, oob.err3[1:5], pch = 16, type = 'b',
     xlab = 'Variance Cosidered at Each Split', 
     ylab = 'OOB Mean Squared Error', 
     main = '3_Random Forest OOB Error Rates\nby # of Variables')
# variance important:
varImpPlot(fit3)
oob.err3[1] # [1] 0.3673775

# mtry = 2 # we believe this model is the most appropriate one
fit0 <- randomForest(lable ~ .-user_id -product_id, data = train_1, mtry = 2, ntree = 5000)
fit0
varImpPlot(fit0)

'''
Call:
randomForest(formula = lable ~ . - user_id - product_id, data = train_1,      mtry = 2, ntree = 5000) 
Type of random forest: classification
Number of trees: 5000
No. of variables tried at each split: 2

OOB estimate of  error rate: 36.92%
Confusion matrix:
0    1 class.error
0 4346 6503   0.5994101
1 1392 9143   0.1321310
'''

# mtry = 3
fit0.1 <- randomForest(lable ~ .-user_id -product_id, data = train_1, mtry = 3, ntree = 5000)
fit0.1
varImpPlot(fit0.1)

'''
Call:
randomForest(formula = lable ~ . - user_id - product_id, data = train_1,      mtry = 3, ntree = 5000) 
Type of random forest: classification
Number of trees: 5000
No. of variables tried at each split: 3

OOB estimate of  error rate: 37.06%
Confusion matrix:
0    1 class.error
0 4858 5991   0.5522168
1 1933 8602   0.1834836
'''

##### Train_2 #####
summary(aggr(train_2, prop = T, number = T, gap = T, only.miss = T))
###### correlation map #######
train_2 <- train_2[, -c(10, 11, 17)] # remove columns 'live_info' 'company_type' with constant value and 'num_rel'

library(corrplot)
train_2_corr <- as.data.frame(sapply(train_2[, -c(1, 11)], as.numeric)) # remove 'user_id' 'product_id'
M_train_2 <- cor(train_2_corr)
corrplot(M_train_2)

library(randomForest)
set.seed(0)

# tree = 500
oob.err11 = numeric(5)
for (mtry in 1:5) {
  fit11 = randomForest(lable ~ .-user_id -product_id, data = train_2, mtry = mtry)
  oob.err11[mtry] = fit11$err.rate[500]
  cat('We are performing iteration', mtry, '\n')
}

plot(1:5, oob.err11, pch = 16, type = 'b',
     xlab = 'Variance Cosidered at Each Split', 
     ylab = 'OOB Mean Squared Error', 
     main = '11_Random Forest OOB Error Rates\nby # of Variables')
# variane important:
varImpPlot(fit11)
oob.err11[4] # 0.4746534

# tree = 1000
oob.err12 = numeric(10)
for (mtry in 1:10) {
  fit12 = randomForest(lable ~ .-user_id -product_id, data = train_2, mtry = mtry, ntree = 1000)
  oob.err12[mtry] = fit12$err.rate[500]
  cat('We are performing iteration', mtry, '\n')
}

plot(1:10, oob.err12, pch = 16, type = 'b',
     xlab = 'Variance Cosidered at Each Split', 
     ylab = 'OOB Mean Squared Error', 
     main = '2_Random Forest OOB Error Rates\nby # of Variables')
# variance important:
varImpPlot(fit12) 
which(oob.err12 == min(oob.err12))
oob.err12[7] # 0.4753033


# tree = 5000
oob.err13 = numeric(10)
for (mtry in 1:10) {
  fit13 = randomForest(lable ~ .-user_id -product_id, data = train_2, mtry = mtry, ntree = 5000)
  oob.err13[mtry] = fit13$err.rate[500]
  cat('We are performing iteration', mtry, '\n')
}

plot(1:10, oob.err13[1:10], pch = 16, type = 'b',
     xlab = 'Variance Cosidered at Each Split', 
     ylab = 'OOB Mean Squared Error', 
     main = '3_Random Forest OOB Error Rates\nby # of Variables')
# variance important:
varImpPlot(fit13)
which(oob.err13 == min(oob.err13)) # 8
oob.err13[8] # 0.4729203

# tree = 10000
oob.err14 = numeric(10)
for (mtry in 1:10) {
  fit14 = randomForest(lable ~ .-user_id -product_id, data = train_2, mtry = mtry, ntree = 10000)
  oob.err14[mtry] = fit14$err.rate[500]
  cat('We are performing iteration', mtry, '\n')
}

plot(1:10, oob.err14[1:10], pch = 16, type = 'b',
     xlab = 'Variance Cosidered at Each Split', 
     ylab = 'OOB Mean Squared Error', 
     main = '4_Random Forest OOB Error Rates\nby # of Variables')
# variance important:
varImpPlot(fit14)
which(oob.err14 == min(oob.err14)) # 10
oob.err14[10] # 0.472487

# tree = 20000
oob.err15 = numeric(11)
for (mtry in 1:11) {
  fit15 = randomForest(lable ~ .-user_id -product_id, data = train_2, mtry = mtry, ntree = 20000)
  oob.err15[mtry] = fit15$err.rate[500]
  cat('We are performing iteration', mtry, '\n')
}

plot(1:11, oob.err15[1:11], pch = ?, type = 'b',
     xlab = 'Variance Cosidered at Each Split', 
     ylab = 'OOB Mean Squared Error', 
     main = '5_Random Forest OOB Error Rates\nby # of Variables')
# variance important:
varImpPlot(fit15)
which(oob.err15 == min(oob.err15)) # 10
oob.err15[10] # 0.472487

##### best model ######
# mtry = 3
fitt0 <- randomForest(lable ~ .-user_id -product_id, data = train_2, mtry = 3, ntree = 10000)
fitt0
varImpPlot(fitt0)

############# Prediction #################
###### test data ############
test <- left_join(test_index, train_mrg, by = 'user_id') # build a test dataset

# split train dataset into two datasets by product_id(1, 2)
index2 <- which(test$product_id == 1) # index of product 1
test_1 <- test[index2, -c(5)] # type = 1
test_2 <- test[-index2, -c(10, 11, 14:20)] # type = 2
summary(aggr(test_1, prop = T, number = T, gap = T, only.miss = T)) # check the missingness of train_1
summary(aggr(test_2, prop = T, number = T, gap = T, only.miss = T)) # check the missingness of train_2
'''
Missings per variable: 
Variable Count
user_id     0
age     0
sex     0
expect_quota     0
occupation     0
education     0
marital_status     0
live_info     0
local_hk     0
money_function     0
company_type     0
salary     0
school_type     0
flow     0
gross_profit     0
business_type     0
business_year     0
personnel_num     0
pay_type     2
product_id     0
tm_encode     0
nrows_unique     0
nrows     0
num_rel   299

Missings in combinations of variables: 
Combinations Count     Percent
0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0  7401 96.09192418
0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:1   299  3.88210854
0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:1:0:0:0:0:0     2  0.02596728
'''

'''
Missings per variable: 
Variable Count
user_id     0
age     1
sex     1
expect_quota     1
max_month_repay     1
occupation     1
education     1
marital_status     1
live_info     1
company_type     1
salary     1
product_id     1
tm_encode     1
nrows_unique     1
nrows     1
num_rel  3222

Missings in combinations of variables: 
Combinations Count     Percent
0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0  1337 29.32660671
0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:1  3221 70.65145865
0:1:1:1:1:1:1:1:1:1:1:1:1:1:1:1     1  0.02193463
'''
###### prediction of test_1 #######
test_1[is.na(test_1)] <- 0
pred_prob1 <- predict(fit0, test_1, type = 'prob')
submit_1 = as.data.frame(cbind(test_1[, 1], pred_prob1))[, c(1, 3)]

###### prediction of test_2 #######
test_2 <- test_2[, -c(9, 10, 16)] # remove columns 'live_info' 'company_type' with constant value and 'num_rel'
test_2[is.na(test_2)] <- 0 # encode 'NA' with 0
# check one user with full missingness
rownames(test_2)[rowSums(is.na(test_2)) > 0]
test_2[rownames(test_2) == '9528', 1] # user_id = "59620cccd3a9eee713b6c95c398973d7"
# rowname = '9528', this is a user without any information for variables

pred_prob2 <- predict(fitt0, test_2[rownames(test_2) != '9528', ], type = 'prob')
submit_2 = as.data.frame(cbind(test_2[rownames(test_2) != '9528', 1], pred_prob2))[, c(1, 3)]
submit_2[, 1] = as.character(submit_2[, 1])
nrow(submit_2) # 4558, one user left
submit_2[4559, ] = c('59620cccd3a9eee713b6c95c398973d7', 0.5) # add the special user

####### complete prediction #######
submit = rbind(submit_1, submit_2)
# format the file
colnames(submit) <- c('user_id', 'probability') # change the name
submit <- left_join(test_index, submit, by = 'user_id') # change back to the orginal order
write.csv(submit, 'submit.csv') # save to .csv file
write.table(submit, file = 'submit.txt', row.names = F, col.names = T, quote = F, sep = ',')
