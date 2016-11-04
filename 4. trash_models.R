################ XGboost1 -> user_info + rela1 + consump  ##################
############################################################################

########## XGBoost with Caret package -> cannot handle missingness, so quit ###########
'''
set.seed(0)
ctrl_1 = trainControl(method = 'repeatedcv', number = 5, verboseIter = T, 
summaryFunction = twoClassSummary, classProbs = T)
searchgrid_1 <- expand.grid(nrounds = c(50, 80, 150), max_depth = 9, eta = 0.1, 
colsample_bytree = 0.7, gamma = 5, min_child_weight = 0.1)

begin_time = Sys.time()
print(paste0('Starting training time: ', begin_time))
train_1[, 2] <- make.names(train_1[, 2])
xgb_1 = train(x = train_1[, -c(1, 2, 21)], y = train_1[, 2], method = 'xgbTree', 
tuneGrid = searchgrid, verbose = 2, trControl = ctrl)
end_time = Sys.time()
print(paste0('Training finished at: ', end_time))
print(paste0('Training takes: ', end_time - begin_time))

best_model_1 = xgb_1$bestTune
file_name = paste0('nrounds = ', best_model_1$nrounds, ', max_depth = ', best_model_1$max_depth, 
', eta = ', best_model_1$eta, ', colsample_bytree = ', best_model_1$colsample_bytree, 
'gamma = ', best_model_1$gamma, ',min_child_weight = ', best_model_1$min_child_weight)

# ================= save models to images ====================
jpeg(paste0('images/model_paras_', file_name, '.jpg'), width = 800, height = 600)
plot(xgb_1)
dev.off()

xgbPred_1 <- predict(xgb_1, newdata = test_1[, -c(1, 20)], type = 'prob')
'''
########################################################################################
library(dplyr)
library(xgboost)

xgb_data <- sapply(train[, -c(1, 2)], as.numeric)
xgb_train <- xgb.DMatrix(data = xgb_data, label = train[, 2], missing = NA)

xgb_datatest <- sapply(test[, -1], as.numeric)
xgb_test <- xgb.DMatrix(data = xgb_datatest, missing = NA)

xgb_model <- xgb.train(data = xgb_train, max_depth = 9, eta = .01, sub_sample = .9, 
                       nround = 3000, objective = 'binary:logistic')
xgb_pred <- predict(xgb_model, xgb_test)

# transform to submit file
submit_xgb0 <- cbind(test[, 1], xgb_pred)
colnames(submit_xgb1) <- c('user_id', 'probability')
write.table(submit_xgb1, file = 'submit_xgb1.txt', row.names = F, col.names = T, quote = F, sep = ',')

################################# with consumption ####################################
consump <- read.csv('df_con_.csv')

train_consump <- left_join(train, consump, by = 'user_id')
test_consump <- left_join(test, consump, by = 'user_id')

xgb_data1 <- sapply(train_consump[, -c(1, 2)], as.numeric)
xgb_train1 <- xgb.DMatrix(data = xgb_data1, label = train_consump[, 2], missing = NA)

xgb_datatest1 <- sapply(test_consump[, -1], as.numeric)
xgb_test1 <- xgb.DMatrix(data = xgb_datatest1, missing = NA)

xgb_model1 <- xgb.train(data = xgb_train1, max_depth = 9, eta = .01, sub_sample = .9, 
                        nround = 3000, objective = 'binary:logistic')
xgb_pred1 <- predict(xgb_model1, xgb_test1)

# check importance of features
names1 <- dimnames(xgb_data1)[[2]]
importance_matrix1 <- xgb.importance(names1, model = xgb_model1)
xgb.plot.importance(importance_matrix1[1:10, ])

# transform to submit file
submit_xgb1 <- cbind(test_consump[, 1], xgb_pred1)
colnames(submit_xgb1) <- c('user_id', 'probability')
write.table(submit_xgb1, file = 'submit_xgb1.txt', row.names = F, col.names = T, quote = F, sep = ',')

################################# with consumption, split into two models ##############
# check the missingness of train_consump
summary(aggr(train_consump, prop = T, number = T, gap = T, only.miss = T))
# check the missingness from consumption information
summary(aggr(train_consump[, c(1, 2, 22, 27:38)], prop = T, number = T, gap = T, only.miss = T))




################ XGboost2 -> user_info + rela1 + consump_clean0 ##################
############################################################################
set.seed(105)
cv.res <- xgb.cv(data = , nfold = 5, label = , nround = 3000, objective = 'binary:logistic', eval_metric = 'auc')
cv.res # return a data.table object containing the cross validation results, which is helpful for chossing the correct iterations

############# model 1: product_type = 1 ##############
xgb_data2.0 <- sapply(train_1[, -c(1, 2)], as.numeric)
xgb_train2.0 <- xgb.DMatrix(data = xgb_data2.0, label = train_1[, 2], missing = NA)

xgb_cv_model2.0 <- xgb.cv(data = xgb_train2.0, max_depth = 7, eta = .01, sub_sample = .9, 
                          nround = 500, eval_metric = 'auc', objective = 'binary:logistic', 
                          early.stop.round = 200, prediction = T, nfold = 5)

xgb_cv_model2.0
# nrounds, eta, max_depth
# early.stop.round is used to avoid overfitting
# If overfitting observed, reduce stepsize eta and increase nround at the same time.
### max_depth = 9, early.stop.round = 200
# seed = 105, nround = 300, best = 300: [299] train-auc:0.844352+0.007536	test-auc:0.667742+0.001953
# seed = 105, nround = 500, best = 167: [166]	train-auc:0.819525+0.004736	test-auc:0.664645+0.009051
# seed = 105, nround = 1000, best = 328: [327]	train-auc:0.841931+0.006249	test-auc:0.666858+0.002319
# seed = 105, nround = 2000, best = 318: [317]	train-auc:0.844811+0.008292	test-auc:0.666135+0.005498
# seed = 105, nround = 3000, best = 298: [297]	train-auc:0.842452+0.009338	test-auc:0.664492+0.002666

### max_depth = 12, early.stop.round = 500
# seed = 105, nround = 3000, best = 352: [351]	train-auc:0.935483+0.008792	test-auc:0.658165+0.004535
### max_depth = 15, early.stop.round = 500
# seed = 105, nround = 3000, best = 426: [425]	train-auc:0.987734+0.002690	test-auc:0.656741+0.007574

### max_depth = 15, early.stop.round = 200
# seed = 105, nround = 3000, best = 483: [482]	train-auc:0.989436+0.003427	test-auc:0.655728+0.006508

### max_depth = 10, early.stop.round = 200
# seed = 105, nround = 3000, best = 181: [180]	train-auc:0.858964+0.010110	test-auc:0.662397+0.003953
# seed = 105, nround = 1000, best = 302: [301]	train-auc:0.873834+0.006803	test-auc:0.662872+0.006273

### max_depth = 8, early.stop.round = 200
# seed = 105, nround = 1000, best = 200: [199]	train-auc:0.795741+0.002447	test-auc:0.666972+0.005977

### max_depth = 7, early.stop.round = 200
# seed = 105, nround = 500, best = 269: [268]	train-auc:0.770949+0.004435	test-auc:0.670817+0.009324 **
# seed = 105, nround = 500, best = 300, sub_sample = .8: [299]	train-auc:0.775846+0.001979	test-auc:0.670077+0.004527

### max_depth = 6, early.stop.round = 200
# seed = 105, nround = 1000, best = 342: [341]	train-auc:0.749814+0.004327	test-auc:0.669340+0.005324


######## The best model ##########
xgb_model2.0 <- xgb.train(data = xgb_train2.0, max_depth = 7, eta = .01, sub_sample = .9, 
                          nround = 500, eval_metric = 'auc', objective = 'binary:logistic')

xgb.importance(xgb_train2.0$dataDimnames[[2]], model = xgb_model2.0)

'''
Feature         Gain        Cover   Frequence
1:      19 0.6397381781 0.3723904117 0.218444464
2:      22 0.1004736629 0.1442672701 0.203294283
3:       0 0.0434858351 0.0595150768 0.097037494
4:      10 0.0417441297 0.0744161788 0.086526322
5:       2 0.0349429972 0.0948028710 0.075134326
6:      21 0.0159502927 0.0376862286 0.035878916
7:       6 0.0156314667 0.0265869581 0.037728647
8:       7 0.0146344934 0.0140313226 0.032678587
9:      12 0.0143101567 0.0368179831 0.029625062
10:       9 0.0109977273 0.0086454978 0.026747703
11:      20 0.0102724780 0.0210189223 0.023929064
12:       4 0.0095282294 0.0124465617 0.024134590
13:      17 0.0093335483 0.0106131386 0.019994715
14:      13 0.0081418042 0.0282810261 0.018056901
15:       8 0.0065272030 0.0081164054 0.012272821
16:       5 0.0063790703 0.0037814781 0.014210634
17:      15 0.0052273054 0.0025402028 0.010129481
18:      14 0.0037906672 0.0048864192 0.008191667
19:       3 0.0031432775 0.0106455136 0.008485276
20:       1 0.0029131909 0.0034315029 0.008367832
21:      11 0.0020538813 0.0249092435 0.007105317
22:      16 0.0007804047 0.0001697871 0.002025896
Feature         Gain        Cover   Frequence
'''

xgb_datatest2.0 <- sapply(test_1[, -1], as.numeric)
xgb_test2.0 <- xgb.DMatrix(data = xgb_datatest2.0, missing = NA)
xgb_pred2.0 <- predict(xgb_model2.0, xgb_test2.0)
xgb_pred2.0_id <- cbind(test_1[, 1], xgb_pred2.0)

############# model 2: product_type = 2 ##############
xgb_data2.1 <- sapply(train_2[, -c(1, 2, 10, 11)], as.numeric)
xgb_train2.1<- xgb.DMatrix(data = xgb_data2.1, label = train_2[, 2], missing = NA)

xgb_cv_model2.1 <- xgb.cv(data = xgb_train2.1, max_depth = 8, eta = .01, sub_sample = .7, 
                          nround = 1000, eval_metric = 'auc', objective = 'binary:logistic', 
                          early.stop.round = 200, prediction = T, nfold = 5)
# set.seed = 105
### max_depth = 12, early.stop.round = 200
# nround = 3000, best = 280: [279]	train-auc:0.986048+0.002023	test-auc:0.666603+0.011616
### max_depth = 11, early.stop.round = 200
# nround = 1000, best = 208: [207]	train-auc:0.965220+0.006570	test-auc:0.662185+0.020415
### max_depth = 10, early.stop.round = 200
# nround = 1000, best = 367: [366]	train-auc:0.965944+0.005171	test-auc:0.665819+0.003264
### max_depth = 9, early.stop.round = 200
# nround = 1000, best = 237: [236]	train-auc:0.924071+0.003180	test-auc:0.672658+0.015114
# nround = 1000, sub_sample = .8, best = 205: [204]	train-auc:0.921374+0.005797	test-auc:0.669313+0.013268

### max_depth = 8, early.stop.round = 200
# nround = 1000, best = 121: [122]	train-auc:0.871667+0.007527	test-auc:0.673707+0.019273
# nround = 1000, sub_sample = .8, best = 246: [245]	train-auc:0.896504+0.005707	test-auc:0.674522+0.012812
# nround = 1000, sub_sample = .7, best = 126: [125]	train-auc:0.869131+0.010227	test-auc:0.677087+0.006542 ***
# nround = 1000, sub_sample = .6, best = 246: [245]	train-auc:0.896398+0.007088	test-auc:0.676580+0.017664

### max_depth = 7, early.stop.round = 200
# nround = 1000, best = 186: [185]	train-auc:0.855022+0.001725	test-auc:0.680919+0.018301


xgb_model2.1 <- xgb.train(data = xgb_train2.1, max_depth = 8, eta = .01, sub_sample = .9, 
                          nround = 1000, eval_metric = 'auc', objective = 'binary:logistic')
xgb_model2.1

xgb_datatest2.1 <- sapply(test_2[, -c(1, 9, 10)], as.numeric)
xgb_test2.1 <- xgb.DMatrix(data = xgb_datatest2.1, missing = NA)
xgb_pred2.1 <- predict(xgb_model2.1, xgb_test2.1)
xgb_pred2.1_id <- cbind(test_2[, 1], xgb_pred2.1)
submit_xgb_2mt <- rbind(xgb_pred2.0_id, xgb_pred2.1_id)
colnames(submit_xgb_2mt) <- c('user_id', 'probability')
submit_xgb2 <- left_join(test_index, as.data.frame(submit_xgb_2mt), by = 'user_id')

write.table(submit_xgb2, file = 'submit_xgb2.txt', row.names = F, col.names = T, quote = F, sep = ',')


############### Model with new dataset #################
#################### consump_clean 0 ###################
#### train, test data and consumption #######
consump_clean <- read.csv('./data/data_clean/consump_clean1.csv')
train_consump <- left_join(train, consump_clean, by = 'user_id')
test_consump <- left_join(test, consump_clean, by = 'user_id')
train_consump <- train_consump[, -27] # reomve column 'X'
test_consump <- test_consump[, -26] # reomve column 'X'

# split train_consumption dataset into two datasets by product_id(1, 2)
index1 <- which(train_consump$product_id == 1) # index of product 1
train_consump_1 <- train_consump[index1, -c(6)] # type = 1
train_consump_2 <- train_consump[-index1, -c(10:13, 15:21)] # type = 2

# split test_consumption dataset into two datasets by product_id(1, 2)
index2 <- which(test_consump$product_id == 1) # index of product 1
test_consump_1 <- test_consump[index2, -c(5)] # type = 1
test_consump_2 <- test_consump[-index2, -c(9:12, 14:20)] # type = 2

############### Model with new dataset #################
#################### consump_clean 1 ###################
################# tag ###################
tag <- read.table('./data/rong_tag.txt', sep = ',', header = T)
tag_df <- tag %>% 
  group_by(user_id) %>% 
  summarise(num_tag = n())


#### train, test data and consumption #######
consump_clean <- read.csv('./data/data_clean/consump_clean1.csv')
train_consump <- left_join(train, consump_clean, by = 'user_id')
train_consump <- left_join(train_consump, tag_df, by = 'user_id')
train_consump <- left_join(train_consump, tim_df, by = 'user_id')

test_consump <- left_join(test, consump_clean, by = 'user_id')
test_consump <- left_join(test_consump, tag_df, by = 'user_id')
test_consump <- left_join(test_consump, tim_df, by = 'user_id')

train_consump <- train_consump[, -27] # reomve column 'X'
test_consump <- test_consump[, -26] # reomve column 'X'

# split train_consumption dataset into two datasets by product_id(1, 2)
index1 <- which(train_consump$product_id == 1) # index of product 1
train_consump_1 <- train_consump[index1, -c(6)] # type = 1
train_consump_2 <- train_consump[-index1, -c(10:13, 15:21)] # type = 2

# split test_consumption dataset into two datasets by product_id(1, 2)
index2 <- which(test_consump$product_id == 1) # index of product 1
test_consump_1 <- test_consump[index2, -c(5)] # type = 1
test_consump_2 <- test_consump[-index2, -c(9:12, 14:20)] # type = 2


################## XGBoost ###########################
######################################################
library(dplyr)
library(xgboost)
set.seed(123)
# 123 -> xgb4
# 1234 -> xgb4.1
# 12345 -> xgb4.1
# 12345, new features in consump-> xgb4.2
[1] "user_id"           "lable"             "age"               "sex"               "expect_quota"     
[6] "occupation"        "education"         "marital_status"    "live_info"         "local_hk"         
[11] "money_function"    "company_type"      "salary"            "school_type"       "flow"             
[16] "gross_profit"      "business_type"     "business_year"     "personnel_num"     "pay_type"         
[21] "product_id"        "tm_encode"         "nrows_unique"      "nrows"             "num_rel"          
[26] "sum_repay"         "sum_avlb_bal"      "mean_borrow_ratio" "num_card_type"     "num_curr"         
[31] "num_cheat"         "num_default"       "sum_cost_cnt"      "num_tag"           "tim_df"    
########## model 1: product_type = 1 ##############

xgb_cv_model3.0

# seed = 123
# max_depth = 3, eta = .01, sub_sample = .9: [633]	train-auc:0.708987+0.001534	test-auc:0.674949+0.008381
# max_depth = 2, eta = .01, sub_sample = .8: [575]	train-auc:0.691247+0.002550	test-auc:0.676329+0.011892
# max_depth = 2, eta = .02, sub_sample = .8: [518]	train-auc:0.699757+0.001341	test-auc:0.675679+0.004689
# max_depth = 3, eta = .02, sub_sample = .8: [302]	train-auc:0.693412+0.002124	test-auc:0.676223+0.008481
# max_depth = 4, eta = .02, sub_sample = .8: [211]	train-auc:0.717676+0.002326	test-auc:0.675729+0.004897
# max_depth = 4, eta = .01, sub_sample = .8: [411]	train-auc:0.716614+0.001870	test-auc:0.677642+0.009977 ***
# max_depth = 4, eta = .01, sub_sample = .9: [399]	train-auc:0.715188+0.000929	test-auc:0.674744+0.004802


# seed = 12
# max_depth = 3, eta = .01, sub_sample = .9: [604]	train-auc:0.707727+0.001272	test-auc:0.677038+0.006544


# seed = 12345
# eta = .01, sub_sample = .9
# max_depth = 2: [418]	train-auc:0.687392+0.002079	test-auc:0.675200+0.006374
# max_depth = 3: [479]	train-auc:0.702627+0.004125	test-auc:0.677404+0.015423 **
# max_depth = 4: [370]	train-auc:0.713503+0.000988	test-auc:0.677249+0.005258
# max_depth = 5:  [279]	train-auc:0.728830+0.001996	test-auc:0.674728+0.008026
# max_depth = 9: [252]	train-auc:0.854875+0.005051	test-auc:0.670938+0.008816
# max_depth = 16: [224]	train-auc:0.988057+0.003721	test-auc:0.661326+0.005884

#max_depth = 4, eta = .01, sub_sample = .8: [444]	train-auc:0.719079+0.002977	test-auc:0.676594+0.007766
# max_depth = 3, eta = 0.01: [59]	train-auc:0.693280+0.001172	test-auc:0.675633+0.002445
# max_depth = 3, eta = .01, sub_sample = .8: [420]	train-auc:0.700115+0.000901	test-auc:0.675741+0.004682
#max_depth = 4, eta = .02, sub_sample = .8: [211]	train-auc:0.717675+0.001863	test-auc:0.674588+0.006357
# max_depth = 5, eta = 0.05: [47]	train-auc:0.724591+0.001849	test-auc:0.675935+0.008402
# max_depth = 5, eta = 0.02, sub_sample =.8: [110]	train-auc:0.722431+0.003103	test-auc:0.676519+0.010142

# nfold = 10
# max_depth = 3, eta = .01, sub_sample = .9: [665]	train-auc:0.707453+0.001465	test-auc:0.677815+0.011877 ***

######### add new feature in consump_clean1 #########
xgb_model3.0 <- xgb.train(data = xgb_train3.0, max_depth = 4, eta = .01, sub_sample = .8, 
                          nround = 450, eval_metric = 'auc', objective = 'binary:logistic')
xgb_model3.0

xgb_datatest3.0 <- sapply(test_consump_1[, -c(1, 33:34)], as.numeric)
xgb_test3.0 <- xgb.DMatrix(data = xgb_datatest3.0, missing = NA)
xgb_pred3.0 <- predict(xgb_model3.0, xgb_test3.0)

xgb_pred3.0_id <- cbind(test_consump_1[, 1], xgb_pred3.0)


########## model 2: product_type = 2 ##############
'''
[1] "user_id"           "lable"             "age"               "sex"              
[5] "expect_quota"      "max_month_repay"   "occupation"        "education"        
[9] "marital_status"    "salary"            "product_id"        "tm_encode"        
[13] "nrows_unique"      "nrows"             "num_rel"           "sum_repay"        
[17] "sum_avlb_bal"      "mean_borrow_ratio" "num_card_type"     "num_curr"         
[21] "num_cheat"         "num_default"       "sum_cost_cnt"   + tag   "tim_std"/"tim_df"
'''
xgb_data3.1 <- sapply(train_consump_2[, -c(1, 2, 7:9, 11, 24)], as.numeric)
xgb_train3.1 <- xgb.DMatrix(data = xgb_data3.1, label = as.integer(train_consump_2[, 2])-1, missing = NA)

xgb_cv_model3.1 <- xgb.cv(data = xgb_train3.1, max_depth = 3, eta = .01, sub_sample = .9, 
                          nround = 1000, eval_metric = 'auc', objective = 'binary:logistic', 
                          early.stop.round = 200, prediction = T, nfold = 5)

xgb_cv_model3.1

# max_depth = 2, eta = .01, sub_sample = .8: [208]	train-auc:0.715604+0.005504	test-auc:0.700564+0.021682
# max_depth = 3, eta = .02, sub_sample = .8: [94]	train-auc:0.735014+0.004588	test-auc:0.698584+0.027916
# max_depth = 2, eta = .02, sub_sample = .8: [165]	train-auc:0.723873+0.005269	test-auc:0.698925+0.020390
# max_depth = 2, eta = .02, sub_sample = .7: [117]	train-auc:0.718689+0.000811	test-auc:0.699406+0.005984
# max_depth = 2, eta = .01, sub_sample = .7: [259]	train-auc:0.719734+0.004740	test-auc:0.701588+0.018111

# eta = .01, sub_sample = .9
# max_depth = 3: [278]	train-auc:0.742014+0.005134	test-auc:0.697119+0.018301 **
# max_depth = 5: [150]	train-auc:0.786969+0.003413	test-auc:0.686619+0.011882
# max_depth = 5, eta = 0.05: [61]	train-auc:0.811612+0.004695	test-auc:0.691387+0.012516

# max_depth = 2, eta = .01, sub_sample = .9: [226]	train-auc:0.718392+0.004739	test-auc:0.695699+0.021300
# max_depth = 3, eta = .01, sub_sample = .8: [300]	train-auc:0.745071+0.003663	test-auc:0.695266+0.018594
# max_depth = 3, eta = .05, sub_sample = .8: [27]	train-auc:0.729421+0.001955	test-auc:0.695618+0.005068
# max_depth = 4, eta = .01, sub_sample = .8: [280]	train-auc:0.771827+0.005557	test-auc:0.695483+0.024633
# max_depth = 4, eta = .04, sub_sample = .8: [34]	train-auc:0.751112+0.002696	test-auc:0.696656+0.020541

#max_depth = 3, eta = .001, sub_sample = .9: [727]	train-auc:0.721086+0.006951	test-auc:0.692666+0.023428

######### add new feature in consump_clean1 #########


xgb_model3.1 <- xgb.train(data = xgb_train3.1, max_depth = 3, eta = .01, sub_sample = .9, 
                          nround = 120, eval_metric = 'auc', objective = 'binary:logistic')
xgb_model3.1

xgb_datatest3.1 <- sapply(test_consump_2[, -c(1, 6:8, 10, 23)], as.numeric)
xgb_test3.1 <- xgb.DMatrix(data = xgb_datatest3.1, missing = NA)
xgb_pred3.1 <- predict(xgb_model3.1, xgb_test3.1)

xgb_pred3.1_id <- cbind(test_consump_2[, 1], xgb_pred3.1)

############# combine predictive results ###############
submit_xgb_3mt <- rbind(xgb_pred3.0_id, xgb_pred3.1_id)
colnames(submit_xgb_3mt) <- c('user_id', 'probability')
submit_xgb3 <- left_join(test_index, as.data.frame(submit_xgb_3mt), by = 'user_id')

write.table(submit_xgb3, file = './Submit/submit_xgb_back_8.txt', row.names = F, col.names = T, quote = F, sep = ',')

################ XGboost2 -> user_info + rela1 + consump_clean1  ##################
############################################################################
library(dplyr)
library(xgboost)
set.seed(123)
# 123 -> xgb4
# 1234 -> xgb4.1
# 12345 -> xgb4.1
# 12345, new features in consump-> xgb4.2
########## model 1: product_type = 1 ##############
[1] "user_id"           "lable"             "age"               "sex"               "expect_quota"     
[6] "occupation"        "education"         "marital_status"    "live_info"         "local_hk"         
[11] "money_function"    "company_type"      "salary"            "school_type"       "flow"             
[16] "gross_profit"      "business_type"     "business_year"     "personnel_num"     "pay_type"         
[21] "product_id"        "tm_encode"         "nrows_unique"      "nrows"             "num_rel"          
[26] "sum_repay"         "sum_avlb_bal"      "mean_borrow_ratio" "num_card_type"     "num_curr"         
[31] "num_cheat"         "num_default"       "sum_cost_cnt"      "num_tag"           "tim_df" 

xgb_data3.0 <- sapply(train_consump_1[, -c(1, 2, 21, 34, 35)], as.numeric)
xgb_train3.0 <- xgb.DMatrix(data = xgb_data3.0, label = as.integer(train_consump_1[, 2])-1, missing = NA)

xgb_cv_model3.0 <- xgb.cv(data = xgb_train3.0, max_depth = 4, eta = .01, sub_sample = .8, 
                          nround = 1000, eval_metric = 'auc', objective = 'binary:logistic', 
                          early.stop.round = 200, prediction = T, nfold = 5)

xgb_cv_model3.0

# seed = 123
# max_depth = 3, eta = .01, sub_sample = .9: [633]	train-auc:0.708987+0.001534	test-auc:0.674949+0.008381
# max_depth = 2, eta = .01, sub_sample = .8: [575]	train-auc:0.691247+0.002550	test-auc:0.676329+0.011892
# max_depth = 2, eta = .02, sub_sample = .8: [518]	train-auc:0.699757+0.001341	test-auc:0.675679+0.004689
# max_depth = 3, eta = .02, sub_sample = .8: [302]	train-auc:0.693412+0.002124	test-auc:0.676223+0.008481
# max_depth = 4, eta = .02, sub_sample = .8: [211]	train-auc:0.717676+0.002326	test-auc:0.675729+0.004897
# max_depth = 4, eta = .01, sub_sample = .8: [411]	train-auc:0.716614+0.001870	test-auc:0.677642+0.009977 ***
# max_depth = 4, eta = .01, sub_sample = .9: [399]	train-auc:0.715188+0.000929	test-auc:0.674744+0.004802

## another one: every time xgboost will give you different results even though you set a fixed seed
# max_depth = 4, eta = .01, sub_sample = .8: [373]	train-auc:0.713698+0.001462	test-auc:0.676305+0.003393

# seed = 12
# max_depth = 3, eta = .01, sub_sample = .9: [604]	train-auc:0.707727+0.001272	test-auc:0.677038+0.006544


# seed = 12345
# eta = .01, sub_sample = .9
# max_depth = 2: [418]	train-auc:0.687392+0.002079	test-auc:0.675200+0.006374
# max_depth = 3: [479]	train-auc:0.702627+0.004125	test-auc:0.677404+0.015423 **
# max_depth = 4: [370]	train-auc:0.713503+0.000988	test-auc:0.677249+0.005258
# max_depth = 5:  [279]	train-auc:0.728830+0.001996	test-auc:0.674728+0.008026
# max_depth = 9: [252]	train-auc:0.854875+0.005051	test-auc:0.670938+0.008816
# max_depth = 16: [224]	train-auc:0.988057+0.003721	test-auc:0.661326+0.005884

#max_depth = 4, eta = .01, sub_sample = .8: [444]	train-auc:0.719079+0.002977	test-auc:0.676594+0.007766
# max_depth = 3, eta = 0.01: [59]	train-auc:0.693280+0.001172	test-auc:0.675633+0.002445
# max_depth = 3, eta = .01, sub_sample = .8: [420]	train-auc:0.700115+0.000901	test-auc:0.675741+0.004682
#max_depth = 4, eta = .02, sub_sample = .8: [211]	train-auc:0.717675+0.001863	test-auc:0.674588+0.006357
# max_depth = 5, eta = 0.05: [47]	train-auc:0.724591+0.001849	test-auc:0.675935+0.008402
# max_depth = 5, eta = 0.02, sub_sample =.8: [110]	train-auc:0.722431+0.003103	test-auc:0.676519+0.010142

# nfold = 10
# max_depth = 3, eta = .01, sub_sample = .9: [665]	train-auc:0.707453+0.001465	test-auc:0.677815+0.011877 ***

######### add new feature in consump_clean1 #########
xgb_model3.0 <- xgb.train(data = xgb_train3.0, max_depth = 4, eta = .01, sub_sample = .8, 
                          nround = 500, eval_metric = 'auc', objective = 'binary:logistic')
xgb_model3.0

# variable importance
xgb_model3.0_impt <- xgb.importance(feature_names = colnames(train_consump_1[, -c(1, 2, 21)]), model = xgb_model3.0)
print(xgb_model3.0_impt)
xgb.plot.importance(importance_matrix = xgb_model3.0_impt)

# prediction
xgb_datatest3.0 <- sapply(test_consump_1[, -c(1, 20, 33, 34)], as.numeric)
xgb_test3.0 <- xgb.DMatrix(data = xgb_datatest3.0, missing = NA)
xgb_pred3.0 <- predict(xgb_model3.0, xgb_test3.0)

xgb_pred3.0_id <- cbind(test_consump_1[, 1], xgb_pred3.0)


########## model 2: product_type = 2 ##############
xgb_data3.1 <- sapply(train_consump_2[, -c(1, 2, 11, 24, 25)], as.numeric)
xgb_train3.1 <- xgb.DMatrix(data = xgb_data3.1, label = as.integer(train_consump_2[, 2])-1, missing = NA)

xgb_cv_model3.1 <- xgb.cv(data = xgb_train3.1, max_depth = 2, eta = .01, sub_sample = .8, 
                          nround = 2000, eval_metric = 'auc', objective = 'binary:logistic', 
                          early.stop.round = 200, prediction = T, nfold = 5)

xgb_cv_model3.1

# max_depth = 2, eta = .01, sub_sample = .8: [208]	train-auc:0.715604+0.005504	test-auc:0.700564+0.021682
# max_depth = 3, eta = .02, sub_sample = .8: [94]	train-auc:0.735014+0.004588	test-auc:0.698584+0.027916
# max_depth = 2, eta = .02, sub_sample = .8: [165]	train-auc:0.723873+0.005269	test-auc:0.698925+0.020390
# max_depth = 2, eta = .02, sub_sample = .7: [117]	train-auc:0.718689+0.000811	test-auc:0.699406+0.005984
# max_depth = 2, eta = .01, sub_sample = .7: [259]	train-auc:0.719734+0.004740	test-auc:0.701588+0.018111

# eta = .01, sub_sample = .9
# max_depth = 3: [278]	train-auc:0.742014+0.005134	test-auc:0.697119+0.018301 **
# max_depth = 5: [150]	train-auc:0.786969+0.003413	test-auc:0.686619+0.011882
# max_depth = 5, eta = 0.05: [61]	train-auc:0.811612+0.004695	test-auc:0.691387+0.012516

# max_depth = 2, eta = .01, sub_sample = .9: [226]	train-auc:0.718392+0.004739	test-auc:0.695699+0.021300
# max_depth = 3, eta = .01, sub_sample = .8: [300]	train-auc:0.745071+0.003663	test-auc:0.695266+0.018594
# max_depth = 3, eta = .05, sub_sample = .8: [27]	train-auc:0.729421+0.001955	test-auc:0.695618+0.005068
# max_depth = 4, eta = .01, sub_sample = .8: [280]	train-auc:0.771827+0.005557	test-auc:0.695483+0.024633
# max_depth = 4, eta = .04, sub_sample = .8: [34]	train-auc:0.751112+0.002696	test-auc:0.696656+0.020541

#max_depth = 3, eta = .001, sub_sample = .9: [727]	train-auc:0.721086+0.006951	test-auc:0.692666+0.023428

## another one: every time xgboost will give you different results even though you set a fixed seed
# max_depth = 2, eta = .01, sub_sample = .7: [231]	train-auc:0.718129+0.002254	test-auc:0.695961+0.007013
# max_depth = 2, eta = .01, sub_sample = .8: [120]	train-auc:0.710392+0.002414	test-auc:0.695349+0.013091

######### add new feature in consump_clean1 #########


xgb_model3.1 <- xgb.train(data = xgb_train3.1, max_depth = 2, eta = .01, sub_sample = .8, 
                          nround = 200, eval_metric = 'auc', objective = 'binary:logistic')
xgb_model3.1

xgb_datatest3.1 <- sapply(test_consump_2[, -c(1, 10, 23, 24)], as.numeric)
xgb_test3.1 <- xgb.DMatrix(data = xgb_datatest3.1, missing = NA)
xgb_pred3.1 <- predict(xgb_model3.1, xgb_test3.1)

xgb_pred3.1_id <- cbind(test_consump_2[, 1], xgb_pred3.1)

############# combine predictive results ###############
submit_xgb_3mt <- rbind(xgb_pred3.0_id, xgb_pred3.1_id)
colnames(submit_xgb_3mt) <- c('user_id', 'probability')
submit_xgb3 <- left_join(test_index, as.data.frame(submit_xgb_3mt), by = 'user_id')

write.table(submit_xgb3, file = './Submit/submit_xgb_good_orig.txt', row.names = F, col.names = T, quote = F, sep = ',')

# This is the highest result unitl now





################ XGboost3 -> user_info + rela1 + consump_clean2  ##################
############################################################################
library(dplyr)
library(xgboost)
set.seed(123)
# 123 -> xgb4
# 1234 -> xgb4.1
# 12345 -> xgb4.1
# 12345, new features in consump-> xgb4.2
########## model 1: product_type = 1 ##############
xgb_data3.0 <- sapply(train_consump_1[, -c(1, 2, 6, 11, 12, 17, 20, 21)], as.numeric)
library(Matrix)
previous_na_action <- options('na.action')
options(na.action='na.pass')
# Do your stuff...
xgb_data3.0 <- sparse.model.matrix(~.-1, data = train_consump_1[, -c(1, 2, 10, 12, 17, 20, 21)])
options(na.action=previous_na_action$na.action)

library(xgboost)
# construct a DMatrix for the xgboost
xgb_train3.0 <- xgb.DMatrix(data = xgb_data3.0, label = as.integer(train_consump_1[, 2])-1, missing = NA)


xgb_cv_model3.0 <- xgb.cv(data = xgb_train3.0, max_depth = 6, eta = .05, sub_sample = .8, 
                          nround = 1000, eval_metric = 'auc', objective = 'binary:logistic', 
                          early.stop.round = 200, prediction = T, nfold = 5)

xgb_cv_model3.0
# 0.677642+0.009977 


# seed = 123
# max_depth = 3, eta = .5, sub_sample = .8: [11]	train-auc:0.684982+0.014917	test-auc:0.671502+0.005854
# max_depth = 6, eta = .5, sub_sample = .8: [2]	train-auc:0.707266+0.008670	test-auc:0.669239+0.003788

# max_depth = 8, eta = .05, sub_sample = .8: [97]	train-auc:0.799871+0.043039	test-auc:0.667961+0.003644
# max_depth = 5, eta = .05, sub_sample = .9: [60]	train-auc:0.716321+0.008268	test-auc:0.673753+0.006910 **
# max_depth = 6, eta = .05, sub_sample = .9: [71]	train-auc:0.749701+0.006214	test-auc:0.672570+0.004537

# max_depth = 5, eta = .01, sub_sample = .9: [302]	train-auc:0.713534+0.011977	test-auc:0.672892+0.007707


######### add new feature in consump_clean1 #########
xgb_model3.0 <- xgb.train(data = xgb_train3.0, max_depth = 5, eta = .05, sub_sample = .9, 
                          nround = 100, eval_metric = 'auc', objective = 'binary:logistic')
xgb_model3.0

# variable importance
xgb_model3.0_impt <- xgb.importance(feature_names = colnames(xgb_data3.0), model = xgb_model3.0)
print(xgb_model3.0_impt)
xgb.plot.importance(importance_matrix = xgb_model3.0_impt)



#xgb_datatest3.0 <- sapply(test_consump_1[, -c(1, 20)], as.numeric)
##########
library(Matrix)
previous_na_action <- options('na.action')
options(na.action='na.pass')
# Do your stuff...
xgb_datatest3.0 <- sparse.model.matrix(~., data = test_consump_1[, -c(1, 20)])
options(na.action=previous_na_action$na.action)
#########

xgb_test3.0 <- xgb.DMatrix(data = xgb_datatest3.0, missing = NA)
xgb_pred3.0 <- predict(xgb_model3.0, xgb_datatest3.0)

xgb_pred3.0_id <- cbind(test_consump_1[, 1], xgb_pred3.0)


########## model 2: product_type = 2 ##############
xgb_data3.1 <- sapply(train_consump_2[, -c(1, 2, 4, 11)], as.numeric)  # it is a matrix
xgb_data3.1 <- train_consump_2[, -c(1, 2, 11)]
xgb_data3.1$occupation <- as.numeric(xgb_data3.1$occupation)
xgb_data3.1$education <- as.numeric(xgb_data3.1$education)
xgb_data3.1$marital_status <- as.numeric(xgb_data3.1$marital_status)
# xgb_data3.1$sex <- as.numeric(xgb_data3.1$sex)

library(Matrix)
previous_na_action <- options('na.action')
options(na.action='na.pass')
# Do your stuff...
xgb_data3.1 <- sparse.model.matrix(~.-1, data = xgb_data3.1)
options(na.action=previous_na_action$na.action)

library(xgboost)
# construct a DMatrix for the xgboost
xgb_train3.1 <- xgb.DMatrix(data = xgb_data3.1, label = as.integer(train_consump_2[, 2])-1, missing = NA)



xgb_cv_model3.1 <- xgb.cv(data = xgb_train3.1, max_depth = 2, eta = .01, sub_sample = .8, 
                          nround = 2000, eval_metric = 'auc', objective = 'binary:logistic', 
                          early.stop.round = 200, prediction = T, nfold = 5, missing = NA)

xgb_cv_model3.1 # return the probability of every observation in train dataset

# max_depth = 8, eta = .05, sub_sample = .8: [50]	train-auc:0.803077+0.137409	test-auc:0.627556+0.059201
# max_depth = 9, eta = .05, sub_sample = .8: [124]	train-auc:0.869529+0.092498	test-auc:0.638082+0.042874
## max_depth = 9, eta = .05, sub_sample = .8: [66]	train-auc:0.909873+0.085330	test-auc:0.651006+0.025233

# max_depth = 10, eta = .05, sub_sample = .8: [86]	train-auc:0.959457+0.018832	test-auc:0.637344+0.024552
# max_depth = 11, eta = .05, sub_sample = .8: [97]	train-auc:0.838137+0.148142	test-auc:0.624797+0.062229
# max_depth = 12, eta = .05, sub_sample = .8: [104]	train-auc:0.996832+0.001631	test-auc:0.652164+0.016261
# max_depth = 13, eta = .05, sub_sample = .8: [94]	train-auc:0.949750+0.099718	test-auc:0.645198+0.014622


######### add new feature in consump_clean1 #########

# verbose = 2 tell you:  this round ends every time, it is useless
xgb_model3.1 <- xgb.train(data = xgb_train3.1, max_depth = 2, eta = .03, sub_sample = .9, 
                          nround = 100, eval_metric = 'auc', objective = 'binary:logistic')
# xgb.train is a more advanced interface compared with xgboost, actually they are similiar
xgb_model3.1

# variable importance
xgb_model3.1_impt <- xgb.importance(feature_names = colnames(xgb_data3.1), model = xgb_model3.1)
print(xgb_model3.1_impt)
xgb.plot.importance(importance_matrix = xgb_model3.1_impt)

# prediction
#xgb_datatest3.1 <- sapply(test_consump_2[, -c(1, 10)], as.numeric)
##########
library(Matrix)
previous_na_action <- options('na.action')
options(na.action='na.pass')
# Do your stuff...
xgb_datatest3.1 <- sparse.model.matrix(~., data = test_consump_2[, -c(1, 10)])
options(na.action=previous_na_action$na.action)
#########
xgb_test3.1 <- xgb.DMatrix(data = xgb_datatest3.1, missing = NA)
xgb_pred3.1 <- predict(xgb_model3.1, xgb_datatest3.1)

xgb_pred3.1_id <- cbind(test_consump_2[, 1], xgb_pred3.1)

############# combine predictive results ###############
submit_xgb_3mt <- rbind(xgb_pred3.0_id, xgb_pred3.1_id)
colnames(submit_xgb_3mt) <- c('user_id', 'probability')
submit_xgb3 <- left_join(test_index, as.data.frame(submit_xgb_3mt), by = 'user_id')

write.table(submit_xgb3, file = './Submit/submit_xgb4_3_3.txt', row.names = F, col.names = T, quote = F, sep = ',')



################ XGboost4 -> user_info + rela1 + consump_clean2(delete, no sparse matrix) + with tag  ##################
#######################################################################################################
library(dplyr)
library(xgboost)
set.seed(1234)
########## model 1: product_type = 1 ##############
xgb_data3.0 <- sapply(train_consump_1[, -c(1, 2, 21)], as.numeric)
'''
[1] "user_id"           "lable"             "age"               "sex"               "expect_quota"     
[6] "occupation"        "education"         "marital_status"    "live_info"         "local_hk"         
[11] "money_function"    "company_type"      "salary"            "school_type"       "flow"             
[16] "gross_profit"      "business_type"     "business_year"     "personnel_num"     "pay_type"         
[21] "product_id"        "tm_encode"         "nrows_unique"      "nrows"             "num_rel"          
[26] "sum_repay"         "sum_avlb_bal"      "mean_borrow_ratio" "num_card_type"     "num_curr"         
[31] "num_cheat"         "num_default"       "sum_cost_cnt"    
'''
# construct a DMatrix for the xgboost
xgb_train3.0 <- xgb.DMatrix(data = xgb_data3.0, label = as.integer(train_consump_1[, 2])-1, missing = NA)


xgb_cv_model3.0 <- xgb.cv(data = xgb_train3.0, max_depth = 4, eta = .02, sub_sample = .8, 
                          nround = 1000, eval_metric = 'auc', objective = 'binary:logistic', 
                          early.stop.round = 200, prediction = T, nfold = 5)

xgb_cv_model3.0
######################## with tag, no categorical removed #######################


######################## with tag in this model #######################
# max_depth = 4, eta = .05, sub_sample = .8：[47]	train-auc:0.707347+0.002130	test-auc:0.680713+0.007461 


# max_depth = 4, eta = .01, sub_sample = .7： [335]	train-auc:0.713832+0.002221	test-auc:0.680662+0.009304
'''
# max_depth = 4, eta = .02, sub_sample = .8: [107]	train-auc:0.713239+0.002222	test-auc:0.681625+0.007789
# max_depth = 4, eta = .02, sub_sample = .8: [161]	train-auc:0.713318+0.001783	test-auc:0.679402+0.003887
# max_depth = 4, eta = .02, sub_sample = .8: [141]	train-auc:0.710559+0.001886	test-auc:0.680565+0.004537
# max_depth = 4, eta = .02, sub_sample = .8: [114]	train-auc:0.707003+0.001304	test-auc:0.679176+0.005994
# max_depth = 4, eta = .02, sub_sample = .8：[169]	train-auc:0.713504+0.000812	test-auc:0.681321+0.002477
# max_depth = 4, eta = .02, sub_sample = .8: [175]	train-auc:0.715012+0.000920	test-auc:0.680097+0.007675
'''
# max_depth = 4, eta = .01, sub_sample = .8：[367]	train-auc:0.715845+0.001570	test-auc:0.681378+0.006633 ***
# max_depth = 4, eta = .02, sub_sample = .8： [179]	train-auc:0.716127+0.001422	test-auc:0.681332+0.005930
# max_depth = 4, eta = .01, sub_sample = .9： [320]	train-auc:0.713151+0.002192	test-auc:0.679471+0.010657
# max_depth = 4, eta = .03, sub_sample = .8：[87]	train-auc:0.709845+0.000998	test-auc:0.678320+0.003843

# max_depth = 5, eta = .01, sub_sample = .8： [78]	train-auc:0.708048+0.002234	test-auc:0.678789+0.009205


######################## without tag in this model #######################
#***** max_depth = 3: [479]	train-auc:0.702627+0.004125	test-auc:0.677404+0.015423 ***** benchmark from above models

# max_depth = 3, eta = .04, sub_sample = .8: [99]	train-auc:0.698581+0.001992	test-auc:0.676626+0.008313

# max_depth = 4, eta = .02, sub_sample = .8: [185]	train-auc:0.712447+0.002330	test-auc:0.677390+0.005677
# max_depth = 4, eta = .03, sub_sample = .8: [85]	train-auc:0.704858+0.001766	test-auc:0.677235+0.007277
# max_depth = 4, eta = .04, sub_sample = .8: [69]	train-auc:0.705996+0.002446	test-auc:0.677309+0.009779 ***
# max_depth = 4, eta = .04, sub_sample = .9: [72]	train-auc:0.706783+0.003216	test-auc:0.675318+0.013626

# max_depth = 4, eta = .05, sub_sample = .7: [58]	train-auc:0.706784+0.002079	test-auc:0.676855+0.010991
# max_depth = 4, eta = .05, sub_sample = .8: [72]	train-auc:0.732897+0.002655	test-auc:0.677184+0.004583
# max_depth = 4, eta = .05, sub_sample = .9: [81]	train-auc:0.715129+0.000867	test-auc:0.676401+0.006528
# max_depth = 4, eta = .02, sub_sample = .7: [368]	train-auc:0.712147+0.000773	test-auc:0.675889+0.007855


######### add new feature in consump_clean1 #########
xgb_model3.0 <- xgb.train(data = xgb_train3.0, max_depth = 4, eta = .02, sub_sample = .8, 
                          nround = 200, eval_metric = 'auc', objective = 'binary:logistic')
xgb_model3.0

# variable importance
xgb_model3.0_impt <- xgb.importance(feature_names = colnames(xgb_data3.0), model = xgb_model3.0)
print(xgb_model3.0_impt)
xgb.plot.importance(importance_matrix = xgb_model3.0_impt)

# prediction
xgb_datatest3.0 <- sapply(test_consump_1[, -c(1, 20)], as.numeric)

xgb_test3.0 <- xgb.DMatrix(data = xgb_datatest3.0, missing = NA)
xgb_pred3.0 <- predict(xgb_model3.0, xgb_test3.0)

xgb_pred3.0_id <- cbind(test_consump_1[, 1], xgb_pred3.0)


########## model 2: product_type = 2 ##############
xgb_data3.1 <- sapply(train_consump_2[, -c(1, 2, 11)], as.numeric)  # it is a matrix

'''
[1] "user_id"           "lable"             "age"               "sex"               "expect_quota"     
[6] "max_month_repay"   "occupation"        "education"         "marital_status"    "salary"           
[11] "product_id"        "tm_encode"         "nrows_unique"      "nrows"             "num_rel"          
[16] "sum_repay"         "sum_avlb_bal"      "mean_borrow_ratio" "num_card_type"     "num_curr"         
[21] "num_cheat"         "num_default"       "sum_cost_cnt" 
'''
# construct a DMatrix for the xgboost
xgb_train3.1 <- xgb.DMatrix(data = xgb_data3.1, label = as.integer(train_consump_2[, 2])-1, missing = NA)

xgb_cv_model3.1 <- xgb.cv(data = xgb_train3.1, max_depth = 4, eta = .01, sub_sample = .9, 
                          nround = 2000, eval_metric = 'auc', objective = 'binary:logistic', 
                          early.stop.round = 200, prediction = T, nfold = 5, missing = NA)

xgb_cv_model3.1 # return the probability of every observation in train dataset
######################## with tag in this model #######################
'''
# max_depth = 4, eta = .01, sub_sample = .9: [124]	train-auc:0.750869+0.003791	test-auc:0.699141+0.009990
# max_depth = 4, eta = .01, sub_sample = .9: [202]	train-auc:0.764161+0.003841	test-auc:0.700363+0.022117 *
# max_depth = 4, eta = .01, sub_sample = .9: [250]	train-auc:0.771700+0.002891	test-auc:0.700697+0.013190
# max_depth = 4, eta = .01, sub_sample = .9: [121]	train-auc:0.752104+0.002882	test-auc:0.703683+0.015363
# max_depth = 4, eta = .01, sub_sample = .9: [258]	train-auc:0.773053+0.004718	test-auc:0.696745+0.017698

# max_depth = 4, eta = .01, sub_sample = .8: [215]	train-auc:0.765096+0.003185	test-auc:0.700445+0.008534

# max_depth = 4, eta = .01, sub_sample = .85： [176]	train-auc:0.761220+0.001368	test-auc:0.702314+0.005159

'''

# max_depth = 4, eta = .01, sub_sample = .8: [113]	train-auc:0.751074+0.002180	test-auc:0.695562+0.008579
# max_depth = 4, eta = .02, sub_sample = .8: [56]	train-auc:0.749723+0.003428	test-auc:0.694890+0.013564


# max_depth = 5, eta = .01, sub_sample = .8: [85]	train-auc:0.768588+0.003147	test-auc:0.692885+0.015008
# max_depth = 5, eta = .01, sub_sample = .9: [169]	train-auc:0.788458+0.002758	test-auc:0.700137+0.028505 **
# max_depth = 5, eta = .02, sub_sample = .9: [15]	train-auc:0.754492+0.003021	test-auc:0.700122+0.013750

# max_depth = 6, eta = .01, sub_sample = .9: [169]	train-auc:0.821419+0.006055	test-auc:0.697895+0.009185



######################## without tag in this model #######################
# *** max_depth = 3, eta = .01, sub_sample = .9: [278]	train-auc:0.742014+0.005134	test-auc:0.697119+0.018301 **

# max_depth = 2, eta = .01, sub_sample = .8: [217]	train-auc:0.715796+0.004062	test-auc:0.694641+0.015998
# max_depth = 2, eta = .03, sub_sample = .8: [76]	train-auc:0.716273+0.004651	test-auc:0.696206+0.023033
# max_depth = 2, eta = .04, sub_sample = .8: [58]	train-auc:0.717507+0.004226	test-auc:0.697068+0.015545

# max_depth = 3, eta = .05, sub_sample = .7: [11]	train-auc:0.714037+0.004508	test-auc:0.695992+0.013382

# max_depth = 3, eta = .01, sub_sample = .8: [189]	train-auc:0.732562+0.003287	test-auc:0.696121+0.007540
# max_depth = 3, eta = .02, sub_sample = .8: [52]	train-auc:0.724358+0.004795	test-auc:0.695468+0.019398
# max_depth = 3, eta = .02, sub_sample = .9: [102]	train-auc:0.734069+0.002883	test-auc:0.697152+0.015643
# max_depth = 3, eta = .03, sub_sample = .8: [45]	train-auc:0.729270+0.002671	test-auc:0.697804+0.010746 ***
# max_depth = 3, eta = .05, sub_sample = .8: [27]	train-auc:0.728214+0.004395	test-auc:0.697237+0.014060


# max_depth = 4, eta = .01, sub_sample = .8: [143]	train-auc:0.751166+0.002407	test-auc:0.691863+0.017892


######### add new feature in consump_clean1 #########

# verbose = 2 tell you:  this round ends every time, it is useless
xgb_model3.1 <- xgb.train(data = xgb_train3.1, max_depth = 4, eta = .01, sub_sample = .9, 
                          nround = 150, eval_metric = 'auc', objective = 'binary:logistic')
# xgb.train is a more advanced interface compared with xgboost, actually they are similiar
xgb_model3.1

# variable importance
xgb_model3.1_impt <- xgb.importance(feature_names = colnames(xgb_data3.1), model = xgb_model3.1)
print(xgb_model3.1_impt)
xgb.plot.importance(importance_matrix = xgb_model3.1_impt)

# prediction
xgb_datatest3.1 <- sapply(test_consump_2[, -c(1, 10)], as.numeric)

#########
xgb_test3.1 <- xgb.DMatrix(data = xgb_datatest3.1, missing = NA)
xgb_pred3.1 <- predict(xgb_model3.1, xgb_test3.1)

xgb_pred3.1_id <- cbind(test_consump_2[, 1], xgb_pred3.1)

############# combine predictive results ###############
submit_xgb_3mt <- rbind(xgb_pred3.0_id, xgb_pred3.1_id)
colnames(submit_xgb_3mt) <- c('user_id', 'probability')
submit_xgb3 <- left_join(test_index, as.data.frame(submit_xgb_3mt), by = 'user_id')

write.table(submit_xgb3, file = './Submit/submit_xgb4_5_6.txt', row.names = F, col.names = T, quote = F, sep = ',')
# submit_xgb4_5.txt -> xgb3.0_Feature Importance_4.png: with all categorical vars









################ XGboost5 -> user_info + rela1 + consump_clean2(delete, no sparse matrix) + with tag + time_dif ##################
#######################################################################################################
library(dplyr)
library(xgboost)
set.seed(123)
########## model 1: product_type = 1 ##############
xgb_data3.0 <- sapply(train_consump_1[, -c(1, 2, 21, 26:28)], as.numeric)
'''
[1] "user_id"           "lable"             "age"               "sex"              
[5] "expect_quota"      "occupation"        "education"         "marital_status"   
[9] "live_info"         "local_hk"          "money_function"    "company_type"     
[13] "salary"            "school_type"       "flow"              "gross_profit"     
[17] "business_type"     "business_year"     "personnel_num"     "pay_type"         
[21] "product_id"        "tm_encode"         "nrows_unique"      "nrows"            
[25] "num_rel"           "sum_repay"         "sum_avlb_bal"      "mean_borrow_ratio"
[29] "num_card_type"     "num_curr"          "num_cheat"         "num_default"      
[33] "sum_cost_cnt"      "num_tag"           "tim_df"  

'''
# construct a DMatrix for the xgboost
xgb_train3.0 <- xgb.DMatrix(data = xgb_data3.0, label = as.integer(train_consump_1[, 2])-1, missing = NA)

#for (i in c(1:10))
xgb_cv_model3.0 <- xgb.cv(data = xgb_train3.0, max_depth = 4, eta = .01, sub_sample = .9, 
                          nround = 1000, eval_metric = 'auc', objective = 'binary:logistic', 
                          early.stop.round = 100, prediction = T, nfold = 5)

#}

xgb_cv_model3.0
######## final ########
# [270]	train-auc:0.710880+0.002238	test-auc:0.682524+0.010536
# [366]	train-auc:0.701935+0.001214	test-auc:0.681478+0.006039
# [383]	train-auc:0.718735+0.002811	test-auc:0.680999+0.003335
# [383]	train-auc:0.719066+0.000967	test-auc:0.682295+0.007852
# [259]	train-auc:0.710817+0.001383	test-auc:0.681576+0.003361

######################## with tag, no categorical removed, removing those in consump, tim_dif ###############

# max_depth = 2, eta = .02, sub_sample = .8: [589]	train-auc:0.704765+0.001203	test-auc:0.679977+0.005845

'''
# max_depth = 3, eta = .001, sub_sample = .9: [4637]	train-auc:0.706753+0.002354	test-auc:0.682016+0.007798
[3656]	train-auc:0.701479+0.002517	test-auc:0.682836+0.009091

# max_depth = 3, eta = .01, sub_sample = .9: [578]	train-auc:0.711428+0.001676	test-auc:0.681481+0.007690
[591]	train-auc:0.711475+0.001224	test-auc:0.681060+0.007531
[590]	train-auc:0.711330+0.002854	test-auc:0.682427+0.008121
[480]	train-auc:0.706910+0.001957	test-auc:0.681540+0.007199
[385]	train-auc:0.702850+0.002039	test-auc:0.682321+0.005626
[695]	train-auc:0.714863+0.002554	test-auc:0.681676+0.011377
[232]	train-auc:0.694582+0.001863	test-auc:0.681328+0.009507

'''
# max_depth = 3, eta = .01, sub_sample = .8: [370]	train-auc:0.702174+0.001141	test-auc:0.681956+0.002537
# max_depth = 3, eta = .02, sub_sample = .8: [273]	train-auc:0.710503+0.002052	test-auc:0.681604+0.011700 
# max_depth = 4, eta = .02, sub_sample = .8: [196]	train-auc:0.719967+0.001907	test-auc:0.681676+0.003836 

'''
# max_depth = 4, eta = .01, sub_sample = .9: [382]	train-auc:0.718938+0.003327	test-auc:0.683300+0.008763
# max_depth = 4, eta = .01, sub_sample = .9: [266]	train-auc:0.710909+0.001637	test-auc:0.682132+0.003942
[363]	train-auc:0.717023+0.002022	test-auc:0.682881+0.004806
[229]	train-auc:0.708919+0.003534	test-auc:0.680851+0.016549
[220]	train-auc:0.707858+0.002374	test-auc:0.682002+0.012422
[245]	train-auc:0.709381+0.000719	test-auc:0.681376+0.001839
[263]	train-auc:0.710400+0.001268	test-auc:0.680881+0.005148
[374]	train-auc:0.718470+0.001963	test-auc:0.681525+0.003782
[313]	train-auc:0.713854+0.001719	test-auc:0.681380+0.006082
[455]	train-auc:0.723861+0.001168	test-auc:0.682692+0.009380
[435]	train-auc:0.723269+0.001439	test-auc:0.680425+0.011219

'''

# max_depth = 5, eta = .02, sub_sample = .8: [245]	train-auc:0.728947+0.002230	test-auc:0.680142+0.008430
# max_depth = 5, eta = .02, sub_sample = .8: [111]	train-auc:0.726967+0.002535	test-auc:0.680907+0.008529
# max_depth = 6, eta = .02, sub_sample = .8: [128]	train-auc:0.754999+0.000786	test-auc:0.680450+0.005355



######################## with tag, no categorical removed, removing those in consump #######################
# max_depth = 3, eta = .02, sub_sample = .8: [294]	train-auc:0.709087+0.003016	test-auc:0.681061+0.006766
#...[321]	train-auc:0.711375+0.002083	test-auc:0.680428+0.012093


# max_depth = 4, eta = .02, sub_sample = .8: [120]	train-auc:0.707715+0.001611	test-auc:0.680876+0.005212
#...[209]	train-auc:0.719677+0.002170	test-auc:0.680670+0.010608
#...[126]	train-auc:0.708017+0.003603	test-auc:0.679027+0.013162

# max_depth = 5, eta = .02, sub_sample = .8: [124]	train-auc:0.726177+0.001622	test-auc:0.682020+0.006000
#...[113]	train-auc:0.724580+0.000862	test-auc:0.680966+0.005445
#...[130]	train-auc:0.727591+0.003127	test-auc:0.679183+0.009315
#... [138]	train-auc:0.729616+0.002781	test-auc:0.680562+0.012422 # 130 ***

# max_depth = 6, eta = .02, sub_sample = .8: [115]	train-auc:0.750653+0.001892	test-auc:0.679925+0.005193

######################## with tag, no categorical removed #######################
# max_depth = 4, eta = .02, sub_sample = .8: [274]	train-auc:0.727434+0.002559	test-auc:0.681466+0.005790
# max_depth = 4, eta = .02, sub_sample = .8: [220]	train-auc:0.722316+0.002404	test-auc:0.678620+0.009767
# ..:

######################## with tag in this model #######################
# max_depth = 4, eta = .05, sub_sample = .8：[47]	train-auc:0.707347+0.002130	test-auc:0.680713+0.007461 


# max_depth = 4, eta = .01, sub_sample = .7： [335]	train-auc:0.713832+0.002221	test-auc:0.680662+0.009304
'''
# max_depth = 4, eta = .02, sub_sample = .8: [107]	train-auc:0.713239+0.002222	test-auc:0.681625+0.007789
# max_depth = 4, eta = .02, sub_sample = .8: [161]	train-auc:0.713318+0.001783	test-auc:0.679402+0.003887
# max_depth = 4, eta = .02, sub_sample = .8: [141]	train-auc:0.710559+0.001886	test-auc:0.680565+0.004537
# max_depth = 4, eta = .02, sub_sample = .8: [114]	train-auc:0.707003+0.001304	test-auc:0.679176+0.005994
# max_depth = 4, eta = .02, sub_sample = .8：[169]	train-auc:0.713504+0.000812	test-auc:0.681321+0.002477
# max_depth = 4, eta = .02, sub_sample = .8: [175]	train-auc:0.715012+0.000920	test-auc:0.680097+0.007675
'''
# max_depth = 4, eta = .01, sub_sample = .8：[367]	train-auc:0.715845+0.001570	test-auc:0.681378+0.006633 ***
# max_depth = 4, eta = .02, sub_sample = .8： [179]	train-auc:0.716127+0.001422	test-auc:0.681332+0.005930
# max_depth = 4, eta = .01, sub_sample = .9： [320]	train-auc:0.713151+0.002192	test-auc:0.679471+0.010657
# max_depth = 4, eta = .03, sub_sample = .8：[87]	train-auc:0.709845+0.000998	test-auc:0.678320+0.003843

# max_depth = 5, eta = .01, sub_sample = .8： [78]	train-auc:0.708048+0.002234	test-auc:0.678789+0.009205


######################## without tag in this model #######################
#***** max_depth = 3: [479]	train-auc:0.702627+0.004125	test-auc:0.677404+0.015423 ***** benchmark from above models

# max_depth = 3, eta = .04, sub_sample = .8: [99]	train-auc:0.698581+0.001992	test-auc:0.676626+0.008313

# max_depth = 4, eta = .02, sub_sample = .8: [185]	train-auc:0.712447+0.002330	test-auc:0.677390+0.005677
# max_depth = 4, eta = .03, sub_sample = .8: [85]	train-auc:0.704858+0.001766	test-auc:0.677235+0.007277
# max_depth = 4, eta = .04, sub_sample = .8: [69]	train-auc:0.705996+0.002446	test-auc:0.677309+0.009779 ***
# max_depth = 4, eta = .04, sub_sample = .9: [72]	train-auc:0.706783+0.003216	test-auc:0.675318+0.013626

# max_depth = 4, eta = .05, sub_sample = .7: [58]	train-auc:0.706784+0.002079	test-auc:0.676855+0.010991
# max_depth = 4, eta = .05, sub_sample = .8: [72]	train-auc:0.732897+0.002655	test-auc:0.677184+0.004583
# max_depth = 4, eta = .05, sub_sample = .9: [81]	train-auc:0.715129+0.000867	test-auc:0.676401+0.006528
# max_depth = 4, eta = .02, sub_sample = .7: [368]	train-auc:0.712147+0.000773	test-auc:0.675889+0.007855


######### add new feature in consump_clean1 #########
xgb_model3.0 <- xgb.train(data = xgb_train3.0, max_depth = 4, eta = .01, sub_sample = .9, 
                          nround = 320, objective = 'binary:logistic')
xgb_model3.0

# variable importance
xgb_model3.0_impt <- xgb.importance(feature_names = colnames(xgb_data3.0), model = xgb_model3.0)
print(xgb_model3.0_impt)
xgb.plot.importance(importance_matrix = xgb_model3.0_impt)

# prediction
xgb_datatest3.0 <- sapply(test_consump_1[, -c(1, 20, 25:27)], as.numeric)

xgb_test3.0 <- xgb.DMatrix(data = xgb_datatest3.0, missing = NA)
xgb_pred3.0 <- predict(xgb_model3.0, xgb_test3.0)

xgb_pred3.0_id <- cbind(test_consump_1[, 1], xgb_pred3.0)


########## model 2: product_type = 2 ##############
xgb_data3.1 <- sapply(train_consump_2[, -c(1, 2, 11, 16:18)], as.numeric)  # it is a matrix

'''
[1] "user_id"           "lable"             "age"               "sex"              
[5] "expect_quota"      "max_month_repay"   "occupation"        "education"        
[9] "marital_status"    "salary"            "product_id"        "tm_encode"        
[13] "nrows_unique"      "nrows"             "num_rel"           "sum_repay"        
[17] "sum_avlb_bal"      "mean_borrow_ratio" "num_card_type"     "num_curr"         
[21] "num_cheat"         "num_default"       "sum_cost_cnt"      "num_tag"          
[25] "tim_df"  
'''
# construct a DMatrix for the xgboost
xgb_train3.1 <- xgb.DMatrix(data = xgb_data3.1, label = as.integer(train_consump_2[, 2])-1, missing = NA)

# 
xgb_cv_model3.1 <- xgb.cv(data = xgb_train3.1, max_depth = 3, eta = .01, sub_sample = .9, 
                          nround = 1000, eval_metric = 'auc', objective = 'binary:logistic', 
                          early.stop.round = 200, prediction = T, nfold = 5, missing = NA)

xgb_cv_model3.1 # return the probability of every observation in train dataset
## final round
# [93]	train-auc:0.729730+0.004068	test-auc:0.700659+0.013533
# [224]	train-auc:0.741613+0.005424	test-auc:0.702176+0.020900
# [281]	train-auc:0.748901+0.002256	test-auc:0.698343+0.005494
# [190]	train-auc:0.739926+0.004023	test-auc:0.704823+0.017369
# [199]	train-auc:0.739871+0.002364	test-auc:0.700635+0.014430

######################## with tag, no categorical removed, removing those in consump, with tim_dif #######################

# max_depth = 2, eta = .01, sub_sample = .9: [241]	train-auc:0.722873+0.004861	test-auc:0.699156+0.020771

# max_depth = 3, eta = .01, sub_sample = .9: [232]	train-auc:0.742138+0.004807	test-auc:0.701816+0.027097
# max_depth = 3, eta = .01, sub_sample = .9: [217]	train-auc:0.741386+0.006574	test-auc:0.697866+0.031633
# max_depth = 3, eta = .01, sub_sample = .8: [458]	train-auc:0.761898+0.004865	test-auc:0.701195+0.018841
# max_depth = 3, eta = .01, sub_sample = .8: [99]	train-auc:0.728337+0.000795	test-auc:0.699314+0.002839
# max_depth = 3, eta = .02, sub_sample = .8: [71]	train-auc:0.734970+0.002965	test-auc:0.699016+0.012812


# max_depth = 4, eta = .01, sub_sample = .8: [164]	train-auc:0.761590+0.004453	test-auc:0.704025+0.016212

# max_depth = 4, eta = .01, sub_sample = .8: [276]	train-auc:0.777289+0.005143	test-auc:0.700988+0.025984
# max_depth = 4, eta = .01, sub_sample = .9: [186]	train-auc:0.763814+0.004117	test-auc:0.700869+0.016468
# max_depth = 4, eta = .01, sub_sample = .9: [243]	train-auc:0.773562+0.004443	test-auc:0.700071+0.011904

# max_depth = 5, eta = .01, sub_sample = .9: [101]	train-auc:0.775345+0.004898	test-auc:0.698101+0.016255


######################## with tag, no categorical removed, removing those in consump #######################
# max_depth = 3, eta = .01, sub_sample = .9: [83]	train-auc:0.725564+0.004788	test-auc:0.699417+0.022986
# max_depth = 3, eta = .01, sub_sample = .9: [242]	train-auc:0.742474+0.007294	test-auc:0.700700+0.027904

# max_depth = 4, eta = .01, sub_sample = .9: [232]	train-auc:0.766037+0.002641	test-auc:0.703775+0.018095 **
# max_depth = 4, eta = .01, sub_sample = .9: [303]	train-auc:0.777489+0.005155	test-auc:0.697036+0.019862
#... [199]	train-auc:0.761704+0.003603	test-auc:0.698382+0.009507
#... [216]	train-auc:0.765131+0.002384	test-auc:0.697241+0.014536
#... [226]	train-auc:0.766565+0.002634	test-auc:0.697615+0.014090


# max_depth = 5, eta = .01, sub_sample = .9:  [98]	train-auc:0.770121+0.006413	test-auc:0.697062+0.015674

######################## with tag in this model #######################
'''
# max_depth = 4, eta = .01, sub_sample = .9: [124]	train-auc:0.750869+0.003791	test-auc:0.699141+0.009990
# max_depth = 4, eta = .01, sub_sample = .9: [202]	train-auc:0.764161+0.003841	test-auc:0.700363+0.022117 *
# max_depth = 4, eta = .01, sub_sample = .9: [250]	train-auc:0.771700+0.002891	test-auc:0.700697+0.013190
# max_depth = 4, eta = .01, sub_sample = .9: [121]	train-auc:0.752104+0.002882	test-auc:0.703683+0.015363
# max_depth = 4, eta = .01, sub_sample = .9: [258]	train-auc:0.773053+0.004718	test-auc:0.696745+0.017698




# max_depth = 4, eta = .01, sub_sample = .8: [215]	train-auc:0.765096+0.003185	test-auc:0.700445+0.008534
# max_depth = 4, eta = .01, sub_sample = .85： [176]	train-auc:0.761220+0.001368	test-auc:0.702314+0.005159

'''

# max_depth = 4, eta = .01, sub_sample = .8: [113]	train-auc:0.751074+0.002180	test-auc:0.695562+0.008579
# max_depth = 4, eta = .02, sub_sample = .8: [56]	train-auc:0.749723+0.003428	test-auc:0.694890+0.013564


# max_depth = 5, eta = .01, sub_sample = .8: [85]	train-auc:0.768588+0.003147	test-auc:0.692885+0.015008
# max_depth = 5, eta = .01, sub_sample = .9: [169]	train-auc:0.788458+0.002758	test-auc:0.700137+0.028505 **
# max_depth = 5, eta = .02, sub_sample = .9: [15]	train-auc:0.754492+0.003021	test-auc:0.700122+0.013750

# max_depth = 6, eta = .01, sub_sample = .9: [169]	train-auc:0.821419+0.006055	test-auc:0.697895+0.009185



######################## without tag in this model #######################
# *** max_depth = 3, eta = .01, sub_sample = .9: [278]	train-auc:0.742014+0.005134	test-auc:0.697119+0.018301 **

# max_depth = 2, eta = .01, sub_sample = .8: [217]	train-auc:0.715796+0.004062	test-auc:0.694641+0.015998
# max_depth = 2, eta = .03, sub_sample = .8: [76]	train-auc:0.716273+0.004651	test-auc:0.696206+0.023033
# max_depth = 2, eta = .04, sub_sample = .8: [58]	train-auc:0.717507+0.004226	test-auc:0.697068+0.015545

# max_depth = 3, eta = .05, sub_sample = .7: [11]	train-auc:0.714037+0.004508	test-auc:0.695992+0.013382

# max_depth = 3, eta = .01, sub_sample = .8: [189]	train-auc:0.732562+0.003287	test-auc:0.696121+0.007540
# max_depth = 3, eta = .02, sub_sample = .8: [52]	train-auc:0.724358+0.004795	test-auc:0.695468+0.019398
# max_depth = 3, eta = .02, sub_sample = .9: [102]	train-auc:0.734069+0.002883	test-auc:0.697152+0.015643
# max_depth = 3, eta = .03, sub_sample = .8: [45]	train-auc:0.729270+0.002671	test-auc:0.697804+0.010746 ***
# max_depth = 3, eta = .05, sub_sample = .8: [27]	train-auc:0.728214+0.004395	test-auc:0.697237+0.014060


# max_depth = 4, eta = .01, sub_sample = .8: [143]	train-auc:0.751166+0.002407	test-auc:0.691863+0.017892


######### add new feature in consump_clean1 #########

# verbose = 2 tell you:  this round ends every time, it is useless
xgb_model3.1 <- xgb.train(data = xgb_train3.1, max_depth = 3, eta = .01, sub_sample = .9, 
                          nround = 200, objective = 'binary:logistic')
# xgb.train is a more advanced interface compared with xgboost, actually they are similiar
xgb_model3.1

# variable importance
xgb_model3.1_impt <- xgb.importance(feature_names = colnames(xgb_data3.1), model = xgb_model3.1)
print(xgb_model3.1_impt)
xgb.plot.importance(importance_matrix = xgb_model3.1_impt)

# prediction
xgb_datatest3.1 <- sapply(test_consump_2[, -c(1, 10, 15:17)], as.numeric)

#########
xgb_test3.1 <- xgb.DMatrix(data = xgb_datatest3.1, missing = NA)
xgb_pred3.1 <- predict(xgb_model3.1, xgb_test3.1)

xgb_pred3.1_id <- cbind(test_consump_2[, 1], xgb_pred3.1)

############# combine predictive results ###############
submit_xgb_3mt <- rbind(xgb_pred3.0_id, xgb_pred3.1_id)
colnames(submit_xgb_3mt) <- c('user_id', 'probability')
submit_xgb3 <- left_join(test_index, as.data.frame(submit_xgb_3mt), by = 'user_id')

write.table(submit_xgb3, file = './Submit/submit_xgb6_999_1.txt', row.names = F, col.names = T, quote = F, sep = ',')
