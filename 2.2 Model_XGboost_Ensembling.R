################ XGboost_Final -> user_info + rela1 + consump_clean1  ##################
############################################################################
library(dplyr)
library(xgboost)
set.seed(123)

###################################################
########## model 1: product_type = 1 ##############
[1] "user_id"           "lable"             "age"               "sex"               "expect_quota"     
[6] "occupation"        "education"         "marital_status"    "live_info"         "local_hk"         
[11] "money_function"    "company_type"      "salary"            "school_type"       "flow"             
[16] "gross_profit"      "business_type"     "business_year"     "personnel_num"     "pay_type"         
[21] "product_id"        "tm_encode"         "nrows_unique"      "nrows"             "num_rel"          
[26] "sum_repay"         "sum_avlb_bal"      "mean_borrow_ratio" "num_card_type"     "num_curr"         
[31] "num_cheat"         "num_default"       "sum_cost_cnt"      "num_tag"           "tim_df" 

########## select parameters with cross validation #########
xgb_data3.0 <- sapply(train_consump_1[, -c(1, 2, 21, 34, 35)], as.numeric)
xgb_train3.0 <- xgb.DMatrix(data = xgb_data3.0, label = as.integer(train_consump_1[, 2])-1, missing = NA)

xgb_cv_model3.0 <- xgb.cv(data = xgb_train3.0, max_depth = 4, eta = .01, sub_sample = .8, 
                          nround = 1000, eval_metric = 'auc', objective = 'binary:logistic', 
                          early.stop.round = 200, prediction = T, nfold = 5)


# seed = 123
# max_depth = 4, eta = .01, sub_sample = .8: [411]	train-auc:0.716614+0.001870	test-auc:0.677642+0.009977 ***

######### train the model1 with selected parameters #########
xgb_model3.0 <- xgb.train(data = xgb_train3.0, max_depth = 4, eta = .01, sub_sample = .8, 
                          nround = 500, eval_metric = 'auc', objective = 'binary:logistic')

# variable importance
xgb_model3.0_impt <- xgb.importance(feature_names = colnames(train_consump_1[, -c(1, 2, 21)]), model = xgb_model3.0)
print(xgb_model3.0_impt)
xgb.plot.importance(importance_matrix = xgb_model3.0_impt)

# auc
library(pROC)
labels_train1 <-  train_consump_1[, 2]
roc1 = roc(labels, xgb_pred3.0)
plot(roc1, print.thres = TRUE)
auc1 <- auc(roc1)

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

# max_depth = 2, eta = .01, sub_sample = .7: [259]	train-auc:0.719734+0.004740	test-auc:0.701588+0.018111

######### train the model2 with selected parameters #########
xgb_model3.1 <- xgb.train(data = xgb_train3.1, max_depth = 2, eta = .01, sub_sample = .8, 
                          nround = 200, eval_metric = 'auc', objective = 'binary:logistic')
xgb_model3.1

# variable importance
xgb_model3.1_impt <- xgb.importance(feature_names = colnames(train_consump_2[, -c(1, 2, 11, 24, 25)]), model = xgb_model3.1)
print(xgb_model3.1_impt)
xgb.plot.importance(importance_matrix = xgb_model3.1_impt)

# auc
library(pROC)
labels_train2 <-  train_consump_2[, 2]
roc2 = roc(labels, xgb_pred3.1)
plot(roc2, print.thres = TRUE)
auc2 <- auc(roc2)

# prediction
xgb_datatest3.1 <- sapply(test_consump_2[, -c(1, 10, 23, 24)], as.numeric)
xgb_test3.1 <- xgb.DMatrix(data = xgb_datatest3.1, missing = NA)
xgb_pred3.1 <- predict(xgb_model3.1, xgb_test3.1)

xgb_pred3.1_id <- cbind(test_consump_2[, 1], xgb_pred3.1)

############# combine predictive results ###############
submit_xgb_3mt <- rbind(xgb_pred3.0_id, xgb_pred3.1_id)
colnames(submit_xgb_3mt) <- c('user_id', 'probability')
submit_xgb3 <- left_join(test_index, as.data.frame(submit_xgb_3mt), by = 'user_id')

write.table(submit_xgb3, file = './Submit/submit_xgb.txt', row.names = F, col.names = T, quote = F, sep = ',')

########## Ensembling models ##############
# for loop 100*40
for (i in c(451:550)) {
  for (j in c(181:220)) {
    # model 1
    xgb_model3.0 <- xgb.train(data = xgb_train3.0, max_depth = 4, eta = .01, sub_sample = .8, 
                              nround = i, eval_metric = 'auc', objective = 'binary:logistic')
    # i = 500
    xgb_pred3.0 <- predict(xgb_model3.0, xgb_test3.0)
    xgb_pred3.0_id <- cbind(test_consump_1[, 1], xgb_pred3.0)
    
    # model 2
    xgb_model3.1 <- xgb.train(data = xgb_train3.1, max_depth = 2, eta = .01, sub_sample = .8, 
                              nround = j, eval_metric = 'auc', objective = 'binary:logistic')
    
    xgb_pred3.1 <- predict(xgb_model3.1, xgb_test3.1)
    xgb_pred3.1_id <- cbind(test_consump_2[, 1], xgb_pred3.1)
    # j = 200
    # save the result
    submit_xgb_3mt <- rbind(xgb_pred3.0_id, xgb_pred3.1_id)
    colnames(submit_xgb_3mt) <- c('user_id', 'probability')
    submit_xgb <- left_join(test_index, as.data.frame(submit_xgb_3mt), by = 'user_id')
    submit_sum[, 2] <- submit_sum[, 2] + as.numeric(as.character(submit_xgb[, 2]))
    print(paste0('Round_', i, '_', j))
  }
}

submit_sum$probability <- submit_sum$probability/4000

colnames(submit_sum) <- c('user_id', 'probability')
write.table(submit_sum, file = paste0('./Submit/submit_ensembling.txt'), row.names = F, col.names = T, quote = F, sep = ',')


