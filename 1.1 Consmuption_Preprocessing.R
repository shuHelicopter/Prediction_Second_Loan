library(dplyr)
library(ggplot2)
# read data from .txt file
consump <- read.table('./data/consumption_recode.txt', sep = ',', header = T)
train_csp <- inner_join(consump, train, by = 'user_id')
temp$curr <- as.factor(temp$curr)
temp <- select(consump, avlb_bal_usd, pre_borrow_cash_amt_usd, credit_lmt_amt_usd, curr)
temp1 <- filter(temp, avlb_bal_usd != 0 | pre_borrow_cash_amt_usd != 0 | credit_lmt_amt_usd != 0)
temp2 <- filter(temp, avlb_bal_usd == 0 & pre_borrow_cash_amt_usd == 0 & credit_lmt_amt_usd == 0)

# explore the relation beweent curr and anothor three

ggplot(temp1, aes(x = curr, y = credit_lmt_amt_usd, group = curr, color = curr)) + 
  geom_boxplot() + 
  theme_bw() +
  title('temp1 without 0') + 
  scale_color_brewer(palette = 'Set1')

temp$curr <- as.numeric(temp$curr)
temp1$curr <- as.numeric(temp1$curr)

# correlation map between dollars credit
library(corrplot)
M <- cor(temp1)
corrplot(M, method = 'number', title = 'temp1 without 0')

# correlation between memeber points 
temp2 <- select(consump, curt_jifen, prior_period_jifen_bal, 
                nadd_jifen, current_adj_jifen, current_convert_jifen, 
                current_award_jifen, curr, is_cheat_bill)
temp2$curt_jifen_cal = temp2$prior_period_jifen_bal + temp2$nadd_jifen + 
  temp2$current_adj_jifen + temp2$current_award_jifen - temp2$current_convert_jifen

M2 <- cor(temp2)
corrplot(M2, method = 'number', title = 'temp2')

temp2.1 <- temp2[temp2$curt_jifen != temp2$curt_jifen_cal, ]

# check the percentage of missingness in every column
NUM = nrow(consump) # the number of rows in one consmuption
for (i in c(1:28)) {
  pect = sum(consump[, i]== 0)/NUM
  msg = paste(colnames(consump)[i], pect, sep = ': ')
  print(msg)
}
'''
[1] "user_id: 0"
[1] "bill_id: 0"
[1] "prior_period_bill_amt: 0.245517607816513"
[1] "prior_period_repay_amt: 0.280178292056558"
[1] "credit_lmt_amt: 0.182992886028869"
[1] "curt_jifen: 0.224805915517903"
[1] "current_bill_bal: 0.0191073589751159"
[1] "current_bill_min_repay_amt: 0.042462437642058"
[1] "is_cheat_bill: 0.977053753283939"  #################
[1] "cost_cnt: 0.558287333589161"
[1] "current_bill_amt: 0.308263718747233"
[1] "adj_amt: 0.9818077161496"  #################
[1] "circle_interest: 0.872581692593795"  #################
[1] "prior_period_jifen_bal: 0.559866576143106"
[1] "nadd_jifen: 0.633434188387402"
[1] "current_adj_jifen: 0.951633851875904"  #################
[1] "avlb_bal_usd: 0.999847979455088" #################
[1] "avlb_bal: 0.936366856569354"  #################
[1] "card_type: 0.000807332408418691"
[1] "pre_borrow_cash_amt_usd: 0.911023703397585" #################
[1] "credit_lmt_amt_usd: 0.910648817781976" #################
[1] "pre_borrow_cash_amt: 0.474847241491277"
[1] "curr: 0.00188771142663164"
[1] "repay_stat: 0.999707766331139" #################
[1] "current_min_repay_amt_usd: 0.999220710216371" #################
[1] "current_repay_amt_usd: 0.998924048764649" #################
[1] "current_convert_jifen: 0.98256043923606" #################
[1] "current_award_jifen: 0.951875904005668" #################
'''
# check correlation between all variables
consump_reorder <- select(consump, user_id, bill_id, prior_period_bill_amt, prior_period_repay_amt, adj_amt, 
                          circle_interest, current_bill_amt, current_bill_min_repay_amt, 
                          card_type, current_bill_bal, credit_lmt_amt, pre_borrow_cash_amt,
                          avlb_bal, repay_stat, cost_cnt, is_cheat_bill,
                          curt_jifen, nadd_jifen, current_adj_jifen, current_convert_jifen, 
                          current_award_jifen, prior_period_jifen_bal, 
                          avlb_bal_usd, pre_borrow_cash_amt_usd, credit_lmt_amt_usd, 
                          current_min_repay_amt_usd, current_repay_amt_usd, curr)
M_all <- cor(consump_reorder[, -c(1, 2)])
corrplot(M_all, method = 'circle')

# cheat consuption record
temp_cheat <- consump_reorder[consump_reorder$is_cheat_bill == 1, ]
consump_reorder_unique <- consump_reorder[!duplicated(consump_reorder[, -c(2)]), ]

############# Transform data with PCA ############
##################################################
library(psych)
consump_pca <- consump_reorder[, -c(9, 14, 16, 28)] # reomove categorical vars 'card_type', 'repay_stat', 'is_cheat_bill', 'curr'
cor(consump_pca[,-c(1, 2)]) # remove 'user_id' & 'bill_id'
# choosing K
fa.parallel(consump_pca[, -c(1, 2)], fa = 'pc', n.iter = 200) 
# performing PCA, k = 5
pc_consump = principal(consump_pca[, -c(1, 2)], #The data in question.
                      nfactors = 5, #The number of PCs to extract.
                      rotate = "none")
new_consump <- predict(pc_consump, consump_pca[, -c(1, 2)])
new_consump <- as.data.frame(new_consump)
new_consump$user_id <- consump_pca$user_id
new_consump[, c(7, 8, 9, 10)] <- consump_reorder[, c(9, 14, 16, 28)]
new_consump_unique <- new_consump %>% 
  group_by(user_id) %>%
  summarise(PC1_avg = mean(PC1), PC2_avg = mean(PC2), PC3_avg = mean(PC3), 
            PC4_acg = mean(PC4), PC5_avg = mean(PC5), num_card_type = length(unique(card_type)), 
            num_curr = length(unique(curr)), num_cheat = sum(is_cheat_bill), num_default = sum(repay_stat))
# For this data, we clean it with below methods:
# For continuous variable: I conducted the PCA
# For catergorical variables: I calculate the summation of different types
write.csv(new_consump_unique, 'consump_clean0.csv')


# PCA is useless, encoding is a better way
consump_recoding <- select(consump_reorder, c(user_id, prior_period_bill_amt, prior_period_repay_amt,
                                              card_type, avlb_bal, credit_lmt_amt, pre_borrow_cash_amt, 
                                              repay_stat, cost_cnt, is_cheat_bill, curr))

# encode the prior_period_repay_amt/prior_period_bill_amt (-1, 0, 1)
consump_recoding$repay_catg <- -1
consump_recoding[consump_recoding$prior_period_repay_amt == 0 & consump_recoding$prior_period_bill_amt <= 0, 'repay_catg'] <- 0
consump_recoding[consump_recoding$prior_period_repay_amt > 0, 'repay_catg'] <- 1

# encode the pre_borrow_cash_amt/redit_lmt_amt (-1, 0, 1)
consump_recoding$avlb_bal_catg <- -1
consump_recoding[consump_recoding$avlb_bal == 0, 'avlb_bal_catg'] <- 0
consump_recoding[consump_recoding$avlb_bal > 0, 'avlb_bal_catg'] <- 1

# round with digit 1
consump_recoding$borrow_ratio <- consump_recoding$pre_borrow_cash_amt/consump_recoding$credit_lmt_amt
#pi / 0 ## = Inf a non-zero number divided by zero creates infinity
#0 / 0  ## =  NaN
consump_recoding[consump_recoding$borrow_ratio %in% c('NaN', 'Inf'), 'borrow_ratio'] <- 0

# meger different rows of the same user into one row
new_consump_recoding <- consump_recoding %>% 
  group_by(user_id) %>%
  summarise(sum_repay = sum(repay_catg), sum_avlb_bal = sum(avlb_bal_catg), mean_borrow_ratio = mean(borrow_ratio),
            num_card_type = length(unique(card_type)), num_curr = length(unique(curr)), num_cheat = sum(is_cheat_bill), 
            num_default = sum(repay_stat), sum_cost_cnt = sum(cost_cnt))

write.csv(new_consump_recoding, './data/data_clean/consump_clean1.csv')
