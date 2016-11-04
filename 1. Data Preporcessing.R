library(ggplot2)
# Read all datasets into the R project
consump <- read.table('./data/consumption_recode.txt', sep = ',', header = T)
user <- read.table('./data/user_info.txt', sep = ',', header = T)
tag <- read.table('./data/rong_tag.txt', sep = ',', header = T)
relation1 <- read.table('./data/relation1.txt', sep = ',', header = T)
relation2 <- read.table('./data/relation2.txt', sep = ',', header = T)
test_with_random_lable <- read.table('./data/test_with_random_lable.txt', sep = ',', header = T)


############## Users Info ##############
########################################
# Check the number of users 
length(unique(user$user_id))
# [1] 38261 < number of rows 253006, we need to remove repeated rows


# Check the missingness in 'age' and 'sex' columns
user_notime_unique_none <-
  user_notime_unique[
    user_notime_unique$age %in% 'NONE' |
      user_notime_unique$age == 0, ] # check the type of missingness in 'age' and 'sex'
# 3 types:
# NONE1 : 31284
# NONE2 : 31
# NONE3 : 12

user_notime_unique_nonone <- 
  user_notime_unique[
    !(user_notime_unique$age %in% 'NONE' |
        user_notime_unique$age == 0), ] # remove rows with missingness in 'age' or 'sex'
# 38260 users are left, only one user missed, which means only one user doesn't have 
# complete information in any row


# Check and handle duplicated data
# scatter plot: check the trend of the number of rows for every single user_id
## With duplicates
nrow_user <- user %>% 
  group_by(user_id) %>% 
  summarise(nrows = n()) %>% 
  group_by(nrows) %>% 
  summarise(nusers = n())

ggplot(nrow_user, aes(nrows, nusers)) + 
  geom_point() + 
  labs(x = 'number of information rows', y = 'number of users', title = 'Users inforamtion 1') + 
  theme_bw()

# Keep the information about the number of duplicates of every user_id
nrow_user_df <- user %>% 
  group_by(user_id) %>% 
  summarise(nrows = n())

## Reomove duplicates
nrow_user_unique <- user_notime_unique %>% 
  group_by(user_id) %>% 
  summarise(nrows = n()) %>% 
  group_by(nrows) %>% 
  summarise(nusers = n())

ggplot(nrow_user_unique, aes(nrows, nusers)) + 
  geom_point() + 
  labs(x = 'number of information rows', y = 'number of users', title = 'Users inforamtion 2_unique') + 
  theme_bw()

# Keep this dataframe with unique information rows and user_id
nrow_user_unique_df <- user_notime_unique %>% 
  group_by(user_id) %>% 
  summarise(nrows_unique = n()) 

######## We remove duplicated information of users: selection criterion - largest tm_encode (lastest data)
user_clean0 <- arrange(user, desc(tm_encode)) # reorder by time 
user_clean <- user_clean0[!duplicated(user_clean0[, 1]), ]
user_clean <- merge(user_clean, nrow_user_unique_df, by = 'user_id') # add one column of the number of unique rows per user
user_clean <- merge(user_clean, nrow_user_df, by = 'user_id') # add one column of the number of rows per user
user_clean <- user_clean[-13368, ] # remove one row with abnormal value in 'live_info' and 'company_type'
rownames(user_clean) <- c(1:38260)
write.csv(user_clean, file = 'user_clean.csv')


########### relation 1&2 ###############
########################################
relation1 <- read.table('./data/relation1.txt', sep = ',', header = T)
relation2 <- read.table('./data/relation2.txt', sep = ',', header = T)
# write two datases in .csv format
write.csv(relation1, file = './data/data_clean/df_re_1.csv')
write.csv(relation2, file = './data/data_clean/df_re_2.csv')


################# tag ##################
########################################
nrow(tag)
##: number of rows = 687374

length(unique(tag$user_id))
##: [1] 16890, the number of unique user_ids

length(unique(tag$rong_tag))
##: [1] 37359, the number of unique tags

# Scatter plot: check the distribution in tags
ggplot(tag, aes(x = c(1:687374), y = rong_tag)) + 
  geom_point(stat = 'identity') + 
  labs(x = 'Observations', y = 'Tag', title = 'Distribution of tags') +
  theme_bw()


########## Consumption #################
########################################
# Check the number of bills
length(unique(consump$bill_id))
# [1] 677540, number of columns 677540, bill_id is unique
length(unique(consump$user_id))
# [1] 23066 < 677540, one user has multiple bills
# There is another single file to deal with Consumption dataset