
# IMPORTANT
# All of the work here was adapted from the user Avik Paul on Kaggle
# For more information, please see: https://www.kaggle.com/code/avikpaul4u/titanic-machine-learning-in-r

# Load important packages
library(dplyr)
library(ROCR)
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(randomForest)
library(readxl)

# Read in the data
test_og = read_excel("~/Desktop/test.xlsx")
train_og = read_excel("~/Desktop/train.xlsx")

# Create new version of both datasets
test_cl = test_og
train_cl = train_og

# Add a "Survived" measure to the test data (it is not present because it needs to be predicted)
test_cl$Survived = " " # this adds a blank value
test_cl = test_cl %>% 
  relocate(Survived, .after = PassengerId) # this moves the "Survived" column next to PassengerID

# Combine the data
comb = rbind(train_cl,test_cl)
nrow(comb) - nrow(train_cl) - nrow(test_cl) # rows match up
# View(comb) - this will allow us to see when "Survived" changes from 1 or 0 to " "
str(comb)
glimpse(comb)

# Make things easier to manipulate
comb = comb %>%
  mutate(Survived = factor(Survived, levels = c(0,1), labels = c("No_Survived", "Yes_Survived")),
         Sex = factor(Sex),
         Pclass = factor(Pclass, levels = c(1,2,3), labels = c("First_Class", "Second_Class", "Third_Class")))

#### Basic Visualizations ####

# Survival rates by class
ggplot(comb, aes(x= Pclass, fill = Survived)) +
  geom_bar(width=0.5) # looks like survival rates in first class were the highest

# Survival rates by sex
ggplot(comb, aes(x= Sex, fill = Survived)) +
  geom_bar(width=0.5) # women survived more than men

# Survival rates by class and sex together
ggplot(comb, aes(Sex)) +
  facet_wrap(~Pclass) +
  geom_bar(aes(y = (..count..)/sum(..count..), fill=Survived), stat= "count")+
  geom_text(aes(label = scales::percent(round((..count..)/sum(..count..),2)),
                y= ((..count..)/sum(..count..))), stat="count",
            vjust = -.25) +
  ggtitle("Class") + labs(y = "percent")

# Look at proportion who survived or didn't survive
tbl = aggregate(PassengerId~Survived, comb, length)
colnames(tbl)[2] = "Number_Passengers"
tbl$Proportion = (tbl$Number_Passengers/sum(tbl$Number_Passengers)*100)
tbl[1,3] # prop who didn't survive
tbl[2,3] # prop who did survive

#### Crude Imputation of Missing Values ####
comb$Embarked = as.factor(comb$Embarked) # this will help with imputation
summary(comb)

# for 'Embarked' variable, there are two missing values
# let's change these to 'S' - the most common value
comb$Embarked[is.na(comb$Embarked)] = "S"
summary(comb$Embarked) # now, two 'NA' values are 'S' values

# look at age values
age_values = summarise(group_by(comb,Pclass,Sex,Embarked), 
                       MeanAge=mean(Age,na.rm=T))
age_values$key=paste0(age_values$Pclass,"_",age_values$Sex,"_", age_values$Embarked)
age_values

# taking means and putting them back into full data
comb$key=paste0(comb$Pclass, "_", comb$Sex, "_", comb$Embarked) # does some nice summarizing in a new column
comb=merge(x=comb, y = age_values[,c("key","MeanAge")], by='key',all.x=T) # imputed mean age for each segment

# now, we can take imputed mean ages from each segment and modify the data to make missing values those means
comb$Age=ifelse(
  is.na(comb$Age),comb$MeanAge,comb$Age) # note: this approach overrides old "Age" variable
summary(comb$Age)

# look at fare values
summary(comb$Fare) # one missing value
# we can just add the mean to this
comb$Fare = ifelse(
  is.na(comb$Fare),mean(comb$Fare,na.rm=T),comb$Fare)
summary(comb$Fare) # we added the mean, so nothing changed (NA just is gone)

#### Feature Engineering ####

# this will be a way to consolidate the amount of information present in the dataset
# in essence, it is a very crude way of simplifying the dataset
# it's pretty p-hacky; you have unlimited degrees of freedom, as it's all ad-hoc

# we can do it for titles
comb$Title = sapply(as.character(comb$Name), # extracts the title from the name
                    FUN = function(x){strsplit(x,"[,.]")[[1]][2]}) # strsplit is cool
comb$Title = sub(' ', '', comb$Title)
comb$Title = as.factor(comb$Title)
summary(comb$Title) # lots of wild titles (e.g., "the Countess")

# grouping different titles together
comb$Title = as.character(comb$Title) # solely to undo what we just did for summary lol
comb$Title[comb$Title %in% c("Mlle","Ms")] = "Miss"
comb$Title[comb$Title == "Mme"] = "Mrs"
comb$Title[comb$Title %in% c("Don", "Sir", "Jonkheer", "Rev", "Dr")] = "Sir"
comb$Title[comb$Title %in% c("Dona", "Lady", "the Countess")] = "Lady"
comb$Title[comb$Title %in%  c("Capt", "Col", "Major")] = "Officer"
comb$Title = as.factor(comb$Title)
summary(comb$Title) # we have now consolidated the Title data into fewer bins

# family grouping
comb$famsz = comb$SibSp + comb$Parch
table(comb$famsz) # size of different families (0 = traveling solo)

# grouping according to different family sizes
comb$FamGrp[comb$famsz == 0] = "Solo"
comb$FamGrp[comb$famsz < 5 & comb$famsz > 0] = "Small"
comb$FamGrp[comb$famsz > 4] = "Large"
comb$FamGrp = as.factor(comb$FamGrp)
summary(comb$FamGrp) # lots of people traveling solo

length(unique(comb$Ticket)) # 929 unique tickets
length(unique(comb$PassengerId)) # 1309 passengers
# what this means is that some people are traveling together (and paid the fare for other people)
# in turn, we should be able to see this on tickets
tick_count = data.frame(table(comb$Ticket))
head(tick_count) # yeah, there are some duplicates

# put frequency of tickets into the full data set
comb = merge(comb, tick_count, by.x="Ticket", by.y="Var1", all.x=T) 

# grouping according to number of people on each ticket
comb$Ticket_size[comb$Freq==1] = "Single_Ticket"
comb$Ticket_size[comb$Freq>1 & comb$Freq < 5] = "Small_Ticket"
comb$Ticket_size[comb$Freq>=5] = "Big_Ticket"
comb$Ticket_size = as.factor(comb$Ticket_size)

# grouping people according to how old they are - minor or adult
comb$Minor[comb$Age < 18] = "Minor"
comb$Minor[comb$Age >= 18] = "Adult"
comb$Minor = as.factor(comb$Minor)
table(comb$Minor) # not very many minors

#### Extracting Training Data ####

train_df = subset(comb, !(comb$Survived=="")) # only getting folks for whom there is "Survived" data
str(train_df$Survived) # two levels

# extracting a portion of the data
train_val = sample_frac(train_df, size=0.8) # train_val is 80% of the training data
test_val = subset(train_df, !(train_df$PassengerId %in% train_val$PassengerId)) # basically, setting aside other 20%

#### Logistic Regression ####

mod = glm(Survived ~ Pclass + Title + FamGrp + Sex + Minor + Ticket_size + 
            Embarked, family = "binomial", data = train_val)
summary(mod) # only things that look inherently meaningful are first/second class

# predicted probabilities obtained from the model - at the individual level
predict_train = predict(mod, train_val, type='response')
prob_train = ifelse(predict_train > 0.5,1,0) # arbitrary cut off of .5
sum(prob_train)

#### Confusion Matrix - Training Data ####

# this is a simple way to assess the accuracy of the model predictions
# there can be true/false positives/negatives

conf_matrix_train = table(prob_train, train_val$Survived)
print(conf_matrix_train) # here, we can see where predictions are accurate and where they are wrong
accuracy_train = sum(diag(conf_matrix_train))/sum(conf_matrix_train)
print(accuracy_train*100) # we can intuitively see where this comes from - accurate cases/inaccurate cases
# to verify, just do the math by hand - it should be intuitive

#### Build an ROC Curve ####

pred1 = prediction(predict(mod), train_val$Survived) # getting the data into ROC-compatible form
perf1 = performance(pred1, "tpr", "fpr")
plot(perf1) # looks pretty good - the Y axis rises very quickly relative to X axis

# let's assess the model's accuracy using the data that we set aside for validation
predict_test = predict(mod, test_val, type = 'response')
prob_test = ifelse(predict_test > 0.5, 1, 0) # same scheme as before, but with test data this time

#### Confusion Matrix - Test Data ####

conf_matrix_test = table(prob_test, test_val$Survived)
accuracy_test = sum(diag(conf_matrix_test))/sum(conf_matrix_test)
print(accuracy_test*100) # slightly worse, which isn't crazy; but these numbers are meaningfully similar

# let's try a different approach - making a decision tree

#### Decision Tree ####

tree_mod = rpart(Survived ~ Pclass + Title + FamGrp + Sex + Minor + Ticket_size + 
  Embarked, data = train_val, method = "class") 
summary(tree_mod) # this is very confusing and kind of opaque
# let's instead try to plot the tree
rpart.plot(tree_mod, fallen.leaves=F, extra=3)

#### Confusion Matrix - Decision Tree ####

predict_train_tree = predict(tree_mod,data=train_val,type = "class")
confusionMatrix(predict_train_tree,train_val$Survived)

# looking at the model accuracy
prediction_test_dt = predict(tree_mod,test_val,type = "class")
confusionMatrix(prediction_test_dt,test_val$Survived) # slightly better using decision tree approach

#### Random Forest Model ####

set.seed(1234) # to make this reproducible
rf_mod = randomForest(Survived ~ Pclass + Title + FamGrp + Sex + Minor + Ticket_size + 
                        Embarked, data = train_val[,c("Survived", "Pclass", "Title" , "FamGrp" , "Sex" , 
                        "Minor" , "Ticket_size", "Embarked")],
                      importance = T, ntree = 1000, mtry=2)
print(rf_mod)

# look at variable importance (for each of the predictors)
varImpPlot(rf_mod)

# what if we drop "Minor" variable - that appears to not be doing a whole lot
set.seed(1234)
rf_mod2 = randomForest(Survived ~ Pclass + Title + FamGrp + Sex + Ticket_size + 
                         Embarked, data = train_val[,c("Survived", "Pclass", "Title" , "FamGrp" , "Sex" , 
                                                       "Ticket_size", "Embarked")],
                       importance = T, ntree = 1000, mtry=2)
print(rf_mod2)

#### Cross Validation on Random Forest ####

set.seed(1234)
rf_mod3 = randomForest(Survived ~ Pclass + Title + FamGrp + Sex + Minor + Ticket_size + 
                         Embarked, data = test_val[,c("Survived", "Pclass", "Title" , "FamGrp" , "Sex" , 
                                                       "Minor" , "Ticket_size", "Embarked")],
                       importance = T, ntree = 1000, mtry=2)
print(rf_mod3)






