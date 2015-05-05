## read in csv fron R drive

product_data <- read.csv("\\\\mintape02/projects$/Adobe/TRU_Sitemap/newest_products2.csv", header = TRUE)

library(tree)
library(ISLR)
library (tm)


### Trying to determine with subtiers related to one and only one parent

# tree.tiers=tree(Tier1 ~ Tier2, product_data)

### Didn't work, too many factors

### Trying Ripper

library(caret)
library(RWeka)

### aborted after running for 3 hours jrip.tiers = train(product_data$Tier2,product_data$Tier2,method="JRip")

# table.tiers = table(product_data$Tier2,product_data$Tier1)

# x <- table.tiers!=0

# tier1.counts <- rowSums(x)

# tier1.counts[tier1.counts>1]

### Sometimes up to 3 tier1s associated with a given tier 2 variables but generally 1-1

### Create table of Tier 1 and order
# x <- table(product_data$Tier1)
# Product.counts.by.tier1 <- x[order(-x)]
# plot(Product.counts.by.tier1)

### Create table of Tier 2 and order
# x <- table(product_data$Tier2)
# Product.counts.by.tier2 <- x[order(-x)]
# plot(Product.counts.by.tier2)

### Most products in 1 of 40 tier1 and 1 of 70 or so tier2

Prod_Desc <- product_data$Product_Description

### Example of creating a termFreq Vector
# PlainTextDocument(Prod_Desc[1])
# Document <- PlainTextDocument(Prod_Desc[1])
### Removed Punctuation and stopwords
# doc1 = termFreq(Document, control = list(removePunctuation=TRUE,stopwords=TRUE))

# PlainTextDocument(Prod_Desc[2])
# Document <- PlainTextDocument(Prod_Desc[2])
### Removed Punctuation and stopwords
# doc2 = termFreq(Document, control = list(removePunctuation=TRUE,stopwords=TRUE))

# corpus <- c(doc1,doc2)



### Removes non products from dataset
product_data_clean <- product_data[product_data$Page_Classifier=="Product",]

### Drops Factor Levels that are not in data
product_data_clean <- droplevels(product_data_clean)




#################### Creation and Preprossessing of entire corpus ##################################

### creating single text corpus
corp <- Corpus(VectorSource(product_data_clean[,17]))

### Removing Punctuation
corp.pp <- tm_map(corp,removePunctuation)

### Removing Stopwords
corp.pp <- tm_map(corp.pp,removeWords,stopwords("english"))

### Remove numbers

corp.pp <- tm_map(corp.pp,removeNumbers)

#################### Creation of document term matrix ##################################

#### Create dtm matrix with tfidf applied
dtm.corp <- DocumentTermMatrix(corp.pp,control=list(weighting=weightTfIdf, minWordLength=2, minDocFreq=40,stopwords=TRUE))


################### Convert to Sparse Matrix  ####################################

library(Matrix)
library(SparseM)

dtm.corp.sparseM <- sparseMatrix(i=dtm.corp$i, j=dtm.corp$j,x=dtm.corp$v)

####################################################################################

# find terms that occur frequently (specify number of docs)
findFreqTerms(dtm.corp,100)





################## Prepare for Classification ###########################################

library(e1071) #ML library including svm and randomForest. works on sparse matrices

################## taking rows out of matrix and class vectors with no words in matrix ###########
x <- rowSums(dtm.corp.sparseM)
x <- x!=0
dtm.corp.sparseM.clean <- dtm.corp.sparseM[x==TRUE,]
tier1.actuals.clean <- product_data_clean$Tier1[x==TRUE]


############### removing entries with no tier 1
x <- tier1.actuals.clean!=""

dtm.corp.sparseM.clean <- dtm.corp.sparseM.clean[x==TRUE,]
tier1.actuals.clean <- tier1.actuals.clean[x==TRUE]

################### Taking Sample of data  ##########################


set.seed(44)
# select rows to sample
sample.rows <- sample(nrow(dtm.corp.sparseM.clean),10000)
#subset into sample and test sets
product_data_clean.sample <- dtm.corp.sparseM.clean[sample.rows,]
product_data_clean.test <- dtm.corp.sparseM.clean[-sample.rows,]
# subset classification into sample and test
tier1.actuals.clean.sample <- tier1.actuals.clean[sample.rows]
tier1.actuals.clean.test <- tier1.actuals.clean[-sample.rows]

#################### Fitting Support Vector Machine (SVM) to Sparse Data Matix  ########

#### Performing some simple tuning on cost factor
# Fits svm including calculating accuracy using 5 fold cross validation for different levels of cost
svm.sparse.sample.1.0 <- svm(product_data_clean.sample,tier1.actuals.clean.sample,kernel = "linear",cross = 5,cost=1,probability=T)
svm.sparse.sample.0.1 <- svm(product_data_clean.sample,tier1.actuals.clean.sample,kernel = "linear",cross = 5,cost=0.1,probability=T)
svm.sparse.sample.10  <- svm(product_data_clean.sample,tier1.actuals.clean.sample,kernel = "linear",cross = 5,cost=10,probability=T)

# gives summary of model
summary(svm.sparse.sample.1.0)

# capture output into text file
x<-summary(svm.sparse.sample.1.0)
capture.output(x,file = "svm.sparse.sample.1.0.txt")
x<-summary(svm.sparse.sample.0.1)
capture.output(x,file = "svm.sparse.sample.0.1.txt")
x<-summary(svm.sparse.sample.10)
capture.output(x,file = "svm.sparse.sample.10.txt")

# Cost of 1 had highest accuracy (72.6%, compared to 71.7% for cost of 10. 
# Cost of 0.1 had far lower performance - 49.6%

# fit the final svm model
svm.sparse.sample <- svm(product_data_clean.sample,tier1.actuals.clean.sample,kernel = "linear",cross = 5,cost=1,probability=T)


# predict results on test set
svm.predict.sparse.test <- predict(svm.sparse.sample,product_data_clean.test)

# put prediction and results in single data frame
results <- as.data.frame(cbind(svm.predict.sparse.test,tier1.actuals.clean.test))
colnames(results) <- c("Predicted","Actual")

# compare results
table(results$Predicted==results$Actual)

#### Accuracy of 0.7580736 on test set ####


######################## Naive Bayes ##################################

#### Changing Sparse Matrix to data frame 

# dtm.corp.df <- as.data.frame(inspect(dtm.corp)) Doesn't work as data frame too large

##### create training data  ######

###### NOTE I HAVE NOT YET REMOVED BLANK ROWS OR ROWS WITH "" CLASS FROM DATA ####

set.seed(45)
# select rows to sample
sample.rows <- sample(nrow(dtm.corp),10000)
# sample original document term matrix
dtm.corp.nb.sample <- dtm.corp[sample.rows,]
# coerce dtm into a data frame
dtm.corp.nb.sample <- as.data.frame(inspect(dtm.corp.nb.sample))
# create matching sample of class
tier1.actuals.nb.sample <-product_data_clean$Tier1[sample.rows]


#### create test data

dtm.corp.nb.test <- dtm.corp[-sample.rows,]
tier1.actuals.nb.test <- product_data_clean$Tier1[-sample.rows]

# select rows to sample because entire test set too large to coerce to data frame
set.seed(46)
sample.rows.test <- sample(nrow(dtm.corp.nb.test),5000)
# subset to sample
dtm.corp.nb.test.s <- dtm.corp.nb.test[sample.rows.test,]
# coerce dtm into a data frame
df.corp.nb.test.s <- as.data.frame(inspect(dtm.corp.nb.test.s))
# create matching sample of class
tier1.actuals.nb.test.s <- tier1.actuals.nb.test[sample.rows.test]




##### train naive bayes model
nb.sparse.sample <- naiveBayes(dtm.corp.nb.sample,tier1.actuals.nb.sample)



#predict results on test set
nb.predict.test <- predict(nb.sparse.sample,df.corp.nb.test.s)

save.image()

#put prediction and results in single data frame
results <- as.data.frame(cbind(nb.predict.test,tier1.actuals.nb.test.s))

colnames(results) <- c("Predicted","Actual")

# compare results
table(results$Predicted==results$Actual)


#################### Old Code ######################################################

################### Attempt to tune using tune function resulted in errors - possibly due to sparse matrix ######

# tuning svm with 5 fold cross validation varying epsilon and cost
# svm.sparse.sample.tune <- tune(svm,dtm.corp.sparseM,product_data_clean.sample$Tier1, ranges = list(epsilson = c(0.01,0.1,1),cost = 2^(-3:3)), probability=T, tunecontrol = tune.control(sampling = "cross", cross=5))
# svm.sparse.sample.tune <- tune(svm,dtm.corp.sparseM,product_data_clean.sample$Tier1, ranges = list(epsilson = c(0.01,0.1,1)), probability=T, tunecontrol = tune.control(sampling = "cross", cross=5))
# svm.sparse.sample.tune <- tune(svm,dtm.corp.sparseM,product_data_clean.sample$Tier1, tunecontrol = tune.control(sampling = "cross", cross=2))
# tune.out=tune(svm ,dtm.corp.sparseM.clean,tier1.actuals.clean,kernel ="linear",ranges =list(cost=c(0.001 , 0.01) ),tunecontrol = tune.control(sampling = "cross", cross=2))
# above tuning resulted in error "Error in tab[lev, lev] : subscript out of bounds". 
# Tried multiple different settings and removed any 0 rows and "" labled tiers from data. still didn't work


#### Changing Sparse Matrix to data frame 

# dtm.corp.df <- as.data.frame(inspect(dtm.corp)) Doesn't work as data frame too large

dtm.corp.df <- removeSparseTerms(dtm.corp,0.9999)


# converts dtm to data frame 
dtm.corp.df <- as.data.frame(inspect(dtm.corp.df))



### Splits dataframe into different list elements by Tier 1  
product_data_clean_t1_subsets <- split (product_data_clean,product_data_clean$Tier1)

tier1.levels <- levels(product_data_clean$Tier1)

number.subsets <- length(product_data_clean_t1_subsets)  # count the number of subsets in the list

### for loop that creates a separate corpus for each subset


for (i in 1:number.subsets) {  
  
  subset <- as.data.frame(product_data_clean_t1_subsets[i])
  prod_desc_1 <- subset[,17]
  corp <- Corpus(VectorSource(prod_desc_1))
  dtm <- DocumentTermMatrix(corp,control = list(removePunctuation=TRUE,stopwords=TRUE))
  assign(paste0("corp.", i), corp)
  assign(paste0("dtm.", i), dtm)
}



##### 2 Approaches to Try. Create Vectors for each corpus and use a cosine distance type approach

###### Use Ida package and slda.em function set to logistic to predict single outcome (like outdoors)



#######################
subset <- as.data.frame(product_data_clean_t1_subsets[1])
prod_desc_1 <- subset[,17]
corp <- Corpus(VectorSource(prod_desc_1))
dtm <- DocumentTermMatrix(corp)
assign(paste0("corp.", tier1.levels[i]), corp)
assign(paste0("dtm.", tier1.levels[i]), dtm)
#######################

### Creates a corpus from a character vector
corp <- Corpus(VectorSource(prod_desc_1))
dtm <- DocumentTermMatrix(corp)



### loop example for assigning name to variable in loop
for (i in 1:5) {
                if (i == 4) {end()}
                else assign(paste0("Varx1", i), 10*i)
                } 

docs.per.subset <- NULL
for ( i in 1:126) {
  docs.per.subset[i] <-nrow(as.data.frame(product_data_clean_t1_subsets[i]))
}

docs.per.subset <- cbind(tier1.levels,docs.per.subset)
