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






################### Taking Sample of data  ##########################

product_data_clean.sample <- product_data_clean[sample(nrow(product_data_clean),10000),]

#################### Creation and Preprossessing of entire corpus ##################################

### creating single text corpus
corp <- Corpus(VectorSource(product_data_clean.sample[,17]))

### Removing Punctuation
corp.pp <- tm_map(corp,removePunctuation)

### Removing Stopwords
corp.pp <- tm_map(corp.pp,removeWords,stopwords("english"))

### Remove numbers

corp.pp <- tm_map(corp.pp,removeNumbers)

#################### Creation of document term matrix ##################################

#### Create dtm matrix with tfidf applied
dtm.corp <- DocumentTermMatrix(corp.pp,control=list(weighting=weightTfIdf, minWordLength=2, minDocFreq=20,stopwords=TRUE))


################### Convert to Sparse Matrix  ####################################

library(Matrix)
library(SparseM)

dtm.corp.sparseM <- sparseMatrix(i=dtm.corp$i, j=dtm.corp$j,x=dtm.corp$v)

####################################################################################

# find terms that occur frequently (specify number of docs)
findFreqTerms(dtm.corp,100)





################## Prepare for Classification ###########################################

# dtm.corp.df <- as.data.frame(inspect(dtm.corp)) Doesn't work as data frame too large

dtm.corp.df <- removeSparseTerms(dtm.corp,0.9999)


# converts dtm to data frame 
dtm.corp.df <- as.data.frame(inspect(dtm.corp.df))

library(e1071) #ML library including svm and randomForest. works on sparse matrices









#################### Fitting Support Vector Machine (SVM) to Sparse Data Matix  ########

# Fits svm including calculating accuracy using 5 fold cross validation
svm.sparse.sample <- svm(dtm.corp.sparseM,product_data_clean.sample$Tier1,kernel = "linear",cross = 5)

# gives summary of model
summary(svm.sparse.sample)

# predict results on training set
svm.predict.sparse.sample <- predict(svm.sparse.sample,dtm.corp.sparseM)

# put prediction and results in single data frame
results <- as.data.frame(cbind(svm.predict.sparse.sample,product_data_clean.sample$Tier1))
colnames(results) <- c("Predicted","Actual")

# compare results
table(results$Predicted==results$Actual)

set.seed(1)

# tuning svm with 5 fold cross validation varying epsilon and cost
# svm.sparse.sample.tune <- tune(svm,dtm.corp.sparseM,product_data_clean.sample$Tier1, ranges = list(epsilson = c(0.01,0.1,1),cost = 2^(-3:3)), probability=T, tunecontrol = tune.control(sampling = "cross", cross=5))
# svm.sparse.sample.tune <- tune(svm,dtm.corp.sparseM,product_data_clean.sample$Tier1, ranges = list(epsilson = c(0.01,0.1,1)), probability=T, tunecontrol = tune.control(sampling = "cross", cross=5))
# svm.sparse.sample.tune <- tune(svm,dtm.corp.sparseM,product_data_clean.sample$Tier1, tunecontrol = tune.control(sampling = "cross", cross=2))
# above tuning resulted in error "Error in tab[lev, lev] : subscript out of bounds"

################## taking rows out of matrix and class vectors with no words in matrix ###########
x <- rowSums(dtm.corp.sparseM)
x <- x!=0
dtm.corp.sparseM.clean <- dtm.corp.sparseM[x==TRUE,]
tier1.actuals.clean <- product_data_clean.sample$Tier1[x==TRUE]

tune.out=tune(svm ,dtm.corp.sparseM,product_data_clean.sample$Tier1,kernel ="linear",ranges =list(cost=c(0.001 , 0.01) ),tunecontrol = tune.control(sampling = "cross", cross=2))









#################### Old Code ######################################################


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
