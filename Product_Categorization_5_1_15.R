## read in csv fron R drive

product_data <- read.csv("\\\\mintape02/projects$/Adobe/TRU_Sitemap/newest_products2.csv", header = TRUE)


library (tm)


### Trying to determine with subtiers related to one and only one parent

# tree.tiers=tree(Tier1 ~ Tier2, product_data)

### Didn't work, too many factors

### Trying Ripper


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


################## Prepare for Classification ###########################################

library(e1071) #ML library including svm and randomForest. works on sparse matrices

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
sample.rows.test <- sample(nrow(dtm.corp.nb.test),5)
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
#### did work ON SAMPLE OF 5. was all wrong. MODEL PROBABLY TOO many variables, NEED TO SIMPLYFY ###

nb.predict.test
save.image()

#put prediction and results in single data frame
results <- as.data.frame(cbind(nb.predict.test,tier1.actuals.nb.test.s))

colnames(results) <- c("Predicted","Actual")

# compare results
table(results$Predicted==results$Actual)
