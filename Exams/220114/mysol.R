library(tree)
rm(list = ls())

data = read.csv("adult.csv", stringsAsFactors = T)

n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.6))
train=data[id,]
id1=setdiff(1:n, id)
set.seed(12345)
id2=sample(id1, floor(n*0.2))
valid=data[id2,]
id3=setdiff(id1,id2)
test=data[id3,]

default_tree = tree(C13~.-C12, data=train)
summary(default_tree)
plot(default_tree)
nr_of_leaves = 8

trainScore=rep(0,nr_of_leaves)
testScore=rep(0,nr_of_leaves)
for(i in 2:nr_of_leaves) {
  prunedTree=prune.tree(default_tree,best=i)
  pred=predict(prunedTree, newdata=valid,
               type="tree")
  trainScore[i]=deviance(prunedTree)
  testScore[i]=deviance(pred)
}
plot(2:nr_of_leaves, trainScore[2:nr_of_leaves], type="b", col="red",
     ylim=c(0,20000))
points(2:nr_of_leaves, testScore[2:nr_of_leaves], type="b", col="blue")

#first one is 0 so remove that column then add 1 to correct index
opt_leaves = which.min(testScore[-1])+1
opt_tree = prune.tree(default_tree, best = opt_leaves)
plot(opt_tree)
text(opt_tree)
opt_tree

#Task 1.2
pis = seq(0.1, 0.9, 0.1)
ps=length(pis)
F1=numeric(ps)
acc=numeric(ps)
test_pred = predict(opt_tree, newdata = test, type = "vector")
for (pi in pis) {
  pred_tree = ifelse(test_pred[,2]>pi, ">50K", "<=50K")
  tab=table(test$C13, pred_tree)
  TP = tab[2,2]
  FP = tab[1,2]
  FN = tab[2,1]
  TN = tab[1,1]
  P = TP + FN
  N = TN + FP
  precision = TP/(TP + FP)
  recall = TP/P
  F1[pi]=(2*precision*recall)/(precision + recall)
  acc[pi]=(TP+TN)/(P+N)
}
rbind(pis,acc,F1)

#Task 1.3
library(glmnet)
library(dplyr)
selected_data = as.matrix(train %>% select(C1, C9, C10))
model=cv.glmnet(selected_data, train$C11, alpha=1,family="gaussian")
opt_lambda = model$lambda.min
print(opt_lambda)
plot(model)
coef(model, s="lambda.min")

#Assignment 2
#Task1
rm(list=ls())
#Given code
set.seed(123456789)
N_class1 <- 1000
N_class2 <- 1000
data_class1 <- NULL
for(i in 1:N_class1){
  a <- rbinom(n = 1, size = 1, prob = 0.3)
  b <- rnorm(n = 1, mean = 15, sd = 3) * a + (1-a) * rnorm(n = 1, mean = 4, sd
                                                           = 2)
  data_class1 <- c(data_class1,b)
}
data_class2 <- NULL
for(i in 1:N_class2){
  a <- rbinom(n = 1, size = 1, prob = 0.4)
  b <- rnorm(n = 1, mean = 10, sd = 5) * a + (1-a) * rnorm(n = 1, mean = 15,
                                                           sd = 2)
  data_class2 <- c(data_class2,b)
}

conditional_class1 <- function(t, h){
  d <- 0
  for(i in 1:800)
    d <- d+dnorm((t-data_class1[i])/h)
  return (d/800)
}

conditional_class2 <- function(t, h){
  d <- 0
  for(i in 1:800)
    d <- d+dnorm((t-data_class2[i])/h)
  return (d/800)
}

#Estimate the class posterior probability distribution: 1p.

prob_class1 <- function(t, h){
  prob_class1 <- conditional_class1(t,h)*800/1600
  prob_class2 <-conditional_class2(t,h)*800/1600
  
  return (prob_class1/(prob_class1 + prob_class2))
}

# Select h value via validation: 1p.

foo <- NULL
for(h in seq(0.1,5,0.1)){
  foo <- c(foo, (sum(prob_class1(data_class1[801:900], h)>0.5)+sum(prob_class1(data_class2[801:900], h)<0.5))/200)
}
plot(seq(0.1,5,0.1),foo)

max(foo)
which(foo==max(foo))*0.1

# Estimate the generalization error: 2p.

# To estimate the generalization error, we use the best h value found previously.
# Note that the training data is now the old training data union the validation data.
# Using just the old training data results results in an estimate that is a bit
# too pessimistic.

conditional_class1 <- function(t, h){
  d <- 0
  for(i in 1:900)
    d <- d+dnorm((t-data_class1[i])/h)
  
  return (d/900)
}

conditional_class2 <- function(t, h){
  d <- 0
  for(i in 1:900)
    d <- d+dnorm((t-data_class2[i])/h)
  
  return (d/900)
}

prob_class1 <- function(t, h){
  prob_class1 <- conditional_class1(t,h)*900/1800
  prob_class2 <-conditional_class2(t,h)*900/1800
  
  return (prob_class1/(prob_class1 + prob_class2))
}

h <- which(foo==max(foo))*0.1
(sum(prob_class1(data_class1[901:1000], h)>0.5)+sum(prob_class1(data_class2[901:1000], h)<0.5))/200

#Task 2.2
rm(list=ls())
library(neuralnet)
set.seed(1234567890)
Var <- runif(50, 0, 10) #Sampling 50 points randomly and uniformly in the interval [0,10]
trva <- data.frame(Var, Sin=sin(Var)) #data frame with the variable and the sin value of the var
train <- trva[1:25,] # Training
valid <- trva[26:50,] # Validation
restr <- vector(length = 10)
resva <- vector(length = 10)
winit <- runif(31, -1, 1) # Random initializaiton of the weights in the interval [-1, 1]
for(i in 1:10) {
  nn <- neuralnet(formula = Sin ~ Var, data = train, hidden = 10, startweights = winit,
                  threshold = i/1000, lifesign = "full")
  
  aux <- predict(nn, train) # Compute predictions for the trainig set and their squared error
  restr[i] <- sum((train[,2] - aux)**2)/2
  
  aux <- predict(nn, valid) # The same for the validation set
  resva[i] <- sum((valid[,2] - aux)**2)/2
}
plot(restr, type = "o")
plot(resva, type = "o")
restr
resva
# The graphs show an example of overfitting, i.e. the threshold that achieves the lowest squared error
# in the training set is not the one that achieves the lowest error in the validation set. Therefore, 
# early stopping is necessary, i.e. running gradient descent until convergence is not the best option,
# as the lowest threshold gives the best error in the training set but not in the validation set.
# Specifically, the validation set indicates that gradient descent should be stoped when 
# threshold = 4/1000. So, the output should be a NN learnt with all (!) the data available and the
# threshold = 4/1000.

winit <- runif(31, -1, 1)
plot(nn <- neuralnet(formula = Sin ~ Var, data = trva, hidden = 10, startweights = winit,
                     threshold = 4/1000, lifesign = "full"))

# Plot of the predictions (blue dots) and the data available (red dots)

plot(trva[,1],predict(nn,trva), col="blue", cex=3)
points(trva, col = "red", cex=3)
