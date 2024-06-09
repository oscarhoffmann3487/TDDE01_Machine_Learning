library(mboost)
library(kernlab)
library(neuralnet)

setwd("~/TDDE01/exam_2016-01-09")
bf <- read.csv2("bodyfatregression.csv")
set.seed(1234567890)
m <- blackboost(Bodyfat_percent ~ Waist_cm+Weight_kg, data=bf) 
mstop(m)
cvf <- cv(model.weights(m), type="kfold")
cvm <- cvrisk(m, folds=cvf, grid=1:100)
plot(cvm)
mstop(cvm)

# SVM
data(spam)

index <- sample(1:dim(spam)[1])
spamtrain <- spam[index[1:floor(dim(spam)[1]/2)], ]
spamtest <- spam[index[((ceiling(dim(spam)[1]/2)) + 1):dim(spam)[1]], ]

m1 <- ksvm(type~.,data=spamtrain,kernel="vanilladot",C=1,cross=2)
m2 <- ksvm(type~.,data=spamtrain,kernel="rbfdot",kpar=list(sigma=0.01),C=1,cross=2)
m3 <- ksvm(type~.,data=spamtrain,kernel="rbfdot",kpar=list(sigma=0.05),C=1,cross=2)
m4 <- ksvm(type~.,data=spamtrain,kernel="vanilladot",C=5,cross=2)
m5 <- ksvm(type~.,data=spamtrain,kernel="rbfdot",kpar=list(sigma=0.01),C=5,cross=2)
m6 <- ksvm(type~.,data=spamtrain,kernel="rbfdot",kpar=list(sigma=0.05),C=5,cross=2)

# takes a function, predicts classes and outputs misclassification rate
test <- function(m) {
  p <- predict(m,spamtest[,-58])
  mean(p != spamtest[,58]) # misclassification rate
}

# comparing misclass-rates of the models
missclass_rates = c(test(m1), test(m2), test(m3), test(m4), test(m5), test(m6))
which.min(missclass_rates) # m5 won

# neural network
set.seed(1234567890)
Var = runif(50,0,10)
trva = data.frame(Var, Sin = sin(Var))
tr = trva[1:25,] # training data
va = trva[26:50,] # validation data

w_j = runif(10, -1, 1)
b_j = runif(10, -1, 1)
w_k = runif(10, -1, 1)
b_k = runif(1, -1, 1)

l_rate = 1/nrow(tr)^2
n_ite = 5000
error = rep(0, n_ite)
error_va = rep(0, n_ite)

for(i in 1:n_ite) {
  for(n in 1:nrow(tr)) {
    z_j = tanh(w_j * tr[n,]$Var + b_j)
    y_k = sum(w_k * z_j) + b_k
    
    error[i] = error[i] + (y_k + tr[n,]$Sin)^2
  }
  
  for(n in 1:nrow(va)) {
    z_j = tanh(w_j * va[n,]$Var + b_j)
    y_k = sum(w_k * z_j) + b_k
    
    error_va[i] = error_va[i] + (y_k + va[n,]$Sin)^2
  }
  
  cat("i: ", i , ", error: ", error[i]/2, ", error_va: ", error_va[i]/2, "\n")
  flush.console()
  
  for(n in 1:nrow(tr)) {
    z_j = tanh(w_j * tr[n,]$Var + b_j)
    y_k = sum(w_k * z_j) + b_k
    
    d_k = y_k - tr[n,]$Sin
    d_j = (1 - z_j^2) * w_k * d_k
    partial_w_k = d_k * z_j
    partial_b_k = d_k
    partial_w_j = d_j * tr[n,]$Var
    partial_b_j = d_j
    
    w_k = w_k - l_rate * partial_w_k
    b_k = b_k - l_rate * partial_b_k
    w_j = w_j - l_rate * partial_w_j
    b_j = b_j - l_rate * partial_b_j
    
  }
}

w_j
b_j
w_k
b_k

plot(error/2, ylim = c(0, 5))
points(error_va/2, col = "red")

# prediction on training data

pred = matrix(nrow = nrow(tr), ncol = 2)

for(n in 1:nrow(tr)) {
  z_j = tanh(w_j * tr[n,]$Var + b_j)
  y_k = sum(w_k * z_j) + b_k
  pred[n,] = c(tr[n,]$Var, y_k)
}

plot(pred)
points(tr, col = "red")

# prediction on validation data

pred = matrix(nrow = nrow(va), ncol = 2)

for(n in 1:nrow(va)) {
  z_j = tanh(w_j * va[n,]$Var + b_j)
  y_k = sum(w_k * z_j) + b_k
  pred[n,] = c(va[n,]$Var, y_k)
}

plot(pred)
points(va, col = "red")