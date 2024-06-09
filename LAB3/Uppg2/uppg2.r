Author: jose.m.pena@liu.se
# Made for teaching purposes

library(kernlab)
set.seed(1234567890)

data(spam)
foo <- sample(nrow(spam))
spam <- spam[foo,]
spam[,-58]<-scale(spam[,-58])
train <- spam[1:3000, ]
val <- spam[3001:3800, ]
trainva <- spam[1:3800, ]
test <- spam[3801:4601, ] 

by <- 0.3
err_va <- NULL
for(i in seq(by,5,by)){
  filter <- ksvm(type~.,data=train,kernel="rbfdot",kpar=list(sigma=0.05),C=i,scaled=FALSE)
  mailtype <- predict(filter,val[,-58])
  t <- table(mailtype,val[,58])
  err_va <-c(err_va,(t[1,2]+t[2,1])/sum(t))
}

filter0 <- ksvm(type~.,data=train,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter0,val[,-58])
t <- table(mailtype,val[,58])
err0 <- (t[1,2]+t[2,1])/sum(t)
err0

filter1 <- ksvm(type~.,data=train,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter1,test[,-58])
t <- table(mailtype,test[,58])
err1 <- (t[1,2]+t[2,1])/sum(t)
err1

filter2 <- ksvm(type~.,data=trainva,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter2,test[,-58])
t <- table(mailtype,test[,58])
err2 <- (t[1,2]+t[2,1])/sum(t)
err2

filter3 <- ksvm(type~.,data=spam,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter3,test[,-58])
t <- table(mailtype,test[,58])
err3 <- (t[1,2]+t[2,1])/sum(t)
err3

# Questions

# 1. Which filter do we return to the user ? filter0, filter1, filter2 or filter3? Why?

# We return filter3 since it uses all of the data in the model.

# 2. What is the estimate of the generalization error of the filter returned to the user? err0, err1, err2 or err3? Why?

# The error that should be returned to the user should be error2. This model uses trainva as training
# and test for the predicitons. These two are separated. In filter3 for example the data and prediction
# is interconnected, which is not optimal. 


# 3. Implementation of SVM predictions.

sv <- alphaindex(filter3)[[1]]
support.vectors <- spam[sv, -58]
co <- coef(filter3)[[1]]
inte <- - b(filter3)

#  Smaller sigma value -> narrower Gaussian function
kernel.function <- rbfdot(0.05)

k <- NULL
dot.products <- NULL

# Predictions for the first 10 points
for (i in 1:10) {
  
  k2 <- NULL
  
  for (j in seq_along(sv)) {
    k2 <- unlist(support.vectors[j, ])
    sample <- unlist(spam[i, -58])
    # Add the dot product to the existing ones
    dot.products <- c(dot.products, kernel.function(sample, k2))
  }
  # Start and end index
  start <- 1 + length(sv) * (i - 1)
  end <- length(sv) * i
  
  # Calculate predictions and add them
  prediction <- co %*% dot.products[start:end] + inte
  k <- c(k, prediction)
}

k
predict(filter3, spam[1:10, -58], type = "decision")
