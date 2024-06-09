library(kknn)
library(dplyr)
#Task1
data = read.csv("tecator.csv")
n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=data[id,]
test=data[-id,]

mse_test = c(length(30))
mse_train = c(length(30))
for (k in 1:30) {
  kknn_train = kknn(train$Fat ~. -Sample -Protein -Moisture, train=train, test=train, kernel="rectangular", k=k)
  kknn_test = kknn(train$Fat ~. -Sample -Protein -Moisture, train=train, test=test, kernel="rectangular", k=k)
  
  train_pred = predict(kknn_train)
  test_pred = predict((kknn_test))
  mse_train[k] = mean((train$Fat - train_pred)^2)
  mse_test[k] = mean((test$Fat - test_pred)^2)
}

plot(mse_test, type="l", x = seq(1,30,1), ylim=c(0,140))
lines(mse_train, col="blue")

print(which.min(mse_test))
mse_test[2]
mse_train[2]
#Test min value at X=2 gives mse of 75.49. This is the
#Optimal balance between a simple and a complex model
#train mse = 21.77

#Task1.2
pca = princomp(data)
pca_scores = pca$scores

train_PC=data.frame(pca_scores[id,])
test_PC=data.frame(pca_scores[-id,])
train_PC$Fat = train$Fat
test_PC$Fat = test$Fat

train_kknn = kknn(Fat ~ + Comp.1 + Comp.2 + Comp.3 + Comp.4 + Comp.5 + Comp.6 + Comp.7 + Comp.8 + Comp.9 + Comp.10, train=train_PC, test=train_PC, kernel="rectangular", k=2)
test_kknn = kknn(Fat ~ + Comp.1 + Comp.2 + Comp.3 + Comp.4 + Comp.5 + Comp.6 + Comp.7 + Comp.8 + Comp.9 + Comp.10, train=train_PC, test=test_PC, kernel="rectangular", k=2)

mse_train_PC = mean((train_PC$Fat - predict(train_kknn))^2)
mse_test_PC = mean((test_PC$Fat - predict(test_kknn))^2)

#1.3
library(caret)
scaler = preProcess(train)

train_scaled = predict(scaler, train)
test_scaled = predict(scaler, test)

theta = rep(0,4)
mse_train_v = c()
mse_test_v = c()
cost_function = function(theta){
  predict_train = cbind(1, train_scaled$Fat, train_scaled$Fat^2, train_scaled$Fat^3)%*%theta
  predict_test = cbind(1, test_scaled$Fat, test_scaled$Fat^2, test_scaled$Fat^3)%*%theta
  
  mse_train = mean((train_scaled$Protein - predict_train)^2)
  mse_test = mean((test_scaled$Protein - predict_test)^2)
  mse_train_v <<- c(mse_train_v, log(mse_train))
  mse_test_v <<- c(mse_test_v, log(mse_test))
  
  return(mse_train)
}

opt = optim(par=theta, fn=cost_function, method = "CG")
mse_train_v = c()
mse_test_v = c()
opt20 = optim(par=theta, fn=cost_function, method = "CG", control = list(maxit=20))
mse_train_v = c()
mse_test_v = c()
opt200 = optim(par=theta, fn=cost_function, method = "CG", control = list(maxit=200))
predicted_protein20 = cbind(1, train_scaled$Fat, train_scaled$Fat^2, train_scaled$Fat^3) %*% opt20$par
predicted_protein200 = cbind(1, train_scaled$Fat, train_scaled$Fat^2, train_scaled$Fat^3) %*% opt200$par
x_axis = seq(1, length(mse_train_v), 1)
plot(x=x_axis,y=mse_train_v, type="l", xlim=c(0,200))
lines(mse_test_v, col="blue")

plot(train_scaled$Fat, train_scaled$Protein)
points(train_scaled$Fat, predicted_protein20, col="green")

plot(train_scaled$Fat, train_scaled$Protein)
points(train_scaled$Fat, predicted_protein200, col="green")

#Task 2.1
library(kernlab)
library(caret)
#data(spam)
"""
foo = sample(nrow(spam))
spam = spam[foo,]

scaler <- preProcess(spam[1:3000, -58])
tr <- predict(scaler, spam[1:3000, ])
va <- predict(scaler, spam[3001:3800, ])
te <- predict(scaler, spam[3801:4601, ])
"""

data(spam)
foo <- sample(nrow(spam))
spam <- spam[foo,]
spam[,-58]<-scale(spam[,-58])
tr <- spam[1:3000, ]
va <- spam[3001:3800, ]
te <- spam[3801:4601, ] 

filter1 = ksvm(type~., data=tr, kenerl="rbfdot", kpar=list(sigma=0.05), C=0.5, scaled=FALSE)
filter2 = ksvm(type~., data=tr, kenerl="rbfdot", kpar=list(sigma=0.05), C=1, scaled=FALSE)
filter3 = ksvm(type~., data=tr, kenerl="rbfdot", kpar=list(sigma=0.05), C=5, scaled=FALSE)

f1_val = predict(filter1, va[, -58])
t <- table(f1_val, va[,58])
t
err1 = (t[1,2]+t[2,1])/sum(t)

f2_val = predict(filter2, va[, -58])
t <- table(f2_val, va[,58])
t
err2 = (t[1,2]+t[2,1])/sum(t)


f3_val = predict(filter3, va[, -58])
t <- table(f3_val, va[,58])
t
err3 = (t[1,2]+t[2,1])/sum(t)

cat(err1, err2, err3)
# Filter 2 has lowest validation error, C=1

opt_error = predict(filter2, te[,-58])
t <- table(opt_error, te[,58])
t
err_opt = (t[1,2]+t[2,1])/sum(t)
cat("Generalization error is: ", err_opt)

# Returns filter trained on all available data to user
filter_user = ksvm(type~., data=spam, kenerl="rbfdot", kpar=list(sigma=0.05), C=1, scaled=FALSE)
pred <- predict(filter_user, te[,-58])
t <- table(pred, te[,58])
t
err_opt = (t[1,2]+t[2,1])/sum(t)
cat("Generalization error for user is: ", err_opt)

# Purpose of parameter C is to account for the tradeoff between train and margin


