################ 3 ############

######## Neural networks #########

library(neuralnet)


set.seed(1234567890)
Var <- runif(50, 0, 10)
tr <- data.frame(Var, Sin=sin(Var)) 
tr1 <- tr[1:25,] # Fold 1
tr2 <- tr[26:50,] # Fold 2



# Random initialization of the weights in the interval [-1, 1]
set.seed(1234567890)
winit <- runif(31, -1, 1)

nn1 <- neuralnet(Sin ~ Var, data = tr1, hidden = c(10), startweights = winit, threshold = 0.001)
fold1 <- predict(nn1, tr2)
mse1 <- mean((fold1 - tr2$Sin)**2)

nn2 <- neuralnet(Sin ~ Var, data = tr2, hidden = c(10), startweights = winit, threshold = 0.001)
fold2 <- predict(nn2, tr1)
mse2 <- mean((fold2 - tr1$Sin)**2)

mse1
mse2


############## SVM #############

library(kernlab)

data(spam)

c_values = c(1, 10, 100)
error = NULL

for(c in c_values) {
  set.seed(1234567890)
  filter <- ksvm(type~.,data=spam, kernel="rbfdot", kpar=list(sigma=0.05), C=c, scaled=TRUE, cross=2)
  error <- c(error, cross(filter))
}

error
c_optimal <- c_values[which.min(error)]
c_optimal


