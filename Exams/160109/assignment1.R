library(tree)
library(glmnet)

setwd("~/TDDE01/exam_2016-01-09/")

crx = read.csv("../crx.csv")
#crx = crx[-2,]

set.seed(12345)

# dividing training och validation
size = dim(crx)[1]
ids = sample(1:size, floor(0.8*size))
train = crx[ids,]
test = crx[-ids,]

# normal tree model fit
fit = tree(Class ~ ., data = train)

png("fit.png")
plot(fit)
text(fit, pretty = 1)
dev.off()

# cross-validation of tree fit
cvtree = cv.tree(fit)

# fit the optimal tree
optimal_tree = prune.tree(fit, best=cvtree$size[which.min(cvtree$dev)])
summary(optimal_tree)

png("optimal_tree.png")
plot(optimal_tree)
text(optimal_tree, pretty = 1)
dev.off()

# looking at residuals
residual = resid(optimal)
hist(residual) # looks like some kind of normal dist

# starting with lasso
# categorical response -> logistic regression, two possible outcomes -> binomial family should work
x_train = model.matrix( ~ . -1, train[,-16])
response = train[,16]

# lasso model
lasso = glmnet(x_train, response, alpha = 1, family = "binomial")
plot(lasso, xvar = "lambda", label = T)

# cross-validation
set.seed(12345)
cv.lasso = cv.glmnet(x_train, response, family = "binomial", alpha = 1, type.measure = "deviance")
#cv.lasso = cv.glmnet(x_train, response, family = "binomial", alpha = 1, type.measure = "class")
plot(cv.lasso)

print(cv.lasso)
