library(e1071)
setwd("~/TDDE01/exam_2017-04-17")

data = read.csv("australian-crabs.csv")

blue = subset(data, species == "Blue")
orange = subset(data, species == "Orange")

plot(blue$CW, blue$BD, col="blue", xlab = "Caparace Width", ylab = "Body Depth")
points(orange$CW, orange$BD, col="orange")

# naive bayes classifier
fit = naiveBayes(species ~ CW+BD, data = data)
pred = predict(fit, newdata = data)

tab = table(data$species, pred)
print(tab) # confusion matrix
print(1-sum(diag(2)*tab)/sum(tab)) # misclassification rate
print(mean(data$species !=pred)) # simpler misclassification rate

# logistic regression
lg = glm(species ~ CW+BD, family = binomial, data = data)
lg.pred = predict(lg, newdata = data)

plot(lg.pred)

summary(lg.pred)


# principal components
res = prcomp(~CW+BD, data = data, scale = TRUE)

pcdata = data.frame(species = data$species, PC1 = res$x[,1], PC2 = res$x[,2])

# improved naive bayes classifier
fit2 = naiveBayes(species ~ .,  data = pcdata)
pred2 = predict(fit2, newdata = pcdata)

tab2 = table(pcdata$species, pred2)
print(tab2) # confusion matrix
print(1-sum(diag(2)*tab2)/sum(tab2)) # misclassification rate