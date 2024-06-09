df=read.csv("adult.csv", stringsAsFactors = T)
n=dim(df)[1]
set.seed(12345) 
id=sample(1:n, floor(n*0.6)) 
train=df[id,] 

id1=setdiff(1:n, id)
set.seed(12345) 
id2=sample(id1, floor(n*0.2)) 
valid=df[id2,]

id3=setdiff(id1,id2)
test=df[id3,] 

library(tree)
fit=tree(C13~.-C12, data=train)
summary(fit)
nleaves=8


trainScore=rep(0,nleaves)
testScore=rep(0,nleaves)
for(i in 2:nleaves) {
  print(i)
  prunedTree=prune.tree(fit,best=i)
  pred=predict(prunedTree, newdata=valid,
               type="tree")
  trainScore[i]=deviance(prunedTree)
  testScore[i]=deviance(pred)
}
plot(2:nleaves, trainScore[2:nleaves], type="b", col="red",
     ylim=c(4000,20000))
points(2:nleaves, testScore[2:nleaves], type="b", col="blue")


which.min(testScore[-1])+1
lT=prune.tree(fit, best=which.min(testScore[-1]+1))
plot(lT)
text(lT)


#2

pis=seq(0.1, 0.9, 0.1)
ps=length(pis)
F1=numeric(ps)
acc=numeric(ps)

Pred=predict(lT, newdata=test, type="vector")
for (i in 1:ps){
  Pr=ifelse(Pred[,2]>pis[i], ">50K", "<=50K")
  tab=table(test$C13, Pr)
  F1[i]=tab[2,2]/(tab[2,2]+0.5*(tab[1,2]+tab[2,1]))
  acc[i]=(tab[1,1]+tab[2,2])/(sum(tab))
}
rbind(pis,acc,F1)

#3

covariates=as.matrix(train[, c(1,9,10)])
response=train[[11]]

library(glmnet)
model=cv.glmnet(as.matrix(covariates), response, alpha=1,family="gaussian")
model$lambda.min
plot(model)
coef(model, s="lambda.min")
