data(spam)

index <- sample(1:dim(spam)[1])
spamtrain <- spam[index[1:floor(dim(spam)[1]/2)], ]
spamtest <- spam[index[((ceiling(dim(spam)[1]/2)) + 1):dim(spam)[1]], ]

## train a support vector machine
values = c(1,10,100)

for(v in values) {
  set.seed(1234567890)
  filter <- ksvm(type~.,data=spam,kernel="rbfdot", kpar=list(sigma=0.05),C=v,cross=2)
  print(v)
  print(cross(filter))
}
