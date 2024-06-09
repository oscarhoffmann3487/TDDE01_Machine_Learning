#General notes
#Help functions
help("function")
help.start()
help.search("expression")
example("function")

#Topic 1
#data reading/writing
#E1 - Create data frame birth by reading birthstatistics.csv into R and by using read.csv. Note that these data has headers.
birth = read.csv("birthstatistics.csv")

#E2 - Create data frame blog by reading blogData_test.csv into R and by using read.csv. Note that headers (column names) are missing.
blog = read.csv("blogData_test.csv", header = FALSE)

#E3 - Create data frame tecator by reading tecator.xls into R by using readxl package.
library(readxl)
data = read_excel("tecator.xls")

#E4 - Save tecator to tecator.csv with write.csv and make sure that row names are not saved
write.csv(data, file="tecator.csv", row.names = FALSE)

#Basic data manipulation
#E1 - Convert tecator to a data frame and call it tecator1
tecator1 = as.data.frame(data)

#E2 - Change row names in tecator1 to the values of Sample column plus 10
rownames(tecator1) = tecator1$Sample + 10

#E3 - Change column name in tecator1 from Sample to ID
colnames(tecator1)[1] ="ID"

#E4 - Extract rows in tecator1 such that Channel1 > 3 and Channel2 > 3 and columns between number 5 and number 8
tecator1[tecator1$Channel1 > 3 & tecator1$Channel2 > 3, 5:8]

#E5 - Remove column ID in tecator1 (two options)
tecator1$ID=c()
tecator1 = tecator1[, -1]

#E6 - Update tecator1 by dividing its all Channel columns with their respective means per column (two options)
for (i in 1:100) {
  # Assuming the channel columns are named as "Channel1", "Channel2", ..., "Channel100"
  channel_col_name = paste("Channel", i, sep="")
  
  # Calculate the mean of the i-th Channel column
  divider = mean(tecator1[[channel_col_name]])
  
  # Divide the i-th Channel column by its mean
  tecator1[[channel_col_name]] = tecator1[[channel_col_name]] / divider
}
#Oleg
library(stringr)
index=str_which(colnames(tecator1), "Channel")
tecatorChannel=tecator1[,index]
means=colMeans(tecatorChannel)
tecator1[,index]=tecator1[,index]/matrix(means, nrow=nrow(tecatorChannel), ncol=ncol(tecatorChannel), byrow=TRUE)

#E7 - Compute a sum of squares for each row between 1 and 5 in tecator1 without writing loops and make it as a matrix with one column
m = matrix(c(rowSums(tecator1[1,]^2), rowSums(tecator1[2,]^2), rowSums(tecator1[3,]^2), rowSums(tecator1[4,]^2), rowSums(tecator1[5,]^2)), nrow = 5, ncol = 1)
print(m)
#Oleg
sumsq=apply(tecator1[1:5,], MARGIN = 1, FUN=function(x) return(sum(x^2)) )
tecator2=matrix(sumsq, ncol=1)
#E8 - Extract X as all columns except of columns 101-103 in tecator1, y as column Fat and compute (XTX)−1XTy
X=as.matrix(tecator1[,1:100])
y=as.matrix(tecator1$Fat)
res=solve(t(X)%*%X)%*%t(X)%*%y
res
#Oleg
X=as.matrix(tecator1[,-c(101, 102, 103)]) #can be written more efficiently as -(101:103)
y=as.matrix(tecator1[,"Fat", drop=F]) #keep it as a matrix, don't reduce dimension.
result=solve(t(X)%*%X, t(X)%*%y)
result
#E9 - Use column Channel1 in tecator1 to compute new column ChannelX which is a factor with the following levels: “high” if Channel1>1 and “low” otherwise
tecator1$ChannelX=as.factor(ifelse(tecator1$Channel1>1, "high", "low"))

#E10 - Write a for loop that computes regressions Fat as function of Channeli,i=1,...100 and then stores the intercepts into vector Intercepts. Print Intercepts.
channels = tecator1[,1:100]
intercepts = numeric(length = ncol(channels))
for (i in 1:ncol(channels)) {
  dataset= data.frame(channels[,i])
  fit1 = lm(tecator1$Fat ~., data=dataset)
  intercepts[i] = coef(fit1)[1]
}
print(intercepts)

#Oleg
Intercepts=numeric(100)
for (i in 1:length(Intercepts)){
  regr=lm(formula=paste("Fat~Channel", i, sep=""), data=tecator1)
  Intercepts[i]=coef(regr)[1]
}
print(Intercepts)

#E11 - Given equation y=5x+1, plot this dependence for x between 1 and 3
x = c(1,3)
y=5*x+1
plot(x,y, type="l")

#Data manipulation: dplyr and tidyr
#E1 - Convert data set birth to a tibble birth1
library(dplyr)
library(tidyr)
birth1=tibble(birth)

#E2 - Select only columns X2002-X2020 from birth1 and save into birth2
birth2 = birth1%>%select(X2002:X2020)

#E3 - Create a new variable Status in birth1 that is equal to “Yes” if the record says “born in Sweden with two parents born in Sweden” and “No” otherwise
birth1$Status = ifelse(birth1$foreign.Swedish.background == "born in Sweden with two parents born in Sweden", "Yes", "No")

#with dplyr
birth1=birth1%>%mutate(Status=ifelse(foreign.Swedish.background=="born in Sweden with two parents born in Sweden", "Yes", "No"))

#E4 - Count the amount of rows in birth 1 corresponding to various combinations of sex and region
birth1%>%count(sex,region)

#E5 - Assuming that X2002-X2020 in birth1 show amounts of persons born respective year, compute total amount of people born these years irrespective gender, given Status and region. Save the result into birth3
birth3=birth1%>%select(-sex,- foreign.Swedish.background)%>%group_by(region, Status)%>%summarise_all(sum)%>%ungroup()
birth3

#E6 - By using birth3, compute percentage of people in 2002 having Status=Yes in different counties. Report a table with column region and Percentage sorted by Percentage.
birth4=birth3%>%
  group_by(region)%>%
  mutate(Percentage=X2002/sum(X2002)*100)%>%
  filter(Status=="Yes")%>%
  select(region, Percentage)%>%
  ungroup()%>%
  arrange(Percentage)

birth4
#E7 - By using birth1, transform the table to a long format: make sure that years are shown in column Year and values from the respective X2002-X2020 are stored in column Born. Make sure also that Year values show years as numerical values and store the table as birth5.
birth5 = birth1 %>% group_by(region, sex, foreign.Swedish.background, Status) %>% pivot_longer(X2002:X2020, names_to = "Year", values_to = "Born") %>% mutate(Year=as.numeric(stringr::str_remove(Year, "X")))

#E8 - By using birth5, transform the table to wide format: make sure that years are shown as separate columns and their corresponding values are given by Born. Columns should be named as “Y_2002” for example.
birth6 = birth5 %>% group_by(region, sex, foreign.Swedish.background, Status)%>% pivot_wider(names_from = Year, values_from = Born, names_prefix = "Y_")

#E9 - By using blog data, filter out columns that have zeroes everywhere.
blogS=tibble(blog)%>%select_if(function(x) !all(x==0))
blogS
