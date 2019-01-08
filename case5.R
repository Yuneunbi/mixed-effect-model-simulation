library(nnet)
library(dplyr)
library(tidyr)
require(Matrix)
require(MASS)
require(mvtnorm)
library(vcrpart)
library(neuralnet)
library(stringr)
library(ggplot2)

# rm(list=ls())
# gc()
load("C:/Users/beom/Dropbox/ebyun/multinomial/final simualation/makeBase.fix.rand.2.RData")

maxidx <- function(arr) {
  return(which(arr == max(arr)))
}


##############
p <- 3  # 계수
k <- 3 # category
nt <- 1 # 각 시행 수 
r <- 6 # 각 그룹별 데이터 수
g <- 50 # 그룹 수
N = r*g



# 
sigb <- diag(c(1.5,2.5))
lo <- matrix(c(1,0.2,0.2,1),2,2)


beta <- matrix(c(-0.05,-2.3,-1,0.3,1,3.1),3,2)

V<- sigb %*% lo %*% sigb   # 임의효과 b에 대해서 하나의 집단이 가지는 공분산

f <- gl(g,r)
Z <- t(as(f,Class="sparseMatrix"))



# set.seed(132221721)

B <- 100
sim.result <- matrix(0,B,7)

for (iter in 1:B) {
  
  data.list<-makeBase.fix.rand.2(p=p,k=k,nt=nt,r=r,g=g,beta=beta,V=V,f=f,Z=Z)
  
  dat1 <-  data.list[[2]]
  dat2 <-  data.list[[4]][,c(1,2,3,4,6,7,8)]
  
  dat1 <- arrange(dat1,id)
  dat2 <- arrange(dat2,id)
  names(dat2)[c(2,3,4)] <- c("c1", "c2", "c3")
  
  dat <- cbind(dat1,dat2[,c("c1", "c2", "c3")])
  
  
  # make dummy variable for factor
  facdat <- as.data.frame(model.matrix(~dat1$factor))[,-1]
  fnames <- sapply("f",paste,1:(g-1),sep="",simplify = F)$f
  names(facdat) <- fnames 
  dat <- cbind(dat,facdat)
  
  dat$choice2 <- relevel(dat$choice,ref='3')
  idx <-  sample(dat$id, round(g*r*0.9), replace=F)
  
  traindat <- dat[dat$id %in% idx,]
  testdat <- dat[!(dat$id %in% idx),]
  testdat <- arrange(testdat,id)
  
  target <- testdat$choice
  
  testdat21 <- testdat[,c('id','X2','X3')]
  testdat22 <- testdat[,c('id','X2','X3',fnames)]
  
  nnet.ce <- multinom(choice2 ~ X2+X3,data=traindat)
  for1 <- as.formula(paste("choice2 ~ X2+X3+", str_c(fnames, collapse = "+")))
  nnet.ce2 <- multinom(for1,data=traindat)
  #summary(nnet.ce)
  lmm.f <- olmm(choice~ce(X2)+ce(X3)+re(1|factor),data=traindat,family=baseline())
  lmm.f2 <- olmm(choice~ce(X2)+ce(X3),data=traindat,family=baseline())
  #summary(lmm.f2)
  
  ntest <- length(target)
  
  a1 <- predict(lmm.f2, newdata = testdat, type = "class", ranef = FALSE)
  a2 <- predict(lmm.f, newdata = testdat, type = "class", ranef = FALSE)
  a3 <- predict(lmm.f, newdata = testdat, type = "class", ranef = TRUE)
  #a21 <- predict(lmm.f, newdata = testdat, type = "prob", ranef = FALSE)
  #a31 <- predict(lmm.f, newdata = testdat, type = "prob", ranef = TRUE)
  b1 <- predict(nnet.ce, newdata = testdat21, type = "class")
  b2 <- predict(nnet.ce2, newdata = testdat22, type = "class")
  
  for (i in 1:100) {
    nn <- neuralnet(c1+c2+c3 ~ X2+X3, data=traindat, hidden=c(3))
    for2 <- as.formula(paste("c1+c2+c3 ~ X2+X3+", str_c(fnames, collapse = "+")))
    nn2 <- neuralnet(for2, data=traindat, hidden=c(3))
    #plot(nn)
    
    mypredict <- try(neuralnet::compute(nn, testdat21[,-1])$net.result)
    mypredict2 <- try(neuralnet::compute(nn2, testdat22[,-1] )$net.result)
    
    if (inherits(mypredict, "try-error") | inherits(mypredict2, "try-error")) {
      cat("re-try", i, "\n")
      next
    } else {
      break
    }
  }
  
  c1 <- apply(mypredict, c(1), maxidx)
  c2 <- apply(mypredict2, c(1), maxidx)
  
  
  sim.result[iter,1] <- sum(factor(a1,levels = c(1,2,3), ordered = FALSE) == target)/ntest
  sim.result[iter,2] <- sum(factor(a2,levels = c(1,2,3), ordered = FALSE) == target)/ntest
  sim.result[iter,3] <- sum(factor(a3,levels = c(1,2,3), ordered = FALSE)  == target)/ntest
  sim.result[iter,4] <- sum(factor(b1,levels = c(1,2,3))  == target)/ntest
  sim.result[iter,5] <- sum(factor(b2,levels = c(1,2,3))  == target)/ntest
  sim.result[iter,6] <- sum(factor(c1,levels = c(1,2,3))  == target)/ntest
  sim.result[iter,7] <- sum(factor(c2,levels = c(1,2,3))  == target)/ntest
  
}  



case5.result <- sim.result
round(case5.result,4)
round(apply(case5.result,2,mean),4)
round(apply(case5.result,2,sd),4)/10
# system.time(lmm.f <- olmm(choice~ce(X2)+ce(X3)+re(1|factor),data=traindat,family=baseline()))
# system.time(nn2 <- neuralnet(for2, data=traindat, hidden=c(3)))
# ggplot(traindat, aes(x = X2, y = X3)) + geom_point(aes(color = choice))
# ggplot(testdat, aes(x = X2, y = X3)) + geom_point(aes(color = choice))

save(p,k,r,g,nt,N,sigb,lo,case5.result, file ="C:/Users/beom/Dropbox/ebyun/
     multinomial/final simualation/case5.RData")

# testidx <- which(1:length(iris[,1])%%5 == 0)
# iristrain <- iris[-testidx,]
# iristest <- iris[testidx,]
# nnet_iristrain <-iristrain
# #Binarize the categorical output
# nnet_iristrain <- cbind(nnet_iristrain, 
#                             iristrain$Species == 'setosa')
# nnet_iristrain <- cbind(nnet_iristrain,
#                           iristrain$Species == 'versicolor')
# nnet_iristrain <- cbind(nnet_iristrain, 
#                           iristrain$Species == 'virginica')
# names(nnet_iristrain)[6] <- 'setosa'
# names(nnet_iristrain)[7] <- 'versicolor'
# names(nnet_iristrain)[8] <- 'virginica'
# nn <- neuralnet(setosa+versicolor+virginica ~ 
#                   Sepal.Length+Sepal.Width
#                 +Petal.Length
#                 +Petal.Width,
#                 data=nnet_iristrain, 
#                 hidden=c(3))
# plot(nn)
# mypredict <- compute(nn, iristest[-5])$net.result
# #==================================================================


