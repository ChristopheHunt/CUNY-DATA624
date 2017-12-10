knitr::opts_chunk$set(echo = TRUE, message=FALSE, warning=FALSE, cache=TRUE)
library(missForest)
library(corrgram)
library(caret)
library(psych)
library(knitr)

dfBevMod <- read.csv("https://github.com/ChristopheHunt/CUNY-DATA624/raw/master/data/StudentData.csv", header = TRUE)
dfBevPred <- read.csv("https://raw.githubusercontent.com/ChristopheHunt/CUNY-DATA624/master/data/StudentEvaluation-%20TO%20PREDICT.csv", header =TRUE)
dim(dfBevMod)
colSums(is.na(dfBevMod))
#Visualizing the single categorical variable
barchart(dfBevMod[,1], col="Gold")
library(psych)
library(knitr)
table.desc <- describe(dfBevMod[,-1])
table.prep <- as.matrix(table.desc)
table.round <- round((table.prep), 2)
kable(table.round)
dfBevModH <- dfBevMod[2:ncol(dfBevMod)] #removing factor var
par(mfrow = c(3,5), cex = .5)
for(i in colnames(dfBevModH)){
hist(dfBevModH[,i], xlab = names(dfBevMod[i]),
  main = names(dfBevModH[i]), col="grey", ylab="")
}
BrandA <- dfBevMod[dfBevMod$Brand.Code == "A",]
BAM <- colMeans(BrandA[,2:ncol(BrandA)], na.rm = TRUE)
BrandB <- dfBevMod[dfBevMod$Brand.Code == "B",]
BBM <- colMeans(BrandB[,2:ncol(BrandB)], na.rm = TRUE)
BrandC <- dfBevMod[dfBevMod$Brand.Code == "C",]
BCM <- colMeans(BrandC[,2:ncol(BrandC)], na.rm = TRUE)
BrandD <- dfBevMod[dfBevMod$Brand.Code == "D",]
BDM <- colMeans(BrandD[,2:ncol(BrandD)], na.rm = TRUE)
BrandE <- dfBevMod[dfBevMod$Brand.Code == "",]
BEM <- colMeans(BrandE[,2:ncol(BrandE)], na.rm = TRUE)

combBrand <- cbind(BAM, BBM, BCM, BDM, BEM)
round(combBrand, 4)
par(mfrow = c(3,5), cex = .5)
for (i in colnames(dfBevModH)) {
 smoothScatter(dfBevModH[,i], main = names(dfBevModH[i]), ylab = "", 
   xlab = "", colramp = colorRampPalette(c("white", "red")))
 }
par(mfrow = c(3,5), cex = .5)
for(i in colnames(dfBevModH)){
boxplot(dfBevModH[,i], xlab = names(dfBevModH[i]),
  main = names(dfBevModH[i]), col="grey", ylab="")
}
PCA <- function(X) {
  Xpca <- prcomp(na.omit(X), center = T, scale. = T) 
  M <- as.matrix(na.omit(X)); R <- as.matrix(Xpca$rotation); score <- M %*% R
  print(list("Importance of Components" = summary(Xpca)$importance[ ,1:5], 
             "Rotation (Variable Loadings)" = Xpca$rotation[ ,1:5],
             "Correlation between X and PC" = cor(na.omit(X), score)[ ,1:5]))
  par(mfrow=c(2,3))
  barplot(Xpca$sdev^2, ylab = "Component Variance")
  barplot(cor(cbind(X)), ylab = "Correlations")
  barplot(Xpca$rotation, ylab = "Loadings")  
  biplot(Xpca); barplot(M); barplot(score)
}
PCA(dfBevModH)
#df_imputed$Brand.Code = NULL
dfBevMod$Brand.Code[dfBevMod$Brand.Code == ""] <- NA
dfBevMod$Brand.Code <- droplevels(dfBevMod$Brand.Code)

dfBevPred$Brand.Code[dfBevPred$Brand.Code == ""] <- NA
dfBevPred$Brand.Code <- droplevels(dfBevPred$Brand.Code)

#Recode categorical factor
dfBevMod$A <- ifelse(dfBevMod$Brand.Code == "A", 1, 0)
dfBevMod$B <- ifelse(dfBevMod$Brand.Code == "B", 1, 0)
dfBevMod$C <- ifelse(dfBevMod$Brand.Code == "C", 1, 0)
dfBevMod$D <- ifelse(dfBevMod$Brand.Code == "D", 1, 0)
dfBevMod$Brand.Code <- NULL

dfBevPred$A <- ifelse(dfBevPred$Brand.Code == "A", 1, 0)
dfBevPred$B <- ifelse(dfBevPred$Brand.Code == "B", 1, 0)
dfBevPred$C <- ifelse(dfBevPred$Brand.Code == "C", 1, 0)
dfBevPred$D <- ifelse(dfBevPred$Brand.Code == "D", 1, 0)
dfBevPred$Brand.Code <- NULL
library(tidyverse)
#dfImpMod = missForest(dfBevMod)
#dfImpMod$OOBerror #error rate looks good?
#dfModImp <- dfImpMod$ximp

#dfImpPred <- missForest(dfBevPred)
#dfPredImp <- dfImpPred$ximp
#dfPredImp$PH <- NA #redadding PH

#write.csv(dfModImp, "TrainImputeData.csv")
#write.csv(dfPredImp, "PredictImputeData.csv")

#Stored current imputation results on github to quicken knitr iterations
dfModImp <- read.csv("https://raw.githubusercontent.com/ChristopheHunt/CUNY-DATA624/master/data/TrainImputeData.csv") %>% dplyr::select(-X)
dfPredImp <- read.csv("https://raw.githubusercontent.com/ChristopheHunt/CUNY-DATA624/master/data/PredictImputeData.csv") %>% dplyr::select(-X)

dfModImpX <- dfModImp[,!(names(dfModImp) == "PH")]
dfModImpY <- dfModImp[, names(dfModImp) == "PH"]

dfPredImpX <- dfPredImp[,!(names(dfPredImp) == "PH")]
dfPredImpY <- dfPredImp[, names(dfPredImp) == "PH"]

#Spatial Sign outlier processing
dfModImpSsX <- spatialSign(dfModImpX)
dfPredImpSsX <- spatialSign(dfPredImpX)

#BoxCox Only
transModB <- preProcess(dfModImpSsX, method = "BoxCox") #transformed all 22 variables
dfModBX <- predict(transModB, dfModImpSsX)

transPredB <- preProcess(dfPredImpSsX, method = "BoxCox") #transformed 23 variables (should we use the Modeling model from above, instead of predicting model?)
dfPredBX <- predict(transPredB, dfPredImpSsX)

#BoxCox, Centering, and Scaling
transModBCS <- preProcess(dfModImpSsX, method = c("BoxCox", "center", "scale")) #22 BC, 35 centered, 35 scaled
dfModBCSX <- predict(transModBCS, dfModImpSsX)

transPredBCS <- preProcess(dfPredImpSsX, method = c("BoxCox", "center", "scale")) #23 BC, 35 centered, 35 scaled
dfPredBCSX <- predict(transPredBCS, dfPredImpSsX)

#BoxCox, Centering, Scaling, and PCA
transModBCSP <- preProcess(dfModImpSsX, method = c("BoxCox", "center", "scale", "pca")) #22 BC, 35 centered, 35 scaled
dfModBCSPX <- predict(transModBCSP, dfModImpSsX)

transPredBCSP <- preProcess(dfPredImpSsX, method = c("BoxCox", "center", "scale", "pca")) #23 BC, 35 centered, 35 scaled
dfPredBCSPX <- predict(transPredBCSP, dfPredImpSsX)
#corrgram(dfModBCSX, order=TRUE,
#         upper.panel=panel.cor, main="Correlation Matrix")
library(corrplot)
#install.packages("corrplot")
correlations <- cor(dfModBCSX)
corrplot(correlations, order = "hclust", tl.cex = 0.55)
hc = findCorrelation(correlations, cutoff=0.75)
length(hc) #18 vars

#Reducing
dfModBCSRX = dfModBCSX[,-c(hc)] #Box-Cox, Center, Scale
dfPredBCSRX = dfPredBCSX[,-c(hc)]

dfModBRX = dfModBX[,-c(hc)] #Box-Cox
dfPredBRX = dfPredBX[,-c(hc)]

dfModSRX = dfModImpSsX[,-c(hc)] #Only Spatial Sign
dfPredSRX = dfPredImpSsX[,-c(hc)]
set.seed(2017)
n75 <- floor(0.75 * nrow(dfBevMod)) #75$ of sample size
n <- sample(seq_len(nrow(dfBevMod)), size = n75)

#Box-Cox, Center, Scale
dfTrainBCSX <- dfModBCSRX[n,]
dfTestBCSX <- dfModBCSRX[-n,]

#Box-Cox
dfTrainBX <- dfModBRX[n,]
dfTestBX <- dfModBRX[-n,]

#Only Spatial Sign
dfTrainX <- dfModSRX[n,]
dfTestX <- dfModSRX[-n,]

#Response variable
dfTrainY <- dfModImpY[n]
dfTestY <- dfModImpY[-n]

knitr::opts_chunk$set(echo = TRUE)
library(caret)
library(mlbench)
library(MASS)
library(AppliedPredictiveModeling)
library(lars)
library(pls)
library(elasticnet)
library(rpart)
library(e1071)
set.seed(1234)
plsFit = plsr(PH ~ ., data=dfModImp, validation="CV")
pls.pred = predict(plsFit, dfPredImp[1:5, ], ncomp=1:2)

pls.pred

validationplot(plsFit, val.type="RMSEP")
validationplot(plsFit, val.type="R2")

pls.RMSEP = RMSEP(plsFit, estimate="CV")
plot(pls.RMSEP, main="RMSEP PLS PH", xlab="Components")
min_comp = which.min(pls.RMSEP$val)
points(min_comp, min(pls.RMSEP$val), pch=1, col="red", cex=1.5)

min_comp

plot(plsFit, ncomp=30, asp=1, line=TRUE)

pls.pred2 = predict(plsFit, dfPredImp, ncomp=30)
summary(pls.pred2)
trainControl <- trainControl(method = "cv", number = 10)

#GLM
set.seed(1234)
fit.glm <- train(PH~., data=dfModImp, method="glm", metric="RMSE", trControl=trainControl,
           tuneLength = 5, preProc = c("center", "scale"))
fit.glm

#SVM
set.seed(1234)
fit.svm <- train(PH~., data=dfModImp, method="svmLinear3", metric="RMSE", trControl=trainControl,                               tuneLength = 5, svr_eps = .1, preProc = c("center", "scale"))
fit.svm

#KNN
set.seed(1234)
fit.knn <- train(PH~., data=dfModImp, method="knn", metric="RMSE", trControl=trainControl,
          tuneLength = 5, preProc = c("center", "scale"))
fit.knn

#CART
set.seed(1234)
fit.cart <- train(PH~., data=dfModImp, method="rpart", metric="RMSE", trControl=trainControl,
          tuneLength = 5, preProc = c("center", "scale"))
fit.cart

#Bagged CART
set.seed(1234)
fit.bagcart <- train(PH~., data=dfModImp, method="treebag", metric="RMSE", trControl=trainControl,
          tuneLength = 5, preProc = c("center", "scale"))
fit.bagcart

#Compare results of the algorithms we ran
results <- resamples(list(GLM=fit.glm, SVM=fit.svm, KNN=fit.knn, CART=fit.cart, BaggedCart=fit.bagcart))

summary(results)

library(randomForest)

set.seed(1234)

rf <- function(df,y){
       fitControl <- trainControl(method = "cv",
       number = 10, #5folds)
                           
rfgrid <- expand.grid(interaction.depth = 2,
            n.trees = 500,
            shrinkage = 0.1,
            n.minobsinnode = 10))
  
return(train(df, y, 
             method = "randomForest", 
             tuneGrid = rfgrid, 
             trControl = fitControl))
}

rf.fit.bcsx <- randomForest(dfTrainBCSX, dfTrainY)
rf.fit.bx   <- randomForest(dfTrainBX, dfTrainY)
rf.fit.x    <- randomForest(dfTrainX,    dfTrainY)
list(RMSE_RF.BCSX = RMSE(predict(rf.fit.bcsx, dfTestBCSX), dfTestY),
     RMSE_RF.BCS  = RMSE(predict(rf.fit.bx, dfTestBX), dfTestY),
     RMSE_RF.X    = RMSE(predict(rf.fit.x, dfTestX), dfTestY))
rf.fit.bcsx
plot(rf.fit.bx)

library(caret)

set.seed(1234)

gbm <- function(df,y){
       fitControl <- trainControl(method = "cv",
       number = 10)
                           
gbmgrid <- expand.grid(interaction.depth = 2,
            n.trees = 500,
            shrinkage = 0.1,
            n.minobsinnode = 10)
  
return(train(df, y, 
             method = "gbm", 
             tuneGrid = gbmgrid, 
             trControl = fitControl,
             verbose = FALSE))
}

gbm.fit.bcsx <- gbm(dfTrainBCSX, dfTrainY)
gbm.fit.bx   <- gbm(dfTrainBX, dfTrainY)
gbm.fit.x    <- gbm(dfTrainX,    dfTrainY)
list(RMSE_GBM.BCSX = RMSE(predict(gbm.fit.bcsx, dfTestBCSX), dfTestY),
     RMSE_GBM.BCS  = RMSE(predict(gbm.fit.bx, dfTestBX), dfTestY),
     RMSE_GBM.X    = RMSE(predict(gbm.fit.x, dfTestX), dfTestY))
summary(gbm.fit.bx, digit=3)
list(RMSE_GBM = RMSE(predict(gbm.fit.bcsx, dfTestBCSX), dfTestY),
     RMSE_RF  = RMSE(predict(rf.fit.x, dfTestX), dfTestY))
library(caret)
library(tidyverse)
library(Metrics)
set.seed(1234)

nnet <- function(df, y){
  
              fitControl <- trainControl(method = "cv", 
                                         number = 3, 
                                         returnResamp = "all")
              
              nnetGrid <- expand.grid(.decay = c(0, 0.01, .1, .5),
                                      .size = c(5:15),
                                      .bag = FALSE)
    
                 return(train(df, y, 
                              method = "avNNet", 
                              tuneGrid = nnetGrid,
                              trControl = fitControl,
                              trace = FALSE,
                              linout = TRUE))
}

nnet.fit.bcsx <- nnet(dfTrainBCSX, dfTrainY)
nnet.fit.bx   <- nnet(dfTrainBX,   dfTrainY)
nnet.fit.x    <- nnet(dfTrainX,    dfTrainY)
list(RMSE_NNET.BCSX = RMSE(predict(nnet.fit.bcsx, dfTestBCSX), dfTestY),
     RMSE_NNET.BCS  = RMSE(predict(nnet.fit.bx,   dfTestBX), dfTestY),
     RMSE_NNET.X    = RMSE(predict(nnet.fit.x,    dfTestX), dfTestY))
nnet.fit.bcsx$finalModel
varImp(nnet.fit.bcsx)
plot(nnet.fit.bcsx)
library(caret)

set.seed(1234)

mars <- function(df, y){
           
          fitControl <- trainControl(method = "cv", 
                                     number = 3, 
                                     returnResamp = "all")

          MARSGrid <- expand.grid(degree= 3:5, 
                                  nprune = seq(20,60,20))
          
          return(train(df, y, 
                 method = "earth", 
                 tuneGrid = MARSGrid,
                 trControl = fitControl))
         }

mars.fit.bcsx <- mars(dfTrainBCSX, dfTrainY)
mars.fit.bx   <- mars(dfTrainBX,   dfTrainY)
mars.fit.x    <- mars(dfTrainX,    dfTrainY)
list(RMSE_MARS.BCSX = RMSE(predict(mars.fit.bcsx, dfTestBCSX), dfTestY),
     RMSE_MARS.BCS  = RMSE(predict(mars.fit.bx,   dfTestBX), dfTestY),
     RMSE_MARS.X    = RMSE(predict(mars.fit.x,    dfTestX), dfTestY))
mars.fit.bcsx$finalModel
plot(mars.fit.bcsx)
varImp(mars.fit.bcsx)
list(RMSE_MARS = RMSE(predict(mars.fit.bcsx, dfTestBCSX), dfTestY),
     RMSE_NNET = RMSE(predict(nnet.fit.bcsx, dfTestBCSX), dfTestY))
library(tidyverse)
library(xlsx)
dfPredImp$PH <- predict(rf.fit.x, dfPredBX)
dfPredImp <- dfPredImp %>%  dplyr::select(PH, everything())
write.xlsx(dfPredImp, "./data/StudentEvaluation- TO PREDICT wPredictions.xlsx", row.names = FALSE)
sessionInfo()
## NA
