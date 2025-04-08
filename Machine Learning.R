library(readr)
library(dplyr)
library(caret)

data <- read.csv("Invistico_Airline.csv")
getwd()

data$Churn <- ifelse(data$Customer.Type == 'disloyal Customer', 1, 0)
data$Customer.Type <- NULL
data$Churn <- factor(data$Churn, levels = c(1,0), labels = c('Churn','Loyal'))

lista <- c("Seat.comfort","Departure.Arrival.time.convenient","Food.and.drink","Gate.location",
           "Inflight.wifi.service","Inflight.entertainment","Online.support","Ease.of.Online.booking",
           "On.board.service","Leg.room.service","Baggage.handling","Checkin.service","Cleanliness", 
           "Online.boarding")

for (elemento in lista){
  data[data[,elemento] == 0, elemento] = 'Cattivo'
  data[data[,elemento] == 1, elemento] = 'Cattivo'
  data[data[,elemento] == 2, elemento] = 'Medio'
  data[data[,elemento] == 3, elemento] = 'Medio'
  data[data[,elemento] == 4, elemento] = 'Buono'
  data[data[,elemento] == 5, elemento] = 'Buono'
  data[,elemento] = as.factor(data[,elemento])
}

data$Gender = as.factor(data$Gender)
data$satisfaction = as.factor(data$satisfaction)
data$Type.of.Travel = as.factor(data$Type.of.Travel)
data$Class = as.factor(data$Class)

#PREPROCESSING####
#1. implementazione na con MICE
sapply(data, function(x)(sum(is.na(x))))
library(mice)

data2 <- data[,-23]
Churn <- data$Churn

tempData <- mice(data2, m = 5, maxit = 50, meth = 'pmm', seed = 500)  
completedData <- complete(tempData,1)

completedData <- cbind(completedData, Churn)

write.csv(completedData, "completedData.csv")
completedData <- read_csv("completedData.csv")
completedData<-completedData%>%select(-'...1')

#2. collinearità e connessione
numeric <- sapply(completedData, function(x) is.numeric(x))
numeric <- completedData[, numeric]
str(numeric)


R=cor(numeric)
R
correlatedPredictors = findCorrelation(R, cutoff = 0.95, names = TRUE)
correlatedPredictors #Arrival.Delay.in.Minutes


# 3. predittori con varianza zero e varianza vicino a zero 
nzv = nearZeroVar(completedData, saveMetrics = TRUE)
nzv #Departure.Delay.in.Minutes e Arrival.Delay.in.Minutes 

completedData$Departure.Delay <- ifelse(completedData$Departure.Delay.in.Minutes > 0, 'delay', 'in time')
completedData$Departure.Delay.in.Minutes <- NULL
completedData$Arrival.Delay.in.Minutes <- NULL
completedData$Departure.Delay <- as.factor(completedData$Departure.Delay)

table(completedData$Departure.Delay)/nrow(completedData)

nzv = nearZeroVar(completedData, saveMetrics = TRUE)
nzv #no zero-variance sulla variabile dummizzata

#4. connessione
str(completedData)

categorical_vars <- c("satisfaction", "Gender", "Type.of.Travel", "Class", 
                      "Seat.comfort", "Departure.Arrival.time.convenient",
                      "Food.and.drink", "Gate.location", "Inflight.wifi.service",
                      "Inflight.entertainment", "Online.support", "Ease.of.Online.booking",
                      "On.board.service", "Leg.room.service", "Baggage.handling",
                      "Checkin.service", "Cleanliness", "Online.boarding", 
                      "Churn")
completedData[categorical_vars] <- lapply(completedData[categorical_vars], as.factor)
str(completedData)

nfactorial <- sapply(completedData, function(x) is.factor(x))
nfactorial
factorial <- completedData[,nfactorial]
factorial <- factorial[,-19]
sapply(factorial, length)
factorial <- as.data.frame(factorial)


library(plyr)
combos <- combn(ncol(factorial),2)
adply(combos, 2, function(x) {
  test <- chisq.test(factorial[, x[1]], factorial[, x[2]])
  tab  <- table(factorial[, x[1]], factorial[, x[2]])
  out <- data.frame("Row" = colnames(factorial)[x[1]]
                    , "Column" = colnames(factorial[x[2]])
                    , "Chi.Square norm"  =round(test$statistic/
                                                  (sum(table(factorial[,x[1]], factorial[,x[2]]))* 
                                                     min(length(unique(factorial[,x[1]]))-1 , 
                                                         length(unique(factorial[,x[2]]))-1)), 3) 
  )
  return(out)
}) 

#niente sopra 0.8 quindi tutto okay

#Training sample####
set.seed(107)
#Dati di score
Scoreindex <-  createDataPartition(y = completedData$Churn, p = .10, list = FALSE)

score <- completedData[Scoreindex, -21]
completedData = completedData[-Scoreindex,]

# create a random index for taking 75% of data stratified by target
Trainindex <- createDataPartition(y = completedData$Churn, p = .75, list = FALSE)

train <- completedData[Trainindex,]

test  <- completedData[-Trainindex,]

train2 <- train[sample(nrow(train), 10000),]

table(train$Churn)/nrow(train)
table(completedData$Churn)/nrow(completedData)

#1.GLM preprocessato
set.seed(1)
control <- trainControl(method = "cv", number = 10, classProbs = T,
                        summaryFunction = twoClassSummary)
glm = train(Churn~., data = train, method = "glm", trControl = control, 
            metric = 'Sens', tuneLength = 5, trace = TRUE, na.action = na.pass)
glm #ROC 0.94, Sens 0.67, Spec 0.95

confusionMatrix(glm) #accuracy 0.9015

#2.TREE
set.seed(1)
control = trainControl(method = "cv", number=10, search="grid", classProbs = TRUE, 
                       summaryFunction=twoClassSummary)
tree <- train(Churn~.,data=train, method = "rpart",tuneLength = 10,
              trControl = control, metric = 'Sens')
tree

#check migliore accuracy con cp di Mallows
confusionMatrix(tree) #accuracy 0.9215
varImp(object = tree)
plot(varImp(object = tree), main = "train tuned - Variable Importance")
vi = as.data.frame(tree$finalModel$variable.importance)
vi
viname = row.names(vi)
viname

#10 variabili più importanti
importanti <- c('Age', 'Type.of.Travel', 'Class', 'satisfaction', 'Flight.Distance', 
                'Seat.comfort','Inflight.entertainment', 'Food.and.drink', 
                'Ease.of.Online.booking', 'Cleanliness')

mod <- c(importanti, 'Churn')

train_model <- train[,mod]


#3. k-nearest neightbour
set.seed(1)
control = trainControl(method="cv", number = 10, classProbs = T,
                       summaryFunction=twoClassSummary)
grid = expand.grid(k=seq(5,20,3)) #prova k da 5 a 20 spostandosi di 3 per volta
grid
knn = train(Churn~.,
            data=train_model,method = "knn",
            trControl = control, metric = 'Sens', tuneLength=5, na.action = na.pass,
            tuneGrid=grid, preProcess=c("scale","corr"))
knn 
plot(knn)
confusionMatrix(knn) #accuracy 0.9583

#4.Lasso
set.seed(1)
control = trainControl(method="cv", number = 10, classProbs = T,
                       summaryFunction=twoClassSummary)
grid = expand.grid(.alpha=1,.lambda=seq(0, 1, by = 0.01))
lasso = train(Churn~.,
              data=train,method = "glmnet",
              trControl = control, metric = 'Sens', tuneLength=5, na.action = na.pass,
              tuneGrid=grid, preProcess=c('center',"scale"))
lasso
plot(lasso)
confusionMatrix(lasso) #accuracy 0.9




#7.RANDOM FOREST
library(randomForest)
set.seed(1234)
control = trainControl(method = "cv", number=10, classProbs=TRUE,
                       summaryFunction = twoClassSummary)
rf=train(Churn~., data = train2 , method = "rf", trControl = control, 
         metric = 'Sens', tuneLength = 5)
rf
confusionMatrix(rf) #accuracy 0.9659


rf_model <- randomForest(Churn ~ ., data = train2, ntree = 100, 
                         mtry = sqrt(ncol(train2) - 1), nodesize = 5)

rf_model

importanza_variabili <- importance(rf_model)
importanza_variabili
varImpPlot(rf_model)
plot(rf)

install.packages("gbm")
library(gbm)

#8.GRADIENT BOOSTING
set.seed(1234) 
control = trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, 
                       classProbs = TRUE, savePrediction = TRUE)
gradient_boost <- train(Churn ~ ., data = train2, method = "gbm", trControl = control, 
                        metric = 'Sens', verbose = FALSE) 
gradient_boost
plot(gradient_boost)
confusionMatrix(gradient_boost) #accuracy 0.9519

#9.NEURAL NETWORKS
#BORUTA
install.packages("Boruta")
library(Boruta)
set.seed(123)
boruta.train <- Boruta(Churn~., data = train2, doTrace = 1)
#rejected Departure.Delay
plot(boruta.train, xlab = "features", xaxt = "n", ylab="MDI")
print(boruta.train)

unique(boruta.train$Chrurn)

#Boruta performed 16 iterations:
#20 attributes confirmed important
#1 attributes confirmed unimportant: Departure.Delay;

boruta.metrics <- attStats(boruta.train)
head(boruta.metrics)
table(boruta.metrics$decision)
rm(boruta_selected)
rm(boruta.train)
rm(boruta.metrics)
#eliminazione variabili rejected da boruta
boruta_selected <- train2[,-22]
dim(boruta_selected)




exists("boruta_selected")
colnames(boruta_selected)
colnames(train2)  # Sostituisci con il nome corretto del dataset originale
boruta_selected$Churn <- train2$Churn
boruta_selected$Churn <- as.factor(boruta_selected$Churn)
colnames(boruta_selected)
str(boruta_selected$Churn)


library(nnet)
nn = train(Churn ~., data = boruta_selected,
           method = "nnet",preProcess = c("center", "scale", "corr", "nzv"), 
           tuneLength = 5, trControl = control, metric = 'Sens',trace = TRUE,
           maxit = 300)
nn
plot(nn)
confusionMatrix(nn) #accuracy 0.9616

#Check overfitting via accuracy
overf_mod <- data.frame(modello = c('glm','tree','knn','lasso','rf','gb','nn'),
                        acc_train = rep(0,7), acc_valid = rep(0,7), perc_diff = rep(0,7),
                        overfitting = rep(FALSE, 7))

lista_mod <- list(glm = glm, tree = tree, knn = knn, lasso = lasso,
                  rf = rf, gb = gradient_boost, nn = nn)

i <- 0
for (mod in lista_mod){
  i <- i+1
  pred <- predict(mod, newdata = test, type = 'raw')
  prova <- cbind(test, pred)
  acct <- confusionMatrix(mod)$table
  accv <- table(pred = prova$pred, ref = prova$Churn)
  
  overf_mod$acc_train[i] = (acct[1,1] + acct[2,2]) / 100
  overf_mod$acc_valid[i] = (accv[1,1] + accv[2,2]) / nrow(prova)
  overf_mod$perc_diff[i] = round((overf_mod$acc_train[i] - overf_mod$acc_valid[i]) / overf_mod$acc_train[i], 3)
  overf_mod$overfitting[i] = ifelse(overf_mod$perc_diff[i] > 0.1, TRUE, FALSE)
}

overf_mod
#no overfitting!!

##
control <- trainControl(method = "cv", number = 5)  # Cross Validation con 5 fold

nn <- train(Churn ~ ., data = boruta_selected,
            method = "nnet",
            preProcess = c("center", "scale", "corr", "nzv"), 
            tuneLength = 5, 
            trControl = control, 
            metric = 'Sens',
            trace = TRUE,
            maxit = 300)

rf <- train(Churn ~ ., data = boruta_selected,
            method = "rf",
            trControl = control, 
            metric = 'Sens')

# Ora puoi confrontarli senza errore:
results <- resamples(list(NN = nn, RF = rf))
summary(results)



##


str(nn$control)
str(rf$control)
identical(nn$control, rf$control)
str(nn$control)  # per il modello nn (rete neurale)
str(rf$control)  # per il modello rf (random forest)
control <- trainControl(
  method = "cv",        # k-fold cross-validation
  number = 10,          # 10 folds
  savePredictions = "all",  # salva tutte le predizioni
  classProbs = TRUE,    # calcola probabilità per classi
  summaryFunction = twoClassSummary  # funzione per il calcolo di metriche di classificazione
)
nn = train(Churn ~ ., data = boruta_selected, method = "nnet", preProcess = c("center", "scale", "corr", "nzv"),
           tuneLength = 5, trControl = control, metric = 'Sens', trace = TRUE, maxit = 300)

rf = train(Churn ~ ., data = boruta_selected, method = "rf", preProcess = c("center", "scale", "corr", "nzv"),
           tuneLength = 5, trControl = control, metric = 'Sens')


#confronto metriche
mod <- list(glm = glm, tree = tree, knn = knn, lasso = lasso, 
            rf = rf, gb = gradient_boost, 
            nn = nn)
results <- resamples(mod)
sort(results, decreasing = TRUE, metric = results$metrics[2])
bwplot(results)


#ASSESSMENT####
#thresholds
test$glm = predict(glm,test, "prob")[,1]
test$tree = predict(tree,test, "prob")[,1]
test$knn = predict(knn,test, type="prob")[,1]
test$lasso = predict(lasso,test, "prob")[,1]
test$rf = predict(rf,test, "prob")[,1]
test$gradient_boost = predict(gradient_boost,test, "prob")[,1]
test$nn = predict(nn,test, type="prob")[,1]

#ROC
library(pROC)
roc.glm = roc(Churn ~ glm, data = test, levels = c('Loyal', 'Churn'))
roc.tree = roc(Churn ~ tree, data = test, levels = c('Loyal', 'Churn'))
roc.knn = roc(Churn ~ knn, data = test, levels = c('Loyal', 'Churn'))
roc.lasso = roc(Churn ~ lasso, data = test, levels = c('Loyal', 'Churn'))
roc.rf = roc(Churn ~ rf, data = test, levels = c('Loyal', 'Churn'))
roc.gradient_boost = roc(Churn ~ gradient_boost, data = test, levels = c('Loyal', 'Churn'))
roc.nn = roc(Churn ~ nn, data = test, levels = c('Loyal', 'Churn'))

lista_roc <- list(roc.glm = roc.glm, roc.tree = roc.tree, roc.knn = roc.knn, 
                  roc.lasso = roc.lasso, 
                  roc.rf = roc.rf, roc.gb = roc.gradient_boost, roc.nn = roc.nn)
i <- 0
for (mod in lista_roc){
  i <- i+1
  print(names(lista_roc)[i])
  print(mod$auc)
}

#ROC curves
plot(roc.glm, col = 'black')
plot(roc.tree,add=T,col="hotpink")
plot(roc.knn,add=T,col="orange")
plot(roc.lasso,add=T,col="red")
plot(roc.rf, add = TRUE, col = "green")
plot(roc.gradient_boost,add=T,col="lightblue")
plot(roc.nn,add=T,col="blue")
legend("bottomright", 
       legend = c('glm',"tree","knn","lasso", "randomforest", "gradientboosting","nn"), 
       col = c('black',"hotpink","orange","red", "green", "lightblue","blue"), 
       lty = 1, cex = 0.7, text.font = 1, y.intersp = 0.5, x.intersp = 0.1, lwd = 3)
#migliori sono rf, nn, gb

#knn migliore di nn e gb ma mai di rf


# LIFT OF A GLM MODEL
install.packages("ROCR")    
library(ROCR)
library(MASS)
library(funModeling)
gain_lift(data = test, score = 'rf', target = 'Churn')
gain_lift(data = test, score = 'gradient_boost', target = 'Churn')
gain_lift(data = test, score = 'nn', target = 'Churn')


#modello migliore è RANDOM FOREST


#soglia ottimale per Random Forest

#è più importante classificare correttamente un cliente prossimo al cambio di compagnia
#matrice di profitti con classificazione corretta di Churn che vale 4 voltela classificazione corretta di Loyal

#1 = Churn 0 = Loyal

#pred  1    0
#1    100  0
# 0    0    25 

true_positive_profit <- 100
true_negative_profit <- 25

result_df <- data.frame(threshold = seq(from = 0.00, to = 1.0, by = 0.01), 
                        mean_profit = rep(0,101), acc = rep(0,101), sens = rep(0,101),
                        spec = rep(0,101))

predProb <- predict(rf, newdata = test, type = "prob")[1]

i <- 0
for(threshold in seq(from = 0.00, to = 1.0, by = 0.01)){
  i <- i + 1
  prediction_v <- ifelse(predProb >= threshold, 'Churn', 'Loyal')
  match_count <- sum(prediction_v == test$Churn)
  true_positive_count <- sum(prediction_v  == 'Churn' & test$Churn == 'Churn')
  
  true_negative_count <- sum(prediction_v  == 'Loyal' & test$Churn == 'Loyal')  
  
  false_positive_count <- sum(prediction_v == 'Churn' & test$Churn == 'Loyal')  
  false_negative_count <- sum(prediction_v == 'Loyal' & test$Churn == 'Churn')
  
  total_profit <- true_positive_count * true_positive_profit + true_negative_count *
    true_negative_profit
  mean_profit <- total_profit / nrow(test)
  result_df$mean_profit[i] <- mean_profit
  
  result_df$acc[i] <- (true_positive_count + true_negative_count) / nrow(test)
  result_df$sens[i] <- true_positive_count / (true_positive_count + false_negative_count)
  result_df$spec[i] <- true_negative_count / (true_negative_count + false_positive_count)
}

head(result_df)
tail(result_df)

#plot cost as function of threshold
plot(result_df$threshold,result_df$mean_profit)
result_df[which(result_df$mean_profit == max(result_df$mean_profit)), ]
#0.31


fin <- ifelse(predProb >= 0.31, 'Churn', 'Loyal')
fin
cf_finale <- table(pred = fin, true = test$Churn)
cf_finale

prec <- cf_finale[1,1] / (cf_finale[1,1] + cf_finale[1,2])
prec

#classifichiamo un cliente come prossimo al cambio di compagnia se  ha P(Churn) >= 0.31
names(score)


#SCORE####
predS <- predict(rf, newdata = score, type = "prob")[1]
score_pred <- ifelse(predS >= 0.31, 'Churn', 'Loyal')

score = cbind(score, score_pred)
table(score$Churn)/nrow(score)

##fine


#5.Naive Bayes
set.seed(1)
library(naivebayes)
library(e1071)
library(klaR)
control = trainControl(method="cv", number = 10, classProbs = T,
                       summaryFunction = twoClassSummary)
naivebayes <- train(Churn~., data=train,
                    method = "naive_bayes", trControl = control,
                    tuneGrid = expand.grid(laplace = c(1,10,50), usekernel = c (TRUE, FALSE),
                                           adjust = c(1,1.25)), metric = 'Sens')

naivebayes
confusionMatrix(naivebayes) #accuracy 0.8468
test$naivebayes = predict(naivebayes,test, "prob")[,1]
test$pls = predict(pls,test, "prob")[,1]