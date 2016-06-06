# load in data
library(lme4)
library(car)
library(randomForest)
library(reshape2)
library(ggplot2)
library(rminer)
library(caret)
library(plyr)
library(scales)
library(colorspace)

# load data
xcor_data <- read.csv('data/xcorr-outputs.csv')
clin_data <- read.csv('data/clinical.csv')
subjnames <- read.csv('data/subj-ids.csv', header=FALSE)

# merge clin_data and xcor_data into 'data'
subjnames$id <- 1:88
xcor_data <- merge(xcor_data, subjnames)
data <- merge(xcor_data, clin_data, by.x='V1', by.y='id')

#data <- data[data$group %in% c(2), ] # remove healthy controls
data$variable <- data$PANSS_Negative # set variable of interest
data <- subset(data, !is.na(variable)) # remove subjects missing scores

# find the threshold for those with persistent negative symptoms (PNS)
g1_thresh <- quantile(data$variable, 0.5)
g2_thresh <- quantile(data$variable, 0.5)

# g1 originally controls, removed to allow subdividing schizophrenia group
data$group <- 0
data$group[which(data$variable <= g1_thresh)] <- 1
data$group[which(data$variable > g2_thresh)] <- 2
data <- data[data$group %in% c(1, 2), ]

# set up the factors
data$group <- factor(data$group, levels = c(1, 2), labels = c("g1", "g2"))
data$condition <- factor(data$condition, levels = c(1,2,3), labels = c("IMI", "OBS", "RST"))
data$roi <- as.factor(paste("roi", data$roi, sep="_"))

# init some constants
n_rois <- length(unique(data$roi))
n_conditions <- length(unique(data$condition))
nodes <- round(n_rois * 0.1)
ntree <- 5000

# i hate how ugly this is ... but this is the only data we actually need...
data <- subset(data, condition=='RST')
data <- subset(data, !is.na(correlation))
data <- data[, c('id', 'group', 'condition', 'correlation', 'roi')]
data <- dcast(data, id + group + condition ~ roi, value.var='correlation')
data[is.na(data)] <- 0
################################################################################
# Classification: LOOCV
# variable ~ roi_1 + roi_2 + ...

pred_clas <- matrix(nrow=length(unique(data$id)), ncol=1)
true_clas <- matrix(nrow=length(unique(data$id)), ncol=1)
pred_prob <- matrix(nrow=length(unique(data$id)), ncol=2)

for ( fold in 1:length(unique(data$id)) ) {
    test  <- subset(data, id == data$id[fold])
    train <- subset(data, id != data$id[fold])
    formula = as.formula(paste("group ~ ", paste(grep("roi", colnames(data), value = T), collapse= " + ")))
    rf = randomForest(formula, importance=T, data=train, nodesize=nodes, ntree=ntree)
    pred_clas[fold] <- predict(rf, test, type='class')
    true_clas[fold] <- test$group
    pred_prob[fold, ] <- predict(rf, test, type='prob')
    print(fold)
}

pred_clas <- revalue(factor(pred_clas), c("1"="g1", "2"="g2"))
true_clas <- revalue(factor(true_clas), c("1"="g1", "2"="g2"))
sens <- sensitivity(pred_clas, true_clas)
spec <- specificity(pred_clas, true_clas)
ppv <- posPredValue(pred_clas, true_clas)
npv <- negPredValue(pred_clas, true_clas)
roc <- mmetric(true_clas, pred_prob, metric=c("ROC"))

# plot
theme_set(theme_minimal(base_size = 18)) # for plots
ggplot(as.data.frame(roc$roc$roc), aes(x=V1, y=V2, pin=3)) +
          xlab("1-Specificity") +
          ylab("Sensitivity") +
          scale_x_continuous(breaks = pretty_breaks(n = 10)) +
          scale_y_continuous(breaks = pretty_breaks(n = 10)) +
          geom_line(size=1.25) +
          scale_color_brewer(palette="Set1") +
          theme_minimal() +
          coord_fixed()

