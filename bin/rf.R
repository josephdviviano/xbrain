library(randomForest)
library(reshape2)
library(ggplot2)
library(rminer)
library(caret)
library(plyr)
library(scales)
library(colorspace)

theme_set(theme_minimal(base_size = 18))

# group      1 = HC , 2 = SZ
# condition  1 = imi, 2 = obs, 3 = rst
n_rois = 268
nodes <- round(n_rois * 0.1)
ntree = 100000

glm_data = read.csv('../data/glm-outputs.csv')
glm_data$variable = "beta"
glm_data$value = glm_data$beta
glm_data$beta = NULL

xcr_data = read.csv('../data/xcorr-outputs.csv')
xcr_data$variable = "corr"
xcr_data$value = xcr_data$correlation
xcr_data$correlation = NULL

# combine datasets
#
# "melted" data.frame looks like:
# id    group    condition    variable  roi     value
#  1    H        IMI          beta      roi_1   -0.70
#  1    H        OBS          beta      roi_1    0.03
#  ...
#
# "data" data.frame is organized with ROIs as "features"... e.g. as columns
# id    group    condition    variable  roi_1   roi_2  ...
#  1    H        IMI          beta      -0.70   -0.13
#  1    H        OBS          beta       0.03   -0.01
melted = rbind(glm_data, xcr_data)
melted$group = factor(melted$group, levels = c(1, 2), labels = c("H", "S"))
melted$condition = factor(melted$condition, levels = c(1,2,3), labels = c("IMI", "OBS", "RST"))
melted$roi = as.factor(paste("roi", melted$roi, sep="_"))
melted$variable = as.factor(melted$variable)
data = dcast(melted, id + group + condition + variable ~ roi)

# combine features sets
#
# "combined" data.frame has both beta and correlation measures for all
# ROIs as columns, e.g.
#
# id    group    condition  roi_1_beta  roi_2_beta  ...  roi_1_corr  roi_2_corr
#  1    H        IMI        -0.70       -0.13            -0.13       0.02
#  2    H        OBS        0.03        -0.01            -0.03       0.05
melted$roi = as.factor(paste(melted$roi, melted$variable, sep = "_"))
melted$variable = NULL
combined = dcast(melted, id + group + condition ~ roi)

################################################################################
# STANDARD ANALYSIS
# group ~ roi_1 + roi_2 + ...
group_formula = as.formula(paste("group ~ ", paste(grep("roi", colnames(data), value = T), collapse= " + ")))
glm_imi_rf = randomForest(group_formula, importance = T, data = subset(data, variable == "beta" & condition=="IMI"), nodesize = nodes, ntree = ntree, sampsize=(c(50,38)))
glm_obs_rf = randomForest(group_formula, importance = T, data = subset(data, variable == "beta" & condition=="OBS"), nodesize = nodes, ntree = ntree, sampsize=(c(50,38)))
xcr_imi_rf = randomForest(group_formula, importance = T, data = subset(data, variable == "corr" & condition=="IMI"), nodesize = nodes, ntree = ntree, sampsize=(c(50,38)))
xcr_obs_rf = randomForest(group_formula, importance = T, data = subset(data, variable == "corr" & condition=="OBS"), nodesize = nodes, ntree = ntree, sampsize=(c(50,38)))

group_formula = as.formula(paste("group ~ ", paste(grep("roi", colnames(combined), value = T), collapse= " + ")))
combined_imi_rf = randomForest(group_formula, importance = T, data = subset(combined, condition=="IMI"), nodesize = nodes, ntree = ntree, sampsize=(c(50,38)))
combined_obs_rf = randomForest(group_formula, importance = T, data = subset(combined, condition=="OBS"), nodesize = nodes, ntree = ntree, sampsize=(c(50,38)))

# basic results
print(paste("IMI glm OOB error rate: ", glm_imi_rf$err.rate[glm_imi_rf$ntree]))
print(paste("OBS glm OOB error rate: ", glm_obs_rf$err.rate[glm_obs_rf$ntree]))
print(paste("IMI xcr OOB error rate: ", xcr_imi_rf$err.rate[xcr_imi_rf$ntree]))
print(paste("OBS xcr OOB error rate: ", xcr_obs_rf$err.rate[xcr_obs_rf$ntree]))
print(paste("IMI combined OOB error rate: ", combined_imi_rf$err.rate[combined_imi_rf$ntree]))
print(paste("OBS combined OOB error rate: ", combined_obs_rf$err.rate[combined_obs_rf$ntree]))

# plot
svg('top-feature.svg')
ggplot(data=subset(data), aes(x=roi_261,fill=group))+geom_histogram() + facet_grid(condition ~ variable, scales = "free_x") + geom_vline(x=0, color = "black", size=4, alpha = 0.1) + geom_vline(x=0, color = "black", size=2, alpha = 0.3)
dev.off()

################################################################################
# LOOCV
glm_pred_clas <- matrix(nrow=length(unique(data$id)), ncol=1)
xcr_pred_clas <- matrix(nrow=length(unique(data$id)), ncol=1)
cmb_pred_clas <- matrix(nrow=length(unique(data$id)), ncol=1)

glm_pred_prob <- matrix(nrow=length(unique(data$id)), ncol=2)
xcr_pred_prob <- matrix(nrow=length(unique(data$id)), ncol=2)
cmb_pred_prob <- matrix(nrow=length(unique(data$id)), ncol=2)

for ( fold in 1:length(unique(data$id)) ) {
    # build our testing and training datasets with each fold
    glm_test  <- subset(data, variable == "beta" & condition == "IMI" & id == fold)
    glm_train <- subset(data, variable == "beta" & condition == "IMI" & id != fold)
    xcr_test  <- subset(data, variable == "corr" & condition == "IMI" & id == fold)
    xcr_train <- subset(data, variable == "corr" & condition == "IMI" & id != fold)
    cmb_test  <- subset(combined, condition == "IMI" & id == fold)
    cmb_train <- subset(combined, condition == "IMI" & id != fold)

    # if unbalanced, include sampsize=(c(10,10))
    frm = as.formula(paste("group ~ ", paste(grep("roi", colnames(glm_train), value = T), collapse= " + ")))
    glm_rf <- randomForest(frm, data=glm_train, ntree = ntree, nodesize = nodes)
    frm = as.formula(paste("group ~ ", paste(grep("roi", colnames(xcr_train), value = T), collapse= " + ")))
    xcr_rf <- randomForest(frm, data=xcr_train, ntree = ntree, nodesize = nodes)
    frm = as.formula(paste("group ~ ", paste(grep("roi", colnames(cmb_train), value = T), collapse= " + ")))
    cmb_rf <- randomForest(frm, data=cmb_train, ntree = ntree, nodesize = nodes)

    glm_pred_prob[fold, ] <- predict(glm_rf, glm_test, type='prob')
    glm_pred_clas[fold, ] <- predict(glm_rf, glm_test, type='class')
    xcr_pred_prob[fold, ] <- predict(xcr_rf, xcr_test, type='prob')
    xcr_pred_clas[fold, ] <- predict(xcr_rf, xcr_test, type='class')
    cmb_pred_prob[fold, ] <- predict(cmb_rf, cmb_test, type='prob')
    cmb_pred_clas[fold, ] <- predict(cmb_rf, cmb_test, type='class')
    print(fold)
}

glm_pred_clas <- factor(glm_pred_clas)
glm_pred_clas <- revalue(glm_pred_clas, c("1"="H", "2"="S"))
xcr_pred_clas <- factor(xcr_pred_clas)
xcr_pred_clas <- revalue(xcr_pred_clas, c("1"="H", "2"="S"))
cmb_pred_clas <- factor(cmb_pred_clas)
cmb_pred_clas <- revalue(cmb_pred_clas, c("1"="H", "2"="S"))

# extract the group vector with no repeats
groups  <- subset(combined, condition == 'IMI')
groups  <- groups$group

# calculate fun metrics about how our model is performing
glm_auc <- mmetric(groups, glm_pred_prob, metric=c("AUC"))
xcr_auc <- mmetric(groups, xcr_pred_prob, metric=c("AUC"))
cmb_auc <- mmetric(groups, cmb_pred_prob, metric=c("AUC"))

glm_acc <- mmetric(groups,     glm_pred_prob, metric=c("ACC"))
xcr_acc <- mmetric(groups,     xcr_pred_prob, metric=c("ACC"))
cmb_acc <- mmetric(groups, cmb_pred_prob, metric=c("ACC"))

glm_roc <- mmetric(groups, glm_pred_prob, metric=c("ROC"))
glm_roc <- as.data.frame(glm_roc$roc$roc)
glm_roc$method <- 'glm'

xcr_roc <- mmetric(groups, xcr_pred_prob, metric=c("ROC"))
xcr_roc <- as.data.frame(xcr_roc$roc$roc)
xcr_roc$method <- 'xcorr'

cmb_roc <- mmetric(groups, cmb_pred_prob, metric=c("ROC"))
cmb_roc <- as.data.frame(cmb_roc$roc$roc)
cmb_roc$method <- 'combined'

glm_sens <- sensitivity(glm_pred_clas, groups)
xcr_sens <- sensitivity(xcr_pred_clas, groups)
cmb_sens <- sensitivity(cmb_pred_clas, groups)

glm_spec <- specificity(glm_pred_clas, groups)
xcr_spec <- specificity(xcr_pred_clas, groups)
cmb_spec <- specificity(cmb_pred_clas, groups)

glm_ppv <- posPredValue(glm_pred_clas, groups)
xcr_ppv <- posPredValue(xcr_pred_clas, groups)
cmb_ppv <- posPredValue(cmb_pred_clas, groups)

glm_npv <- negPredValue(glm_pred_clas, groups)
xcr_npv <- negPredValue(xcr_pred_clas, groups)
cmb_npv <- negPredValue(cmb_pred_clas, groups)

# save data
save.image('xbrain-rf.RSession')

# plot our resultsi
pdf('glm_imi_rf.pdf')
varImpPlot(glm_imi_rf)
varImpPlot(xcr_imi_rf)
varImpPlot(combined_imi_rf)
dev.off()

plotbro <- rbind(rbind(glm_roc, xcr_roc), cmb_roc)
pdf('roc.pdf')
ggplot(plotbro, aes(x=V1, y=V2, colour=method, pin=3)) +
          xlab("1-Specificity") +
          ylab("Sensitivity") +
          scale_x_continuous(breaks = pretty_breaks(n = 10)) +
          scale_y_continuous(breaks = pretty_breaks(n = 10)) +
          geom_line(size=1.25) +
          scale_color_brewer(palette="Set1") +
          theme_minimal() +
          coord_fixed()
dev.off()

