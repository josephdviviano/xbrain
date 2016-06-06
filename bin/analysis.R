# load in data
library(lme4)
library(car)
library(ez)

glm_data <- read.csv('glm-outputs.csv')
xcor_data <- read.csv('xcorr-outputs.csv')

# remove the resting-state data
xcor_data <- xcor_data[xcor_data$condition %in% c(1,2),]

# merge data
data <- merge(glm_data, xcor_data)

# Convert variables to factors
data <- within(data, {
  id <- factor(id)
  group <- factor(group)
  condition <- factor(condition)
  roi <- factor(roi)
})

n_rois <- length(unique(glm_data$roi))
n_conditions <- length(unique(glm_data$condition))

# group, condition, group*condition
ez.mat_betas <- matrix(, nrow=n_rois, ncol=3)
ez.mat_corrs <- matrix(, nrow=n_rois, ncol=3)

for (roi in seq(n_rois)) {
    testdata <- data[data$roi == roi, ]
    # ROI-wise ANOVA on beta weights
    ez.out <- ezANOVA(testdata, dv=beta, wid=id, within=.(condition), between=.(group), detailed=1, type=3)
    ez.mat_betas[roi, ] <- ez.out$ANOVA$p[2:4]
    # ROI-wise ANOVA on correlation values
    ez.out <- ezANOVA(testdata, dv=correlation, wid=id, within=.(condition), between=.(group), detailed=1, type=3)
    ez.mat_corrs[roi, ] <- ez.out$ANOVA$p[2:4]
}

# find significant ROIs.
crit_p <- 0.05
ez.mat_betas[ ,1] <- p.adjust(ez.mat_betas[ ,1], method='fdr')
ez.mat_betas[ ,2] <- p.adjust(ez.mat_betas[ ,2], method='fdr')
ez.mat_betas[ ,3] <- p.adjust(ez.mat_betas[ ,3], method='fdr')
ez.mat_corrs[ ,1] <- p.adjust(ez.mat_corrs[ ,1], method='fdr')
ez.mat_corrs[ ,2] <- p.adjust(ez.mat_corrs[ ,2], method='fdr')
ez.mat_corrs[ ,3] <- p.adjust(ez.mat_corrs[ ,3], method='fdr')

sig_grp_betas <- which(ez.mat_betas[ ,1] < crit_p)
sig_cnd_betas <- which(ez.mat_betas[ ,2] < crit_p)
sig_int_betas <- which(ez.mat_betas[ ,3] < crit_p)
sig_grp_corrs <- which(ez.mat_corrs[ ,1] < crit_p)
sig_cnd_corrs <- which(ez.mat_corrs[ ,2] < crit_p)
sig_int_corrs <- which(ez.mat_corrs[ ,3] < crit_p)

# follow up t-tests for betas: condition
test_cnd_betas = matrix(, length(sig_cnd_betas), ncol=5)
count <- 1
for (i in sig_cnd_betas) {
    testdata <- subset(data, roi == i)
    tst <- t.test(testdata$beta[testdata$condition == 1], testdata$beta[testdata$condition == 2])
    test_cnd_betas[count, 1] <- tst$p.value
    test_cnd_betas[count, 2] <- tst$statistic
    test_cnd_betas[count, 3] <- tst$estimate[1] - tst$estimate[2]
    test_cnd_betas[count, 4] <- tst$estimate[1]
    test_cnd_betas[count, 5] <- tst$estimate[2]
    count = count + 1
}
test_cnd_betas <- as.data.frame(test_cnd_betas)
colnames(test_cnd_betas)[1] <- "p"
colnames(test_cnd_betas)[2] <- "t"
colnames(test_cnd_betas)[3] <- "imitate-observe"
colnames(test_cnd_betas)[4] <- "imitate"
colnames(test_cnd_betas)[5] <- "observe"

test_cnd_betas$roi <- sig_cnd_betas
test_cnd_betas$p <- p.adjust(test_cnd_betas$p, method='fdr')

# follow up t-tests for xcorr: group
test_grp_corrs = matrix(, length(sig_grp_corrs), ncol=5)
count <- 1
for (i in sig_grp_corrs) {
    testdata <- subset(data, roi == i)
    tst <- t.test(testdata$correlation[testdata$group == 1], testdata$correlation[testdata$group == 2])
    test_grp_corrs[count, 1] <- tst$p.value
    test_grp_corrs[count, 2] <- tst$statistic
    test_grp_corrs[count, 3] <- tst$estimate[1] - tst$estimate[2]
    test_grp_corrs[count, 4] <- tst$estimate[1]
    test_grp_corrs[count, 5] <- tst$estimate[2]
    count = count + 1
}
test_grp_corrs <- as.data.frame(test_grp_corrs)
colnames(test_grp_corrs)[1] <- "p"
colnames(test_grp_corrs)[2] <- "t"
colnames(test_grp_corrs)[3] <- "healthy-schizophenia"
colnames(test_grp_corrs)[4] <- "healthy"
colnames(test_grp_corrs)[5] <- "schizophrenia"

test_grp_corrs$roi <- sig_grp_corrs
test_grp_corrs$p <- p.adjust(test_grp_corrs$p, method='fdr')

# follow up t-tests for xcorr: condition
test_cnd_corrs = matrix(, length(sig_cnd_corrs), ncol=5)
count <- 1
for (i in sig_cnd_corrs) {
    testdata <- subset(data, roi == i)
    tst <- t.test(testdata$correlation[testdata$group == 1], testdata$correlation[testdata$group == 2])
    test_cnd_corrs[count, 1] <- tst$p.value
    test_cnd_corrs[count, 2] <- tst$statistic
    test_cnd_corrs[count, 3] <- tst$estimate[1] - tst$estimate[2]
    test_cnd_corrs[count, 4] <- tst$estimate[1]
    test_cnd_corrs[count, 5] <- tst$estimate[2]
    count = count + 1
}
test_cnd_corrs <- as.data.frame(test_cnd_corrs)
colnames(test_cnd_corrs)[1] <- "p"
colnames(test_cnd_corrs)[2] <- "t"
colnames(test_cnd_corrs)[3] <- "imitate-observe"
colnames(test_cnd_corrs)[4] <- "imitate"
colnames(test_cnd_corrs)[5] <- "observe"

test_cnd_corrs$roi <- sig_cnd_corrs
test_cnd_corrs$p <- p.adjust(test_cnd_corrs$p, method='fdr')

# follow up t-tests for xcorr: group*condition interaction
test_intim_corrs = matrix(, length(sig_int_corrs), ncol=5)
count <- 1
for (i in sig_int_corrs) {
    testdata <- subset(data, roi == i)

    # group 1 data for imitate only
    testdata_group1 <- testdata$correlation[testdata$group == 1][testdata$condition == 1]
    testdata_group1 <- testdata_group1[!is.na(testdata_group1)]

    # group 2 data for imitate only
    testdata_group2 <- testdata$correlation[testdata$group == 2][testdata$condition == 1]
    testdata_group2 <- testdata_group2[!is.na(testdata_group2)]

    tst <- t.test(testdata_group1, testdata_group2)
    test_intim_corrs[count, 1] <- tst$p.value
    test_intim_corrs[count, 2] <- tst$statistic
    test_intim_corrs[count, 3] <- tst$estimate[1] - tst$estimate[2]
    test_intim_corrs[count, 4] <- tst$estimate[1]
    test_intim_corrs[count, 5] <- tst$estimate[2]

    count = count + 1
}
test_intim_corrs <- as.data.frame(test_intim_corrs)
colnames(test_intim_corrs)[1] <- "p"
colnames(test_intim_corrs)[2] <- "t"
colnames(test_intim_corrs)[3] <- "hc-sz-imitate"
colnames(test_intim_corrs)[4] <- "hc-imitate"
colnames(test_intim_corrs)[5] <- "sz-imitate"

test_intim_corrs$roi <- sig_int_corrs
test_intim_corrs$p <- p.adjust(test_intim_corrs$p, method='fdr')

# follow up t-tests for xcorr: group*condition interaction
test_intob_corrs = matrix(, length(sig_int_corrs), ncol=5)
count <- 1
for (i in sig_int_corrs) {
    testdata <- subset(data, roi == i)

    # group 1 data for observe only
    testdata_group1 <- testdata$correlation[testdata$group == 1][testdata$condition == 2]
    testdata_group1 <- testdata_group1[!is.na(testdata_group1)]

    # group 2 data for observe only
    testdata_group2 <- testdata$correlation[testdata$group == 2][testdata$condition == 2]
    testdata_group2 <- testdata_group2[!is.na(testdata_group2)]

    tst <- t.test(testdata_group1, testdata_group2)
    test_intob_corrs[count, 1] <- tst$p.value
    test_intob_corrs[count, 2] <- tst$statistic
    test_intob_corrs[count, 3] <- tst$estimate[1] - tst$estimate[2]
    test_intob_corrs[count, 4] <- tst$estimate[1]
    test_intob_corrs[count, 5] <- tst$estimate[2]

    count = count + 1
}
test_intob_corrs <- as.data.frame(test_intob_corrs)
colnames(test_intob_corrs)[1] <- "p"
colnames(test_intob_corrs)[2] <- "t"
colnames(test_intob_corrs)[3] <- "hc-sz-observe"
colnames(test_intob_corrs)[4] <- "hc-observe"
colnames(test_intob_corrs)[5] <- "sz-observe"

test_intob_corrs$roi <- sig_int_corrs
test_intob_corrs$p <- p.adjust(test_intob_corrs$p, method='fdr')

# write data
write.table(test_cnd_betas, 'betas-condition.csv', row.names=FALSE)
write.table(test_cnd_corrs, 'correlations-condition.csv', row.names=FALSE)
write.table(test_grp_corrs, 'correlations-group.csv', row.names=FALSE)
write.table(test_intim_corrs, 'correlations-int-im.csv', row.names=FALSE)
write.table(test_intob_corrs, 'correlations-int-ob.csv', row.names=FALSE)



