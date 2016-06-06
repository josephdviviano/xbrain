#!/usr/bin/env R

behav <- read.csv('database_BEHAV_150920.csv')
panss <- read.csv('database_PANSS_150920.csv')
rmet  <- read.csv('database_RMET_150920.csv')
tasit <- read.csv('database_TASIT_150920.csv')

# first pass -- only using behavioural data and tasit
database <- merge(behav, tasit, by='ID', all.y=TRUE)

# save table
write.table(database, file='database.csv', row.names=FALSE, sep=',')

