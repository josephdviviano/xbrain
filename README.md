**xbrain: cross-brain correlations during imitate observe task**

+ subjects from DTI sample (3T) and SPINS (CMH site only).
+ uses a variation on the method implemeted [here](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041196)
+ this method depends on a set of template subjects that are held out of the final analysis.
+ all subject are correlated against this template, regardless of group membership.
+ the output cross-subject correlations can then be brought forward to do classification and regression experiments without worry about lack of independence in observations.

**code/**

+ clone from: https://bitbucket.org/josephdviviano/xbrain-2015

**app for ISC analysis**

https://machow.shinyapps.io/shiny-isc

**exceptions**

+ `DTI_CMH_S148` resting state scan is spiral, due to lack of available EPI data.
