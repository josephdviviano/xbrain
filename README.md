**xbrain: cross-brain correlation for cognitive score prediction**

A set of tools for performing a special form of cognitive score prediction using a simple cross-brain correlation metric using fMRI data.

+ uses a variation on the method implemeted [here](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041196)
+ this method pulls out a random subset of 'template' subjects with average cognitive scores.
+ template chosen based on a pre-selected cognitive trait of interest.
+ all subject are correlated against this template, regardless of group membership.
+ the output cross-subject correlations can then be brought forward to do classification and regression experiments without worry about lack of independence in observations.

**app for ISC analysis**

https://machow.shinyapps.io/shiny-isc

