# BIAS VS VARIANCE IN ML

## BIAS aka overfit
bias is how well it fits the training set.
a straight line linear regression that fits a noisy train dataset has high bias. ('sum of squares' errors)
a very curvy polynomial line that fits the train data exceptionally well has low bias.
also it can be said if the 'true relationship' of the IV and DV is perhaps some complex curve, and we had models that had straight lines, or various curve lines, but they're all off from the true relationship, that is still considered bias.

## VARIANCE aka generalization/drift, variance is low across batches of data
but variance is measured over variances between samples/datasets.
on a test set, the straight line that was more general(less overfit on train data) overall has less errors, compared to the polynomial that overfit the train set and fits poorly on the test set. in this case, the straight line has lower variance (across various datasets/samples), whereas the performance/scores of the polynomial will have higher variance across multiple datasets.
In other words, its hard to predict how well the squiggly line will perform on future datasets. It might score well sometimes, and do horribly sometimes.
The straight line might give good predictions, not great predictions, but it does so with consistency.

## OUR GOAL, FINDING BALANCE
Ideally, in ML, the algorithm has low bias (accurately model the true relationship) and low variance.
i.e simple and generalizable, yet complex enough to still model true relationship well.

3 methods of finding sweet spot between bias and variance:
Regularization
Bagging
Boosting



# PRECISION V RECALL

precision = TP / (TP+FP)    i.e minimizing FP

recall = TP / (TP+FN)       i.e minimizing FN

recall = cancer prediction, customer churn (find as many positives as possible, false prediction nevermind) (very permissive)
precision = spam (be as accurate as possible in finding positives) (very strict)
