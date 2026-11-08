# Answers
## Dataset

|  Dataset | Full-pool | Greedy (Accuracy)  |
|---|---|---|
|   Toe Segmentation|   0.7325|   0.9693|

The full pool ensemble was a lot lower scored than the Greedy one as the full-pool includes many classifiers that are not
great on time-series data, like Logistic Regression and RandomForest. However, greedy will have discarded these classifiers
due to their poor scores on time-series data like Toe Segmentation to prevent them from dragging down th ensemble average.
## Accuracy V Log Loss
Optimising for log loss consistently selects a larger number of classifiers than optimising for accuracy. For example: DistalPhalanxOutlineAgeGroup selects 1 classifier for accuracy and 6 classifiers for log loss. However, it is worth noting that there are a few counter examples but the difference is a lot less, such as DistalPhalanxTW selects 3 for accuracy and 2 for log loss.

This is because accuracy is threshold-based metric; once the ensemble predicts the highest class, adding another model increases the confidence without affecting the accuracy score, causing the search to end early. On the other hand, log loss is a continuous metric that penalises overconfident wrong answers and rewards well-calibrated probabilities.
## OOF Accuracy V Greedy Accuracy
The out-of-fold (OOF) accuracy is not a reliable estimate of unseen test performance because it is consistently overly optimistic. Across all bar one dataset, OOF accuracy is noticeably higher than greedy accuracy. For example: OOF is 0.804 in Computers compared with the equivalent in 0.732 in greedy accuracy.

This is due to overfitting to the testing set. Because the greedy algorithm explicitly searches for the specific combination of model that maximises the OOF score, it capitalises on random noise and coincidental alginments in the OOF predictions. Consequently, the final OOF score becomes an artificially inflated metric that overestimates how well the selected ensemble will generalise to genuinely new data.