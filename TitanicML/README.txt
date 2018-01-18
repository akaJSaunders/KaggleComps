Tutorial Competition
Not-Ranked

Credit to these great tutorials
Machine Learning from Start to Finish with Scikit-Learn
https://www.kaggle.com/jeffd23/titanic/scikit-learn-ml-from-start-to-finish
Introduction to Ensembling/Stacking in Python
https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
XGBoost example (Python)
https://www.kaggle.com/datacanary/xgboost-example-python
An Interactive Data Science Tutorial
https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial
Sina's Tutorial
https://www.kaggle.com/sinakhorami/titanic-best-working-classifier

Notes for myself:
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

Posted by Cryst 
I quickly forked the notebook and ran some tests and what I noted: e.g. for
the GradientBoostingClassifier or it doesn't make any difference whether
'Fare' and 'Age' are discretized or not, accuracy hovers still around 82%
(even gets slightly improved by NOT discretizing). SVC() drops really 
heavily to around 71%! So it seems that some classifiers learn too much
noise and discretizing can help them internalize the general rule (e.g.
higher fare -> higher survival chance) instead of individual data
points (=prevent overfitting).
It is really interesting to see how one classifier needs the feature
engineering but the other one is good enough to learn the rule 'all by 
itself'.

One hot encoder

