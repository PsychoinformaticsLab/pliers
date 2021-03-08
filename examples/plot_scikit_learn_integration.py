"""
This is my example script
=========================

This example doesn't do much, it just makes a simple plot
"""
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Scikit-Learn Integration
# ==================
# Example using `pliers` as a node in a typical scikit-learn pipeline. Example code taken from scikit-learn's [website](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

# %%
# Data setup
from glob import glob
from pliers.tests.utils import get_test_data_path
from os.path import join
import numpy as np

X = glob(join(get_test_data_path(), 'image', '*.jpg'))
# Just use random classes since this is just an example
y = np.random.randint(0, 3, len(X))
print('Number of images found: %d' % len(X))


# %%
# Pliers setup
from pliers.graph import Graph
from pliers.utils.scikit import PliersTransformer
g = Graph({'roots':[{'transformer':'BrightnessExtractor'},
                    {'transformer':'SharpnessExtractor'},
                    {'transformer':'VibranceExtractor'}]})


# %%
# Sklearn setup
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline

# ANOVA SVM-C Pipeline
pliers_transformer = PliersTransformer(g)
anova_filter = SelectKBest(f_regression, k=2)
clf = svm.SVC(kernel='linear')
pipeline = Pipeline([('pliers', pliers_transformer), ('anova', anova_filter), ('svc', clf)])


# %%
# Fit and get training accuracy
pipeline.set_params(svc__C=.1).fit(X, y)
prediction = pipeline.predict(X)
pipeline.score(X, y)


# %%
# Getting the selected features chosen by anova_filter
pipeline.named_steps['anova'].get_support()


# %%



