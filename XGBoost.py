'''
Caution: The premise of the following analysis is to successfully install graphviz in python and install graphviz in the system. And add the configuration environment variable Path to the system path, add C:\Program Files (x86)\Graphviz2.38\bin;)
————————————————
'''

import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
from xgboost import plot_tree
from matplotlib import pyplot
import graphviz
import matplotlib.pyplot as plt
#input dataset
train = pd.read_csv(r'training-peanuts.csv')
tests = pd.read_csv(r'test-peanuts.csv')

print(train.info())

#Convert train and tests into matrix type to facilitate XGBoost operations:：
#feature_columns_to_use = ['5-Methoxysterigmatocystin', 'P1unk0wn22', 'Versicolorin_B_(VerB)',]
#feature_columns_to_use = ['5-Methoxysterigmatocystin', 'P1unk0wn22',]
#feature_columns_to_use = ['5-Methoxysterigmatocystin', 'Versicolorin_B_(VerB)',]
#feature_columns_to_use = ['P1unk0wn22', 'Versicolorin_B_(VerB)',]
#feature_columns_to_use = ['5-Methoxysterigmatocystin',]
#feature_columns_to_use = ['P1unk0wn22',]
feature_columns_to_use = [ 'Versicolorin_B_(VerB)',]
train_for_matrix = train[feature_columns_to_use]
test_for_matrix = tests[feature_columns_to_use]
train_X = train_for_matrix.values
test_X = test_for_matrix.values
# train_y is the outcome.
train_y = train['label']


# Set training parameters
gbm = xgb.XGBClassifier(max_depth=10,learning_rate=0.1,n_estimators=1)  #最大深度3,150颗树

# Start training
gbm.fit(train_X, train_y)



# Data is generated by tree to predict results.
predictions = gbm.predict(test_X)  # results dataset
predictions_proba = gbm.predict_proba(test_X)  # Probability set

# Output accuracy.
test_y = tests['label']
test_accuracy = accuracy_score(test_y,predictions)
print("test_accuracy = %.2f%%" % (test_accuracy*100))

# Print results.
#submission = pd.DataFrame({'name': tests['no'],  'outcome': predictions })
#print(predictions_proba)
#submission.to_csv("submission.csv",index=False)



# Visually display the names of actual parameters.
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
ceate_feature_map(train_for_matrix.columns)


# Model tree visualization
plot_tree(gbm, num_trees=0,fmap='xgb.fmap')
xgb.plot_importance(gbm,height=0.5,max_num_features=64)
pyplot.show()
#fig = plt.gcf()
#fig.set_size_inches(12,12)
#fig.savefig('tree.png')
