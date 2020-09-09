'''
根据参考文献1中的提示表明不仅需要在python中成功安装graphviz,还要在系统中安装graphviz，
并在系统路径里添加配置(控制面板——》系统——》高级系统设置——》系统属性——》高级——》环境变量——》系统变量——》Path
中添加C:\Program Files (x86)\Graphviz2.38\bin;)。
————————————————
版权声明：本文为CSDN博主「lizzy05」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/lizzy05/java/article/details/88529030
'''

import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
from xgboost import plot_tree
from matplotlib import pyplot
import graphviz
import matplotlib.pyplot as plt
#导入数据
train = pd.read_csv(r'training-peanuts.csv')
tests = pd.read_csv(r'test-peanuts.csv')

print(train.info())

#接下来需要将train和tests转化成matrix类型，方便XGBoost运算：
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
#待训练目标是outcome,所以train_y是outcome。
train_y = train['label']


#设置训练参数
gbm = xgb.XGBClassifier(max_depth=10,learning_rate=0.1,n_estimators=1)  #最大深度3,150颗树

#开始训练
gbm.fit(train_X, train_y)



#数据通过树生成预测结果
predictions = gbm.predict(test_X)  #结果集
predictions_proba = gbm.predict_proba(test_X)  #概率集

#输出正确率
test_y = tests['label']
test_accuracy = accuracy_score(test_y,predictions)
print("test_accuracy = %.2f%%" % (test_accuracy*100))

#打印结果。
#submission = pd.DataFrame({'name': tests['no'],  'outcome': predictions })
#print(predictions_proba)
#submission.to_csv("submission.csv",index=False)



#可视化中显示实际参数的名称
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
ceate_feature_map(train_for_matrix.columns)


#模型树可视化
plot_tree(gbm, num_trees=0,fmap='xgb.fmap')
xgb.plot_importance(gbm,height=0.5,max_num_features=64)
pyplot.show()
#fig = plt.gcf()
#fig.set_size_inches(12,12)
#fig.savefig('tree.png')
