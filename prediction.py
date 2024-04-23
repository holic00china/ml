import pickle
import pandas as pd

# 加载已经训练好的模型
with open('logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# 加载新的数据集
new_data = pd.read_csv('nsfocus_isop_train.csv')  # 根据实际的数据集文件名和路径进行修改

# 假设新数据集的特征列为 'text'
X_new = new_data['url']

# 使用之前训练模型时的向量化器对新数据进行文本向量化
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
X_new = vectorizer.transform(X_new)

# 使用模型进行预测
y_pred_new = model.predict(X_new)

# 输出预测结果
for i, prediction in enumerate(y_pred_new):
    print("样本", i+1, "的预测结果:", prediction)