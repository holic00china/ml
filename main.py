import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from keras_preprocessing.text import Tokenizer, text_to_word_sequence
from keras_preprocessing.sequence import pad_sequences
import time
from sklearn.metrics import confusion_matrix

#回归模型
def Train(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression

    # 创建 TF-IDF 向量化器
    vectorizer = TfidfVectorizer()

    # 对训练集和测试集进行向量化
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # 创建 LogisticRegression 模型
    model = LogisticRegression(solver='sag')

    # 训练模型
    model.fit(X_train, y_train)

    # 保存模型和向量化器
    with open('logistic_regression.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 评估模型
    cm1 = confusion_matrix(y_test, y_pred)
    print(cm1)
    accuracy = model.score(X_test, y_test)
    print("Accuracy:", accuracy)


#随机森林模型
def TrainRandomForest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    # 创建 TF-IDF 向量化器
    vectorizer = TfidfVectorizer()

    # 对训练集和测试集进行向量化
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)


    # 创建随机森林模型 需要调整超参数
    model = RandomForestClassifier(n_estimators=300, max_depth=40,min_samples_split=10)

    # 训练模型
    model.fit(X_train, y_train)

    # 保存模型和向量化器
    with open('random_forest.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 评估模型
    cm1 = confusion_matrix(y_test, y_pred)
    print(cm1)
    accuracy = model.score(X_test, y_test)
    print("Accuracy:", accuracy)

if __name__ == '__main__':
    start = time.time()
    df = pd.read_csv('all_data_url_random.csv')
    df['label'] = df['label'].map({'异常':0,'正常':1})
    X=df['content']
    y=df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5)
    #print(X_train.values, X_test.values, y_train.values, y_test.values)
    TrainRandomForest(X_train, X_test, y_train, y_test)
    end = time.time()
    print('程序运行时间为: %s Seconds'%(end-start))
