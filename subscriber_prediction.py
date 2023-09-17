import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
# clf = lgb.LGBMClassifier()
# 设置LightGBM分类器使用f1_score作为评估指标，并指定轮次（迭代次数）
from sklearn.model_selection import train_test_split

clf = lgb.LGBMClassifier(objective='binary',    # 二分类
                         metric='binary_logloss',   # 指定评估指标
                         boosting_type='gbdt') # 指定弱学习器的类型，默认值为 ‘gbdt’



train_data = pd.read_csv('用户新增预测挑战赛公开数据/train.csv')
test_data = pd.read_csv('用户新增预测挑战赛公开数据/test.csv')
"""
其中uuid为样本唯一标识，
eid为访问行为ID，
udmap为行为属性，其中的key1到key9表示不同的行为属性，如项目名、项目id等相关字段，
common_ts为应用访问记录发生时间（毫秒时间戳），
其余字段x1至x8为用户相关的属性，为匿名处理字段。
target字段为预测目标，即是否为新增用户。
"""

train_data['common_ts'] = pd.to_datetime(train_data['common_ts'], unit='ms')
test_data['common_ts'] = pd.to_datetime(test_data['common_ts'], unit='ms')


def udmap_onethot(d):
    """
    将udmap字段的字符串表示解析为一个包含9个元素的向量，其中每个元素对应于不同的行为属性（key1到key9），
    如果属性存在，则为1，否则为0。如果udmap为'unknown'，则返回一个全零向量。
    """
    v = np.zeros(9)
    if d == 'unknown':
        return v

    d = eval(d)
    for i in range(1, 10):
        if 'key' + str(i) in d:
            v[i - 1] = d['key' + str(i)]

    return v


train_udmap_df = pd.DataFrame(np.vstack(train_data['udmap'].apply(udmap_onethot)))
test_udmap_df = pd.DataFrame(np.vstack(test_data['udmap'].apply(udmap_onethot)))

train_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]
test_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]
# 提取特征
train_data = pd.concat([train_data, train_udmap_df], axis=1)
test_data = pd.concat([test_data, test_udmap_df], axis=1)
# 创建了两个新的特征：eid_freq和eid_mean，
# 它们分别表示eid字段的频率和对应target字段的平均值。这些特征用于提供关于eid的统计信息。
train_data['eid_freq'] = train_data['eid'].map(train_data['eid'].value_counts())
test_data['eid_freq'] = test_data['eid'].map(train_data['eid'].value_counts())

train_data['eid_mean'] = train_data['eid'].map(train_data.groupby('eid')['target'].mean())
test_data['eid_mean'] = test_data['eid'].map(train_data.groupby('eid')['target'].mean())
# 创建了一个二元特征 udmap_isunknown，表示udmap字段是否为'unknown'，如果是则为1，否则为0。
train_data['udmap_isunknown'] = (train_data['udmap'] == 'unknown').astype(int)
test_data['udmap_isunknown'] = (test_data['udmap'] == 'unknown').astype(int)
train_data['common_ts_hour'] = train_data['common_ts'].dt.hour
test_data['common_ts_hour'] = test_data['common_ts'].dt.hour
# 使用LightGBM分类器 (clf) 对训练数据进行拟合，使用除了udmap、common_ts、uuid和target以外的所有特征进行训练。
# clf.fit(
#     train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1),
#     train_data['target']
# )

# 分割训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(
    train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1),
    train_data['target'],
    test_size=0.2,  # 设置验证集的大小
    random_state=42
)
# 将训练数据和验证数据转换为LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

# 定义模型参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
}
clf = lgb.train(
    params,
    train_data,
    num_boost_round=10, #指定最大迭代次数，默认值为10
    valid_sets=[valid_data],  # 指定验证数据集
    valid_names=['valid'],
)

# 预测测试集
predictions = clf.predict(test_data.drop(['udmap', 'common_ts', 'uuid'], axis=1), num_iteration=clf.best_iteration)

# 将概率转换为二进制预测
predictions_binary = [1 if x >= 0.5 else 0 for x in predictions]

# 保存预测结果到CSV文件
pd.DataFrame({
    'uuid': test_data['uuid'],
    'target': predictions_binary
}).to_csv('submit.csv', index=None)
