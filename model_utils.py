from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
def xgb_model(X_train, y_train, X_test, y_test=None):
    # 将训练数据进一步划分为训练集和验证集（80%训练，20%验证）
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2)
    # 将数据转换为XGBoost专用的DMatrix格式（提升计算效率）
    train_matrix = xgb.DMatrix(X_train_split, label=y_train_split)
    valid_matrix = xgb.DMatrix(X_val, label=y_val)
    test_matrix = xgb.DMatrix(X_test)

    # XGBoost模型参数配置
    params = {
        'device':'cuda:0',
        'booster': 'gbtree',  # 使用基于树的模型（而非线性模型）
        'objective': 'binary:logistic',  # 二分类任务，输出概率
        'eval_metric': 'auc',  # 评估指标为AUC（适用于分类不平衡场景）
        'gamma': 1,  # 节点分裂所需的最小损失减少值（正则化）
        'min_child_weight': 1.5,  # 叶子节点最小样本权重和（防止过拟合）
        'max_depth': 5,  # 树的最大深度（控制模型复杂度）
        'lambda': 10,  # L2正则化系数（控制过拟合）
        'subsample': 0.7,  # 每棵树随机采样的样本比例（行采样）
        'colsample_bytree': 0.7,  # 每棵树随机采样的特征比例（列采样）
        'colsample_bylevel': 0.7,  # 每层树的列采样比例
        'eta': 0.04,  # 学习率（控制每步更新的步长）
        'tree_method': 'exact',  # 树的构建方法（精确贪心算法）
        'seed': 2025,  # 随机种子（保证结果可复现）
        'n_jobs': 8,  # 并行线程数（加速训练）
        "silent": True,  # 关闭运行时信息输出
    }
    # 设置监控的验证集列表（训练集和验证集）
    watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]
    # 训练XGBoost模型
    model = xgb.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200,
                      early_stopping_rounds=200)
    """计算在验证集上的得分"""
    val_pred = model.predict(valid_matrix, iteration_range=(0,model.best_iteration))# 使用最佳迭代次数预测
    fpr, tpr, threshold = metrics.roc_curve(y_val, val_pred)# 计算ROC曲线坐标点
    roc_auc = metrics.auc(fpr, tpr) # 计算AUC值
    print('调参后xgboost单模型在验证集上的AUC：{}'.format(roc_auc))
    """对测试集进行预测"""
    test_pred = model.predict(test_matrix, iteration_range=(0,model.best_iteration)) # 输出验证结果

    return test_pred


def lgb_model(X_train, y_train, X_test, y_test=None):
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2)
    train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
    valid_matrix = lgb.Dataset(X_val, label=y_val)

    # 调参后的最优参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'num_leaves': 37,
        'max_depth': 7,
        'min_data_in_leaf': 83,
        'min_child_weight':6.6,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.58,
        'bagging_freq': 83,
        'reg_lambda': 4,
        'reg_alpha': 9,
        'min_split_gain': 0.5,
        'nthread': 8,
        'seed': 2025,
        'silent': True,
        'verbose': -1,
    }

    model = lgb.train(params, train_matrix, num_boost_round=50000, valid_sets=[train_matrix, valid_matrix],
                      callbacks=[lgb.log_evaluation(500),lgb.early_stopping(500)])
    """计算在验证集上的得分"""
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    fpr, tpr, threshold = metrics.roc_curve(y_val, val_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print('调参后lightgbm单模型在验证集上的AUC：{}'.format(roc_auc))
    """对测试集进行预测"""
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    return test_pred