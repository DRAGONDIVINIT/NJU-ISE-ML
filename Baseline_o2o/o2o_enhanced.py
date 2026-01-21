# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')  # 不显示警告

def prepare(dataset):
    data = dataset.copy()
    # 折扣率处理
    data['is_manjian'] = data['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    data['discount_rate'] = data['Discount_rate'].map(
        lambda x: float(x) if ':' not in str(x) else
        (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0])
    )
    data['min_cost_of_manjian'] = data['Discount_rate'].map(
        lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0])
    )
    data['discount_man'] = data['Discount_rate'].map(
        lambda x: 0 if ':' not in str(x) else float(str(x).split(':')[0])
    )
    data['discount_jian'] = data['Discount_rate'].map(
        lambda x: 0 if ':' not in str(x) else float(str(x).split(':')[1])
    )
    # 距离处理
    data['Distance'].fillna(-1, inplace=True)
    data['null_distance'] = data['Distance'].map(lambda x: 1 if x == -1 else 0)
    # 时间处理
    if 'Date_received' in data.columns:
        data['Date_received'].fillna(0, inplace=True)
        data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d', errors='coerce')
    
    if 'Date' in data.columns.tolist():
        data['Date'].fillna(0, inplace=True)
        data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d', errors='coerce')
    
    return data

def get_label(dataset):
    data = dataset.copy()
    data['label'] = list(map(
        lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0,
        data['date'], data['date_received']
    ))
    return data

def get_simple_feature(label_field):
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    feature = data.copy()
    # 用户领券数
    keys = ['User_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户领取特定优惠券数
    keys = ['User_id', 'Coupon_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户当天领券数
    keys = ['User_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户当天领取特定优惠券数
    keys = ['User_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户是否在同一天重复领取了特定优惠券
    keys = ['User_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=lambda x: 1 if len(x) > 1 else 0)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'repeat_receive'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    feature.drop(['cnt'], axis=1, inplace=True)
    return feature

def get_historical_feature(history_field, label_field):
    hist = history_field.copy()
    hist['cnt'] = 1
    label = label_field.copy()
    label['cnt'] = 1

    # 用户历史领券数
    user_receive_cnt = hist.groupby('User_id')['cnt'].count().reset_index().rename(
        columns={'cnt': 'user_historical_receive_cnt'})
    label = pd.merge(label, user_receive_cnt, on='User_id', how='left')

    # 用户历史核销数
    user_use_cnt = hist[hist['label'] == 1].groupby('User_id')['cnt'].count().reset_index().rename(
        columns={'cnt': 'user_historical_use_cnt'})
    label = pd.merge(label, user_use_cnt, on='User_id', how='left')

    # 用户历史核销率
    label['user_historical_use_rate'] = label['user_historical_use_cnt'] / label['user_historical_receive_cnt'].replace(0, np.nan)

    # 商家历史被领券数
    merchant_receive_cnt = hist.groupby('Merchant_id')['cnt'].count().reset_index().rename(
        columns={'cnt': 'merchant_historical_receive_cnt'})
    label = pd.merge(label, merchant_receive_cnt, on='Merchant_id', how='left')

    # 商家历史核销数
    merchant_use_cnt = hist[hist['label'] == 1].groupby('Merchant_id')['cnt'].count().reset_index().rename(
        columns={'cnt': 'merchant_historical_use_cnt'})
    label = pd.merge(label, merchant_use_cnt, on='Merchant_id', how='left')

    # 商家历史核销率
    label['merchant_historical_use_rate'] = label['merchant_historical_use_cnt'] / label['merchant_historical_receive_cnt'].replace(0, np.nan)

    # 优惠券历史被领取次数
    coupon_receive_cnt = hist.groupby('Coupon_id')['cnt'].count().reset_index().rename(
        columns={'cnt': 'coupon_historical_receive_cnt'})
    label = pd.merge(label, coupon_receive_cnt, on='Coupon_id', how='left')

    # 优惠券历史核销次数
    coupon_use_cnt = hist[hist['label'] == 1].groupby('Coupon_id')['cnt'].count().reset_index().rename(
        columns={'cnt': 'coupon_historical_use_cnt'})
    label = pd.merge(label, coupon_use_cnt, on='Coupon_id', how='left')

    # 优惠券历史核销率
    label['coupon_historical_use_rate'] = label['coupon_historical_use_cnt'] / label['coupon_historical_receive_cnt'].replace(0, np.nan)

    label.drop(['cnt'], axis=1, inplace=True)
    return label

def get_interaction_feature(label_field, history_field):
    label = label_field.copy()
    hist = history_field.copy()
    hist['cnt'] = 1

    # 用户-商家交互
    user_merchant_receive = hist.groupby(['User_id', 'Merchant_id'])['cnt'].count().reset_index().rename(
        columns={'cnt': 'user_merchant_historical_receive_cnt'})
    label = pd.merge(label, user_merchant_receive, on=['User_id', 'Merchant_id'], how='left')

    user_merchant_use = hist[hist['label'] == 1].groupby(['User_id', 'Merchant_id'])['cnt'].count().reset_index().rename(
        columns={'cnt': 'user_merchant_historical_use_cnt'})
    label = pd.merge(label, user_merchant_use, on=['User_id', 'Merchant_id'], how='left')

    label['user_merchant_historical_use_rate'] = label['user_merchant_historical_use_cnt'] / label['user_merchant_historical_receive_cnt'].replace(0, np.nan)

    # 用户-优惠券交互
    user_coupon_receive = hist.groupby(['User_id', 'Coupon_id'])['cnt'].count().reset_index().rename(
        columns={'cnt': 'user_coupon_historical_receive_cnt'})
    label = pd.merge(label, user_coupon_receive, on=['User_id', 'Coupon_id'], how='left')

    user_coupon_use = hist[hist['label'] == 1].groupby(['User_id', 'Coupon_id'])['cnt'].count().reset_index().rename(
        columns={'cnt': 'user_coupon_historical_use_cnt'})
    label = pd.merge(label, user_coupon_use, on=['User_id', 'Coupon_id'], how='left')

    label['user_coupon_historical_use_rate'] = label['user_coupon_historical_use_cnt'] / label['user_coupon_historical_receive_cnt'].replace(0, np.nan)

    hist.drop(['cnt'], axis=1, inplace=True)
    return label

def get_discount_feature(label_field):
    data = label_field.copy()
    if 'discount_type' not in data.columns:
        data['discount_type'] = data['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    return data

def get_week_feature(label_field):
    data = label_field.copy()
    if 'date_received' in data.columns:
        data['weekday_received'] = data['date_received'].dt.dayofweek
        data['is_weekend'] = data['weekday_received'].apply(lambda x: 1 if x in [5, 6] else 0)
        weekday_dummies = pd.get_dummies(data['weekday_received'], prefix='weekday')
        data = pd.concat([data, weekday_dummies], axis=1)
        data.drop(['weekday_received'], axis=1, inplace=True)
    return data

def get_dataset(history_field, middle_field, label_field):
    # 特征工程组合 - 直接在label_field上逐步添加特征
    label_field = label_field.copy()
    
    # 顺序添加特征
    label_field = get_week_feature(label_field)
    label_field = get_simple_feature(label_field)
    label_field = get_historical_feature(history_field, label_field)
    label_field = get_interaction_feature(label_field, history_field)
    label_field = get_discount_feature(label_field)
    
    # 清理无用列
    dataset = label_field.copy()
    
    if 'Date' in dataset.columns.tolist():
        # 训练集/验证集
        drop_cols = ['Merchant_id', 'Discount_rate', 'Date', 'date_received', 'date']
        drop_cols = [col for col in drop_cols if col in dataset.columns]
        dataset.drop(drop_cols, axis=1, inplace=True)
        
        # 确保 label 在最后一列
        if 'label' in dataset.columns:
            label = dataset['label']
            dataset.drop(['label'], axis=1, inplace=True)
            dataset['label'] = label
    else:
        # 测试集
        drop_cols = ['Merchant_id', 'Discount_rate', 'date_received']
        drop_cols = [col for col in drop_cols if col in dataset.columns]
        dataset.drop(drop_cols, axis=1, inplace=True)
    
    # 修正数据类型
    if 'User_id' in dataset.columns:
        dataset['User_id'] = dataset['User_id'].map(int)
    if 'Coupon_id' in dataset.columns:
        dataset['Coupon_id'] = dataset['Coupon_id'].map(int)
    if 'Date_received' in dataset.columns:
        dataset['Date_received'] = dataset['Date_received'].map(int)
    if 'Distance' in dataset.columns:
        dataset['Distance'] = dataset['Distance'].map(int)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['label'].map(int)
    
    # 去重
    dataset.drop_duplicates(keep='first', inplace=True)
    dataset.index = range(len(dataset))
    
    return dataset

def model_xgb_cv(train, test, n_folds=5):
    # 优化后的XGB参数
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'silent': 1,
        'eta': 0.01,
        'max_depth': 6,
        'min_child_weight': 3,
        'gamma': 0.1,
        'lambda': 3,
        'colsample_bylevel': 0.8,
        'colsample_bytree': 0.8,
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'seed': 2024
    }
    
    # 特征列
    feature_cols = [col for col in train.columns if col not in ['User_id', 'Coupon_id', 'Date_received', 'label']]
    
    # 交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2024)
    train_preds = np.zeros(len(train))
    test_preds = np.zeros((len(test), n_folds))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        X_train, X_val = train.iloc[train_idx][feature_cols], train.iloc[val_idx][feature_cols]
        y_train, y_val = train.iloc[train_idx]['label'], train.iloc[val_idx]['label']
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(test[feature_cols])
        
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(params, dtrain, num_boost_round=3500, evals=watchlist, early_stopping_rounds=50, verbose_eval=False)
        
        train_preds[val_idx] = model.predict(dval)
        test_preds[:, fold] = model.predict(dtest)
    
    # 计算验证集AUC
    auc_score = roc_auc_score(train['label'], train_preds)
    print(f'CV AUC Score: {auc_score:.6f}')
    
    # 取测试集预测均值
    test_pred = test_preds.mean(axis=1)
    
    # 处理结果
    result = pd.DataFrame({
        'User_id': test['User_id'].values,
        'Coupon_id': test['Coupon_id'].values,
        'Date_received': test['Date_received'].values,
        'Probability': test_pred
    })
    return result, auc_score

def model_xgb_full(train, test):
    # 全量训练
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'silent': 1,
        'eta': 0.01,
        'max_depth': 6,
        'min_child_weight': 3,
        'gamma': 0.1,
        'lambda': 3,
        'colsample_bylevel': 0.8,
        'colsample_bytree': 0.8,
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'seed': 2024
    }
    
    feature_cols = [col for col in train.columns if col not in ['User_id', 'Coupon_id', 'Date_received', 'label']]
    
    dtrain = xgb.DMatrix(train[feature_cols], label=train['label'])
    dtest = xgb.DMatrix(test[feature_cols])
    
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=3500, evals=watchlist, verbose_eval=False)
    
    test_pred = model.predict(dtest)
    
    result = pd.DataFrame({
        'User_id': test['User_id'].values,
        'Coupon_id': test['Coupon_id'].values,
        'Date_received': test['Date_received'].values,
        'Probability': test_pred
    })
    return result

if __name__ == '__main__':
    # 源数据
    off_train = pd.read_csv(r'E:\Baseline_o2o\ccf_offline_stage1_train.csv')
    off_test = pd.read_csv(r'E:\Baseline_o2o\ccf_offline_stage1_test_revised.csv')
    
    # 预处理
    off_train = prepare(off_train)
    off_test = prepare(off_test)
    
    # 打标
    off_train = get_label(off_train)

    # 划分区间
    # 训练集
    train_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/2', periods=60))]  # [20160302,20160501)
    train_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/5/1', periods=15))]  # [20160501,20160516)
    train_label_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]  # [20160516,20160616)
    
    # 验证集
    validate_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/1/16', periods=60))]  # [20160116,20160316)
    validate_middle_field = off_train[
        off_train['date'].isin(pd.date_range('2016/3/16', periods=15))]  # [20160316,20160331)
    validate_label_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/31', periods=31))]  # [20160331,20160501)
    
    # 测试集
    test_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/4/17', periods=60))]  # [20160417,20160616)
    test_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/6/16', periods=15))]  # [20160616,20160701)
    test_label_field = off_test.copy()  # [20160701,20160801)

    # 构造数据集
    print('构造训练集...')
    train = get_dataset(train_history_field, train_middle_field, train_label_field)
    print(f'训练集大小: {train.shape}')
    
    print('构造验证集...')
    validate = get_dataset(validate_history_field, validate_middle_field, validate_label_field)
    print(f'验证集大小: {validate.shape}')
    
    print('构造测试集...')
    test = get_dataset(test_history_field, test_middle_field, test_label_field)
    print(f'测试集大小: {test.shape}')

    # 合并训练集和验证集用于最终训练
    big_train = pd.concat([train, validate], axis=0)
    print(f'合并后训练集大小: {big_train.shape}')
    
    print('使用交叉验证训练模型...')
    result_cv, auc_cv = model_xgb_cv(big_train, test, n_folds=5)
    
    print('使用全量数据训练模型...')
    result_full = model_xgb_full(big_train, test)
    
    # 可选：模型融合（简单平均）
    result_cv['Probability'] = (result_cv['Probability'] + result_full['Probability']) / 2
    
    # 保存结果文件
    result_cv.to_csv(r'E:\Baseline_o2o\enhanced_result.csv', index=False, header=None)
    print('结果已保存至 enhanced_result.csv')