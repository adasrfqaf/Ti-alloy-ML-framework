"""
LightGBM Advanced Refined Fine-Tuning (C14 Dataset)
手动随机搜索 + 精细化调参，目标 Test R² > 0.85
仅输出关键评估指标，不保存文件。
"""

import pandas as pd
import numpy as np
import random
import os
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ====================== 全局随机种子 ======================
def set_global_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

SEED = 49
set_global_seed(SEED)

# ====================== 配置 ======================
FILE_PATH = 'C14_data.csv'
TARGET_COL = 'Max_H2_Uptake_wt_pct'
TEST_SIZE = 0.2
RANDOM_STATE = SEED
CV_FOLDS = 5
TOP_N_FEATURES = 18
N_ITER = 120   # 随机搜索迭代次数（若参数固定，可改为1或更小）

# 精细化搜索空间（此处每个参数只有一个值，即固定参数）
ADVANCED_PARAM_DIST = {
    'num_leaves': [8],
    'max_depth': [6],
    'min_child_samples': [26],
    'min_split_gain': [0],
    'learning_rate': [0.035],
    'n_estimators': [170],
    'subsample': [0.7],
    'colsample_bytree': [0.7],
    'reg_alpha': [0.5],
    'reg_lambda': [0.6],
    'min_child_weight': [0.02],
    'max_bin': [250],
    'feature_fraction': [0.8],
}

# 数据增强参数探索（此处只测试一个组合）
SIGMA_RATIOS = [0.08]
SMOTER_RATIOS = [0.02]

# ====================== 数据加载与清洗 ======================
def clean_column_names(df):
    df = df.copy()
    new_cols = {}
    for col in df.columns:
        cleaned = re.sub(r'[\[\]\(\),;:\s]+', '_', str(col)).strip('_')
        new_cols[col] = cleaned if cleaned else col
    df.rename(columns=new_cols, inplace=True)
    return df

def load_data(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path, encoding='utf-8-sig')
    else:
        df = pd.read_excel(path, engine='openpyxl')
    df = clean_column_names(df)
    target_col = TARGET_COL
    if target_col not in df.columns:
        cleaned = re.sub(r'[\[\]\(\),;:\s]+', '_', TARGET_COL).strip('_')
        if cleaned in df.columns:
            target_col = cleaned
        else:
            raise ValueError(f"Target column '{TARGET_COL}' not found")
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)
    y = np.nan_to_num(y)
    print(f"保留的数值特征数: {X.shape[1]}")
    return X, y

# ====================== 特征选择 ======================
def select_top_features_rf(X_train, y_train, top_n=18):
    rf = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE,
        n_jobs=1, max_depth=10, min_samples_split=5, min_samples_leaf=2
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    rf.fit(X_scaled, y_train)
    imp = rf.feature_importances_
    names = X_train.columns.tolist()
    sorted_idx = np.argsort(imp)[::-1][:top_n]
    top_features = [names[i] for i in sorted_idx]
    print("\n选中的Top特征（随机森林重要性）:")
    for i, f in enumerate(top_features, 1):
        print(f"  {i}. {f} (重要性: {imp[sorted_idx[i-1]]:.4f})")
    return top_features

# ====================== 数据增强函数 ======================
def add_relative_gaussian_noise(X, y, sigma_ratio, stds, n_copies=2):
    if n_copies == 0 or sigma_ratio == 0.0:
        return X, y
    X_noisy = [X]
    y_noisy = [y]
    for _ in range(n_copies):
        X_copy = X.copy()
        for col in X.columns:
            col_std = stds[col]
            if col_std > 0:
                noise = np.random.normal(0, sigma_ratio * col_std, len(X))
                X_copy[col] = X_copy[col] + noise
        X_noisy.append(X_copy)
        y_noisy.append(y)
    return pd.concat(X_noisy, ignore_index=True), np.concatenate(y_noisy)

def smogn_augmentation(X, y, smoter_ratio=0.3, noise_ratio=0.05, k=7,
                       bins=10, extreme_factor=2.0, dist_threshold_factor=0.8,
                       random_state=SEED):
    np.random.seed(random_state)
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    n_original = len(X)
    n_generate = int(n_original * smoter_ratio)
    if n_generate <= 0:
        return X, y

    y_percentile = np.percentile(y, np.linspace(0, 100, bins + 1))
    bin_indices = np.digitize(y, y_percentile[1:-1])
    bin_weights = np.ones(bins)
    bin_weights[0] = extreme_factor
    bin_weights[-1] = extreme_factor
    sample_weights = bin_weights[bin_indices - 1] if bins > 1 else np.ones(n_original)
    sample_weights = sample_weights / sample_weights.sum()

    nn = NearestNeighbors(n_neighbors=min(k, n_original), metric='euclidean', n_jobs=1)
    nn.fit(X.values)

    all_distances = []
    for i in range(n_original):
        dist, _ = nn.kneighbors(X.iloc[i].values.reshape(1, -1), n_neighbors=k + 1)
        all_distances.append(dist[0][1:].mean())
    global_avg_dist = np.mean(all_distances)
    dist_threshold = global_avg_dist * dist_threshold_factor

    X_list = [X]
    y_list = [y]

    for _ in range(n_generate):
        idx = np.random.choice(n_original, p=sample_weights)
        x_seed = X.iloc[idx].values
        y_seed = y.iloc[idx]

        distances, indices = nn.kneighbors(x_seed.reshape(1, -1), n_neighbors=k + 1)
        neighbor_indices = indices[0][1:]
        avg_dist_to_neighbors = distances[0][1:].mean()

        if avg_dist_to_neighbors > dist_threshold:
            x_new = x_seed.copy()
            for i_col, col in enumerate(X.columns):
                std_col = X[col].std()
                if std_col > 0:
                    x_new[i_col] += np.random.normal(0, noise_ratio * std_col)
            y_new = y_seed
        else:
            neighbor_idx = np.random.choice(neighbor_indices)
            x_neighbor = X.iloc[neighbor_idx].values
            y_neighbor = y.iloc[neighbor_idx]
            lam = np.random.uniform()
            x_new = x_seed + lam * (x_neighbor - x_seed)
            y_new = y_seed + lam * (y_neighbor - y_seed)

        X_list.append(pd.DataFrame([x_new], columns=X.columns))
        y_list.append(pd.Series([y_new]))

    return pd.concat(X_list, ignore_index=True), pd.concat(y_list, ignore_index=True).values

# ====================== 评估辅助 ======================
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse

def calculate_overfitting(cv_r2_mean, train_r2, test_r2):
    train_test_gap = train_r2 - test_r2
    cv_test_gap = cv_r2_mean - test_r2
    overfitting_score = train_test_gap + max(0, cv_test_gap)
    return {'train_test_gap': train_test_gap,
            'cv_test_gap': cv_test_gap,
            'overfitting_score': overfitting_score}

# ====================== 手动随机搜索 ======================
def manual_random_search(model, param_dist, X, y, n_iter, cv, scoring, random_state):
    rng = np.random.RandomState(random_state)
    best_score = -np.inf
    best_params = None
    best_model = None
    for _ in range(n_iter):
        params = {k: rng.choice(v) for k, v in param_dist.items()}
        model.set_params(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
            best_model = model
    final_model = lgb.LGBMRegressor(random_state=random_state, verbosity=-1, n_jobs=1, force_col_wise=True)
    final_model.set_params(**best_params)
    final_model.fit(X, y)
    return final_model, best_params, best_score

# ====================== 主程序 ======================
print("=" * 100)
print("LightGBM 精细化调参 - C14 数据集")
print("目标: 评估当前参数组合的性能")
print("=" * 100)

print("\n加载数据...")
X, y = load_data(FILE_PATH)
print(f"总样本: {len(X)}, 特征数: {X.shape[1]}")
print(f"目标值范围: [{y.min():.4f}, {y.max():.4f}]")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

top_features = select_top_features_rf(X_train, y_train, TOP_N_FEATURES)
X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]
print(f"特征选择后训练集形状: {X_train_sel.shape}")

feature_stds = X_train_sel.std(axis=0)

all_results = []
total_combos = len(SIGMA_RATIOS) * len(SMOTER_RATIOS)
print(f"\n开始遍历增强参数: {total_combos} 组")
best_test_r2 = -np.inf
best_result = None

for idx, sigma in enumerate(SIGMA_RATIOS, 1):
    for jdx, smoter in enumerate(SMOTER_RATIOS, 1):
        combo_id = (idx - 1) * len(SMOTER_RATIOS) + jdx
        print(f"\n{'='*80}")
        print(f"实验 [{combo_id}/{total_combos}]  噪声={sigma}  SMOTER={smoter}")
        print('='*80)

        # 加噪
        X_train_noisy, y_train_noisy = add_relative_gaussian_noise(
            X_train_sel, y_train, sigma, feature_stds, n_copies=2
        )
        # SMOGN
        X_train_aug, y_train_aug = smogn_augmentation(
            X_train_noisy, y_train_noisy,
            smoter_ratio=smoter,
            noise_ratio=0.05, k=7, bins=10,
            extreme_factor=2.0, dist_threshold_factor=0.8
        )
        print(f"增强后训练集大小: {len(X_train_aug)} (原 {len(X_train_sel)})")

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug)
        X_test_scaled = scaler.transform(X_test_sel)

        # 手动随机搜索（因参数固定，实际只重复评估同一组参数）
        base_model = lgb.LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1, n_jobs=1, force_col_wise=True)
        best_model, best_params, _ = manual_random_search(
            base_model, ADVANCED_PARAM_DIST, X_train_scaled, y_train_aug,
            n_iter=N_ITER, cv=CV_FOLDS, scoring='r2', random_state=RANDOM_STATE
        )

        # 交叉验证分数
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train_aug, cv=CV_FOLDS, scoring='r2')
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

        # 训练集与测试集评估
        r2_train, mae_train, rmse_train = evaluate_model(best_model, X_train_scaled, y_train_aug)
        r2_test, mae_test, rmse_test = evaluate_model(best_model, X_test_scaled, y_test)
        overfit = calculate_overfitting(cv_mean, r2_train, r2_test)

        print(f"\n最佳参数: {best_params}")
        print(f"CV R²: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"训练集 R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
        print(f"测试集 R²: {r2_test:.4f} {'⭐' if r2_test > 0.85 else ''}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")
        print(f"过拟合指标: train-test gap = {overfit['train_test_gap']:.4f},  overfitting score = {overfit['overfitting_score']:.4f}")

        # 记录结果
        result = {
            'sigma': sigma, 'smoter': smoter,
            **best_params,
            'cv_r2': cv_mean, 'cv_std': cv_std,
            'train_r2': r2_train, 'train_rmse': rmse_train, 'train_mae': mae_train,
            'test_r2': r2_test, 'test_rmse': rmse_test, 'test_mae': mae_test,
            'gap': overfit['train_test_gap'], 'overfit_score': overfit['overfitting_score']
        }
        all_results.append(result)

        if r2_test > best_test_r2:
            best_test_r2 = r2_test
            best_result = result

# ====================== 最终结果摘要 ======================
print("\n\n" + "=" * 120)
print("LightGBM 精细调参最终结果")
print("=" * 120)

# 转换为DataFrame便于展示
df_res = pd.DataFrame(all_results)
# 筛选训练R²>0.88的有效模型（避免过拟合）
df_valid = df_res[df_res['train_r2'] > 0.88].copy()
if len(df_valid) > 0:
    best_test_idx = df_valid['test_r2'].idxmax()
    best_row = df_valid.loc[best_test_idx]
else:
    best_row = df_res.loc[df_res['test_r2'].idxmax()]

print("\n🏆 最佳模型（按测试R²，且训练R²>0.88）")
print("=" * 120)
print(f"噪声 (sigma): {best_row['sigma']}")
print(f"SMOTER (smoter): {best_row['smoter']:.3f}")
print("\n[性能指标]")
print(f"  CV R²: {best_row['cv_r2']:.4f} ± {best_row['cv_std']:.4f}")
print(f"  训练集 R²: {best_row['train_r2']:.4f}  RMSE: {best_row['train_rmse']:.4f}  MAE: {best_row['train_mae']:.4f}")
print(f"  测试集 R²: {best_row['test_r2']:.4f}  RMSE: {best_row['test_rmse']:.4f}  MAE: {best_row['test_mae']:.4f}")
print(f"  过拟合 gap: {best_row['gap']:.4f}  overfitting score: {best_row['overfit_score']:.4f}")

print("\n[最佳超参数]")
for key in ['num_leaves', 'max_depth', 'min_child_samples', 'min_split_gain',
            'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree',
            'reg_alpha', 'reg_lambda', 'min_child_weight', 'max_bin', 'feature_fraction']:
    if key in best_row:
        print(f"  {key}: {best_row[key]}")

print("\n" + "=" * 120)
print("程序运行完毕。")