import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

__all__ = ["DataNormalize"]

class DataNormalize(object):
    """数据归一化，支持L2归一化和最大-最小归一化。"""
    def __init__(self, norm_type="min_max"):
        """
           参数：
           -----
           norm_type: 归一化类型，可选值为 "l2" 或 "min_max"
        """
        
        self.norm_type = norm_type
        
    def fit(self,X):
        """拟合归一化器，获取数据的最大最小值。
        
           参数：
           -----
           X: 类数组类型，为训练集所有样本的feature值
        """
        continuous_features = X.columns[(X.nunique() > 2)]
        binary_features = X.columns[(X.nunique() <= 2)]
        self.max_X = X.max(axis=0)
        self.min_X = X.min(axis=0)
        self.continuous_features = continuous_features

        self.max_bin_X = X[binary_features].max(axis=0)
        self.min_bin_X = X[binary_features].min(axis=0)
        self.binary_features = binary_features

    def transform(self,X):
        """用归一化器变换数据。
        
           参数：
           -----
           X: 类数组类型，为数据集样本的feature值
        """
        X_copy = X.copy()
        if self.norm_type == "min_max":
            X_copy.loc[:,self.continuous_features] = (X[self.continuous_features] - self.min_X) / (self.max_X - self.min_X)
        elif self.norm_type == "l2":
            X_copy.loc[:,self.continuous_features] = pd.DataFrame(normalize(X[self.continuous_features], axis=0), columns=self.continuous_features)
        else:
            raise ValueError("无效的规范化类型. Expected 'min_max' or 'l2'.")

        # 处理二进制特征，将最大值的数据点变成1，最小值变为0
        for feature in self.binary_features:
            X_copy[feature] = np.where(X_copy[feature] == self.max_bin_X[feature], 1, 
                                    np.where(X_copy[feature] == self.min_bin_X[feature], 0, X_copy[feature]))

        return X_copy
    
    def fit_transform(self,X):
        """拟合并转换训练集数据"""
        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':
    pass
