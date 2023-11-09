import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class Prepare:
    def __init__(self, data, param=None):
        self.__data = data
        self.__param = param

    def SplitTarget(self):
        X = self.__data.drop(['activity'], axis = 1)
        Y = self.__data['activity']
        return X, Y
    
    def Deduplicate(self):
        duplicates = self.__data.duplicated(subset=self.__data.columns)
        dd = self.__data[~duplicates]
        #print(dd)
        return dd
    
    def Merge(self):
        X = self.SplitTarget()[0]
        y = self.SplitTarget()[1]
        total = X
        total['activity'] = y
        class_counts = y.value_counts()
        class_percents = class_counts * 100 / sum(class_counts)
        threshold = self.__param
        low_freq = class_percents[class_percents < threshold].index
        small = total[total['activity'].isin(low_freq)]
        y_merged = y.apply(lambda x: 99 if x in low_freq else x)
        large = X
        large['activity'] = y_merged
        return large, small
    
    def Balance(self):
        X = self.SplitTarget()[0]
        y = self.SplitTarget()[1]
        
        if self.__param == 1:
            smote = SMOTE()
            X_balance, y_balance = smote.fit_resample(X, y)
        elif self.__param == 2:
            undersample = NearMiss()
            X_balance, y_balance = undersample.fit_resample(X, y)
        else:
            X_balance, y_balance = (X, y)
            
        balanced_data = pd.DataFrame(X_balance, columns=X.columns)
        balanced_data['activity'] = y_balance   
        return balanced_data
    
    def FeatSelect(self):
        X = self.SplitTarget()[0]
        y = self.SplitTarget()[1]
        selector = SelectKBest(score_func=f_classif, k=self.__param)  # Select top features based on ANOVA F-value
        X_sel = selector.fit_transform(X, y)
        sel_index = selector.get_support(indices=True) # Get the selected features indices
        sel_feat = list(X.columns[sel_index]) # Get the selected features names
        print("Selected Features:", sel_feat)
        return X.iloc[:, sel_index], y, sel_feat
    
    def PCA2(self):
        X = self.SplitTarget()[0]
        y = self.SplitTarget()[1]
        pca = PCA(n_components=self.__param)
        X_pca = pca.fit_transform(X)
        PCA_df = pd.DataFrame(data = X_pca)
        #PCA_df = pd.concat([PCA_df, y], axis = 1)
        return PCA_df, y
    
    def Split(self):
        X = self.SplitTarget()[0]
        y = self.SplitTarget()[1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.__param, random_state=0)
        return X_train, X_test, y_train, y_test
    
    def Standard(self):
        Xtrn = self.__data
        Xtst = self.__param
        scaler = StandardScaler()
        X_train = scaler.fit_transform(Xtrn)
        X_test = scaler.fit_transform(Xtst)
        return X_train, X_test
    
    def Normal(self):
        Xtrn = self.__data
        Xtst = self.__param
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(Xtrn)
        X_test = scaler.transform(Xtst)
        return X_train, X_test