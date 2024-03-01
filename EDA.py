#Just testing Git
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st
import scipy.stats as ss

class EDA:
    def __init__(self, data):
        self.__data = data

    def mean(self):
        means = {}
        for column in self.__data.columns:
            if self.__data[column].dtype != object:  # Exclude non-numeric columns
                means[column] = round(st.mean(self.__data[column]), 3)
        return means
    
    def median(self):
        medians = {}
        for column in self.__data.columns:
            if self.__data[column].dtype != object:  # Exclude non-numeric columns
                medians[column] = round(st.median(self.__data[column]), 3)
        return medians
    
    def stddev(self):
        stddevs = {}
        for column in self.__data.columns:
            if self.__data[column].dtype != object:  # Exclude non-numeric columns
                stddevs[column] = round(st.stdev(self.__data[column]), 3)
        return stddevs

    def var(self):
        vars = {}
        for column in self.__data.columns:
            if self.__data[column].dtype != object:  # Exclude non-numeric columns
                vars[column] = round(st.variance(self.__data[column]), 3)
        return vars
    
    def min(self):
        mins = {}
        for column in self.__data.columns:
            if self.__data[column].dtype != object:  # Exclude non-numeric columns
                mins[column] = round(min(self.__data[column]), 3)
        return mins
    
    def max(self):
        maxs = {}
        for column in self.__data.columns:
            if self.__data[column].dtype != object:  # Exclude non-numeric columns
                maxs[column] = round(max(self.__data[column]), 3)
        return maxs
    
    def skew(self):
        skews = {}
        for column in self.__data.columns:
            if self.__data[column].dtype != object:  # Exclude non-numeric columns
                skews[column] = round(ss.skew(self.__data[column]), 3)
        return skews
    
    def kurt(self):
        kurts = {}
        for column in self.__data.columns:
            if self.__data[column].dtype != object:  # Exclude non-numeric columns
                kurts[column] = round(ss.kurtosis(self.__data[column]), 3)
        return kurts
    
    def valcount(self):
        display(self.__data['activity'].value_counts())
        values = self.__data['activity'].value_counts().values
        explode = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.3,0.8,1,1.2,1.8,2.3]
        labels = self.__data['activity'].value_counts().index
        plt.pie(values, labels=labels, explode = explode, autopct='%1.1f%%', textprops={'fontsize': 6})
        plt.show()

    def dist(self):
        fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(16, 20))
        fig.subplots_adjust(hspace=0.5)
                                    
        for i, column in enumerate(self.__data.columns[1:18]):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            sns.histplot(data=self.__data, x=column, kde=True, ax=ax, bins=20)
            ax.set_title(column)
        #hist = self.__data.hist(column=self.__data.columns[1:18], xlabelsize=8, ylabelsize=8, layout=(6,3), figsize=(16, 20))               
        plt.show()
            
    def encode(self, data):
        activity_class = LabelEncoder()
        data_encode = self.__data.drop("_id" , axis=1) #Remove the ID column
        #Encode the target variable
        data_encode['activity']=activity_class.fit_transform(data_encode['activity'])
        return data_encode
    
    def correlation(self):
        corr = self.encode(self.__data).corr()
        plt.subplots(figsize=(10,8))
        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True,
            cmap=sns.diverging_palette(250, 10, as_cmap=True), annot_kws={"fontsize": 10}, fmt=".2f")
        plt.show()
        
    def scatter(self): 
        data_encode = self.encode(self.__data).corr()
        sns.pairplot(data_encode)
        plt.show()