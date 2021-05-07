import pandas as pd
from pandas.io.pickle import read_pickle
pd.set_option('display.max_rows', 50)
import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import style
plt.style.use(['seaborn-darkgrid','seaborn-poster'])
plt.rcParams['figure.figsize'] = [13, 6.5]
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import yfinance as yf
from sklearn.tree import export_graphviz
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
from datetime import datetime
# from .web_plotRoc import plot_roc

from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.style.use('seaborn') 
sm, med, lg = 10, 15, 25
plt.rc('font', size = sm)         # controls default text sizes
plt.rc('axes', titlesize = med)   # fontsize of the axes title
plt.rc('axes', labelsize = med)   # fontsize of the x & y labels
plt.rc('xtick', labelsize = sm)   # fontsize of the tick labels
plt.rc('ytick', labelsize = sm)   # fontsize of the tick labels
plt.rc('legend', fontsize = sm)   # legend fontsize
plt.rc('figure', titlesize = lg)  # fontsize of the figure title
plt.rc('axes', linewidth=2)       # linewidth of plot lines
plt.rcParams['figure.figsize'] = [15, 6.5]
plt.rcParams['figure.dpi'] = 134
    

class The_Random_Forest(object):
    def __init__(self, symbols, file):
        self.file = file
        self.tickers=symbols
        self.mkt_index = '^GSPC'


    def collect_data(self):
        self.component_hist = yf.download(self.tickers, period='1y')['Adj Close']
        self.index_hist = yf.download(self.mkt_index, period='1y')['Adj Close']


    def clean_data(self):
        self.collect_data()
        self.component_df = pd.DataFrame(self.component_hist)
        self.component_df[self.file+'_idx'] = self.index_hist
        weights = np.arange(1,16)
        self.component_df['wma15'] = self.component_df[self.file+'_idx'].rolling(15).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
        self.component_df['GL'] = self.component_df[self.file+'_idx'] >= self.component_df['wma15']
        self.component_df = self.component_df.drop((self.file+'_idx'), axis=1)
        self.component_df = self.component_df.drop(('wma15'), axis=1)
        self.component_df.fillna(0.0, inplace=True)


    def score(self):
        self.clean_data()
        self.y = self.component_df.pop('GL').values
        self.X = self.component_df.values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        self.rf = RandomForestClassifier()
        self.rf.fit(self.X_train, self.y_train)
        st.write("\n * score:", self.rf.score(self.X_test, self.y_test))
        self.y_predict = self.rf.predict(self.X_test)
        st.write("\n * confusion matrix:")
        st.write(confusion_matrix(self.y_test, self.y_predict))
        st.write("\n * precision:", precision_score(self.y_test, self.y_predict))
        st.write(" * recall:", recall_score(self.y_test, self.y_predict))
        rf = RandomForestClassifier(n_estimators=30, oob_score=True)
        rf.fit(self.X_train, self.y_train)
        st.write("\n * accuracy score:", rf.score(self.X_test, self.y_test))
        st.write(" * out of bag score:", rf.oob_score_)
        return self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test


    def feature_importance(self):
        self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test = self.score()
        feature_importances = np.argsort(self.rf.feature_importances_)
        st.write("\n * top five:", list(self.component_df.columns[feature_importances[-1:-6:-1]]))
        present = pd.DataFrame()
        present['tickers'] = list(self.component_df.columns[feature_importances[-1::-1]])
      # top 10 features
        n = len(self.component_df.columns) 
      # importances = forest_fit.feature_importances_[:n]
        importances = self.rf.feature_importances_[:n]
        std = np.std([tree.feature_importances_ for tree in self.rf.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        features = list(self.component_df.columns[indices])
      # st.write the feature ranking
        st.write("\n * Feature ranking:")
        for f in range(n):
            st.write("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))
      # Plot the feature importances of the forest
        fig, ax = plt.subplots()
        ax.bar(range(n), importances[indices], yerr=std[indices], color="r", align="center")
        ax.set_xticks(range(n))
        ax.set_xticklabels(features, rotation = 60)
        ax.set_xlim([-1, n])
        ax.set_xlabel("importance")
        ax.set_title("Feature Importances")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


    def trees(self):
        self.feature_importance()
        num_trees = range(5, 100, 5)
        accuracies = []
        for n in num_trees:
            tot = 0
            for i in range(5):
                rf = RandomForestClassifier(n_estimators=n)
                rf.fit(self.X_train, self.y_train)
                tot += rf.score(self.X_test, self.y_test)
            accuracies.append(tot / 5)
        tree_prediction = rf.predict(self.X_test)
        fig, ax = plt.subplots()
        ax.plot(num_trees, accuracies)
        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("Accuracy")
        ax.set_title('Accuracy vs Num Trees')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
      # Pull out one tree from the forest
        tree = rf.estimators_[10]
      # Export the image to a dot file
        export_graphviz(
          tree, 
        #   out_file = f'/home/gordon/one/report/portfolio_{today}/I_wideView/tree.png', 
          feature_names=self.tickers, 
          rounded = True, 
          precision = 1
        )
      # Use dot file to create a graph
        # (graph, ) = pydot.graph_from_dot_file(f'/home/gordon/one/report/portfolio_{today}/I_wideView/tree.png')
      # Write graph to a png file
        # graph.write_png(savePlot / f"rf_tree.png")
        


    def features(self):
        self.trees()
        num_features = range(1, len(self.component_df.columns) + 1)
        accuracies = []
        for n in num_features:
            tot = 0
            for i in range(5):
                rf = RandomForestClassifier(max_features=n)
                rf.fit(self.X_train, self.y_train)
                tot += rf.score(self.X_test, self.y_test)
            accuracies.append(tot / 5)
        fig, ax = plt.subplots()
        ax.plot(num_features, accuracies)
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Accuracy")
        ax.set_title('Accuracy vs Num Features')
        plt.tight_layout()
        st.pyplot(fig)
        return self.X, self.y


    def get_scores(self, classifier, **kwargs):    
        model = classifier(**kwargs)
        model = classifier(**kwargs)
        model.fit(self.X_train, self.y_train)
        self.y_predict = model.predict(self.X_test)
        return model.score(self.X_test, self.y_test), precision_score(self.y_test, self.y_predict), recall_score(self.y_test, self.y_predict)


    def report_scores(self):
        self.features()
        lR = self.get_scores(LogisticRegression)
        dT = self.get_scores(DecisionTreeClassifier)
        rF = self.get_scores(RandomForestClassifier, n_estimators=50, max_features=7)
        nB = self.get_scores(MultinomialNB)
        st.write("\n* * * Model, Accuracy, Precision, Recall * * *")
        st.write(f"\n    Logistic Regression: {lR}" )
        st.write(f"    Decision Tree: {dT}")
        st.write(f"    Random Forest: {rF}")
        st.write(f"    Naive Bayes: {nB}\n")


    def plot_roc(self, file, X, y, clf_class, plot_name, **kwargs):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        n_splits=5
        kf = KFold(n_splits=n_splits, shuffle=True)
        y_prob = np.zeros((len(y),2))
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            clf = clf_class(**kwargs)
            clf.fit(X_train,y_train)
            # Predict probabilities, not classes
            y_prob[test_index] = clf.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        mean_tpr /= n_splits
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        fig, ax = plt.subplots()
        plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random', lw=1.5)
        plt.axvline(x=0.15, color='r', ls='--', lw=2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{plot_name} - Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)

    def plot_plot_roc(self):
        self.report_scores()
        st.write("* * * visualize the roc curve of each model * * * \n")
        self.plot_roc(self.file, self.X, self.y, LogisticRegression, 'Logistic_Regression')
        self.plot_roc(self.file, self.X, self.y, DecisionTreeClassifier, 'Decision_Tree')
        self.plot_roc(self.file, self.X, self.y, RandomForestClassifier, 'Random_Forest', n_estimators=45, max_features=5)        


if __name__ == '__main__':
<<<<<<< HEAD
    # RF = The_Random_Forest(sp100, 'sp100').plot_plot_roc()
=======
>>>>>>> 93935cfc86a9ef88df2bf9334753d688f78a51fc
    dow = [
      'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW', 
      'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 
      'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
      ]


    The_Random_Forest(dow, 'dow').plot_plot_roc()