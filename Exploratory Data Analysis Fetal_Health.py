import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import time

fetal = pd.read_csv("fetal_health.csv")
eda_fetal = fetal.copy(deep=True)

diabetes = pd.read_csv("diabetes.csv")
eda_diabetes = diabetes.copy(deep=True)


# EDA
def EDA_diabetes():
    print(diabetes.head())
    print(diabetes.shape)
    print(diabetes.dtypes)
    print(diabetes.describe())
    print(diabetes.info)

    # Let's take a look at the frequency of each of the labels, 0(No diabetes), 1(Diabetes) or the class distribution
    number_of_occurences = fetal["Outcome"].value_counts()
    print(number_of_occurences)

    print(diabetes.isnull().any())

    # UNIVARIATE ANALYSIS
    columns_list = eda_diabetes.columns

    plt.figure(figsize=(35, 20))
    for position, column in enumerate(columns_list):
        new_position = position + 1
        plt.subplot(3, 3, new_position)
        sns.boxplot(eda_diabetes[column])
        plt.title(column)

    plt.show()

    # Histogram
    plt.figure(figsize=(35, 20))
    for position, column in enumerate(columns_list):
        new_position = position + 1
        plt.subplot(3, 3, new_position)
        sns.histplot(eda_diabetes[column])
        plt.title(column)

    plt.show()

    # Correlation
    correlations = diabetes.corr()
    # print(correlations)

    plt.figure(figsize=(20, 20))
    sns.heatmap(correlations, cmap='RdBu_r', annot=True, vmax=1, vmin=-1)
    plt.show()
def EDA_fetal():
    # Let's first take a copy of the data and then analyze on that
    print(fetal.head())
    print(fetal.shape)
    print(fetal.dtypes)
    print(fetal.describe())
    print(fetal.info)
    # Let's take a look at the frequency of each of the labels, 1(Normal), 2(Suspect)or 3(Pathological) or the class distribution
    number_of_occurences = fetal["fetal_health"].value_counts()
    print(number_of_occurences)
    # There are clearly way more Normal(1655), than Suspect(295) and then Pathological(176)

    # Let's check for missing values
    print(fetal.isnull().any())
    # There are no missing values
    # UNIVARIATE ANALYSIS

    # Univariate Box Plot
    columns_list = eda_fetal.columns
    plt.figure(figsize=(35, 20))
    for position, column in enumerate(columns_list):
        new_position = position + 1
        plt.subplot(4, 6, new_position)
        sns.boxplot(eda_fetal[column])
        plt.title(column)

    plt.show()

    # Histogram
    plt.figure(figsize=(35, 20))
    for position, column in enumerate(columns_list):
        new_position = position + 1
        plt.subplot(4, 6, new_position)
        sns.histplot(eda_fetal[column])
        plt.title(column)

    plt.show()

    # Correlation
    correlations = eda_fetal.corr()
    # print(correlations)

    plt.figure(figsize=(20, 20))
    sns.heatmap(correlations, cmap='RdBu_r', annot=True, vmax=1, vmin=-1)
    plt.show()

def diabetes_preprocess():
    X = eda_fetal.drop("Outcome", axis=1)
    Y = eda_fetal["Outcome"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    return X, Y, X_train, X_test, Y_train, Y_test
def fetal_preprocess():
    # We want to first divide X and Y into test and train
    # X = eda_fetal.values[:, 0:21]
    # Y = eda_fetal.values[:, 21]
    X = eda_fetal.drop("fetal_health", axis=1)
    Y = eda_fetal["fetal_health"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # scaled_information = StandardScaler()
    # scaled_information.fit(X)
    # X = pd.DataFrame(scaled_information.transform(X), index=X.index, columns=X.columns)

    # scaled_information = StandardScaler()
    # scaled_information.fit(X_train)
    # X_train = pd.DataFrame(scaled_information.transform(X_train), index=X_train.index, columns=X_train.columns)
    # X_test = pd.DataFrame(scaled_information.transform(X_test), index=X_test.index, columns=X_test.columns)

    return X, Y, X_train, X_test, Y_train, Y_test

def decision_Tree_stuff_diabetes(X_train, Y_train, X_test, Y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    print(model.score(X_train, Y_train) * 100)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print(accuracy)

    print(classification_report(Y_test, Y_pred))
    # print(confusion_matrix(Y_test, Y_pred))

    conf_mat = confusion_matrix(Y_test, Y_pred)
    conf_dataframe = pd.DataFrame(conf_mat, index=["No Diabetes", "Diabetes"],
                                  columns=["No Diabetes", "Diabetes"])
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_dataframe, annot=True, fmt='g', cmap="crest")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
def decision_Tree_stuff_fetal(X_train, Y_train, X_test, Y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    print(model.score(X_train, Y_train) * 100)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print(accuracy)

    # plt.figure(figsize=(70, 40))
    # tree.plot_tree(model, filled=True)

    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))

    conf_mat = confusion_matrix(Y_test, Y_pred)
    conf_dataframe = pd.DataFrame(conf_mat, index=["Normal", "Suspicious", "Pathological"],
                                  columns=["Normal", "Suspicious", "Pathological"])
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_dataframe, annot=True, fmt='g', cmap="crest")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    # 89.49843260188088
    # criterion = "entropy", random_state = 0, max_depth = 3, min_samples_leaf = 5
    # 92.6332288401254
    # criterion: Any = "gini",
    # splitter: Any = "best",
    # max_depth: Any = None,
    # min_samples_split: Any = 2,
    # min_samples_leaf: Any = 1,
    # min_weight_fraction_leaf: Any = 0.0,
    # max_features: Any = None,
    # random_state: Any = None,
    # max_leaf_nodes: Any = None,
    # min_impurity_decrease: Any = 0.0,
    # class_weight: Any = None,
    # ccp_alpha: Any = 0.0) -> None

def Boost_stuff(X_train, Y_train, X_test, Y_test):
    model = AdaBoostClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("First we get", accuracy)
    # 87.93103448275862

    # After scaling, we get better
    scaled_information = StandardScaler()
    scaled_information.fit(X_train)
    transformed_data_train = scaled_information.transform(X_train)
    transformed_data_test = scaled_information.transform(X_test)
    index_training = X_train.index
    columns_training = X_train.columns
    index_testing = X_test.index
    columns_testing = X_test.columns
    X_train = pd.DataFrame(transformed_data_train, index=index_training, columns=columns_training)
    X_test = pd.DataFrame(transformed_data_test, index=index_testing, columns=columns_testing)

    model = AdaBoostClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("After scaling X data, we get", accuracy)
    # 87.93103448275862

    # Classification Report
    print(classification_report(Y_test, Y_pred))

    # Confusion Matrix
    # conf_mat = confusion_matrix(Y_test, Y_pred)
    # indexing = ["Normal", "Suspicious", "Pathological"]
    # column_names = ["Normal", "Suspicious", "Pathological"]
    # conf_dataframe = pd.DataFrame(conf_mat, index=indexing, columns=column_names)
    # plt.figure(figsize=(6, 6))
    # color_map = sns.color_palette("Greens", 12)
    # sns.heatmap(conf_dataframe, annot=True, annot_kws={'size': 15}, fmt='g', cmap=color_map)
    # plt.ylabel("True label", fontsize=14)
    # plt.xlabel("Predicted label", fontsize=14)
    # plt.title("Confusion Matrix Heatmap", fontsize=18)
    # plt.show()

#     Hyperparameter Tuning manually - n_estimators

#     rates = [10, 20, 30, 40, 50, 60, 100, 150, 200, 300]
#     length_of_rates = len(rates)
#     # accuracy_array_training = np.empty(length_of_neighbors)
#     accuracy_array_testing = np.empty(length_of_rates)
#
#     for i, rate_number in enumerate(rates):
#
#         model = AdaBoostClassifier(n_estimators=rate_number)
#         model.fit(X_train, Y_train)
#         # accuracy_array_training[i] = model.score(X_train, Y_train)
#         accuracy_array_testing[i] = model.score(X_test, Y_test)
#
#     plt.title("AdaBoost with Varying N_Estimator Hyperparameter", fontsize=16)
#     plt.plot(rates, accuracy_array_testing, label="Test Accuracy", color='green')
#     # plt.plot(neighbors, accuracy_array_training, label='Training accuracy')
#     plt.legend()
#     plt.grid(alpha=0.5)
#     plt.xlabel("N Estimator", fontsize=14)
#     plt.ylabel("Accuracy", fontsize=14)
#     plt.show()
#
# #     Looks like at 40, the accuracy is maximum
#     start_time = time.time()
#     model = AdaBoostClassifier(n_estimators=40)
#     model.fit(X_train, Y_train)
#     end_time = time.time()
#     Y_pred = model.predict(X_test)
#     accuracy = accuracy_score(Y_test, Y_pred) * 100
#     print("The best supposed accuracy after hypertuning N_estimators is ", accuracy)
#     # maximum
#     # 88.71473354231975
#     print("The wall clock time for Adaboost in milliseconds is", (end_time - start_time) * 1000)

    #     Hyperparameter Tuning manually - Learning Rate

    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    length_of_rates = len(rates)
    # accuracy_array_training = np.empty(length_of_neighbors)
    accuracy_array_testing = np.empty(length_of_rates)

    for i, rate_number in enumerate(rates):
        model = AdaBoostClassifier(learning_rate=rate_number)
        model.fit(X_train, Y_train)
        # accuracy_array_training[i] = model.score(X_train, Y_train)
        accuracy_array_testing[i] = model.score(X_test, Y_test)

    plt.title("AdaBoost with Varying Learning Rate Hyperparameter", fontsize=16)
    plt.plot(rates, accuracy_array_testing, label="Test Accuracy", color='green')
    # plt.plot(neighbors, accuracy_array_training, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Learning Rate", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.show()

#     The maximum is at 1.4, so with learning rate 1.4 we get
    model = AdaBoostClassifier(learning_rate=1.4)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("With learning rate hyperparameter tuning, we get", accuracy)
    # 89.34169278996865

#     Hyperparameter Tuning with CV
    print("Now the CV begins")
    parameters_to_hypertune = {"n_estimators": [10, 20, 30, 40, 50], "learning_rate": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}
    model = AdaBoostClassifier()
    model_cross_val = GridSearchCV(model, parameters_to_hypertune, cv=5)
    model_cross_val.fit(X, Y)
    accuracy_CV = model_cross_val.best_score_ * 100
    print("The best possible accuracy score from CV ", accuracy_CV)
    print("The best score through CV was obtained using this parameter", model_cross_val.best_params_)

def KNN_stuff_diabetes(X, Y, X_train, Y_train, X_test, Y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("First we get", accuracy)

    # After scaling, we get the same
    scaled_information = StandardScaler()
    scaled_information.fit(X_train)
    transformed_data_train = scaled_information.transform(X_train)
    transformed_data_test = scaled_information.transform(X_test)
    index_training = X_train.index
    columns_training = X_train.columns
    index_testing = X_test.index
    columns_testing = X_test.columns
    X_train = pd.DataFrame(transformed_data_train, index=index_training, columns=columns_training)
    X_test = pd.DataFrame(transformed_data_test, index=index_testing, columns=columns_testing)

    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("After scaling X data, we get", accuracy)

    # Classification Report
    print(classification_report(Y_test, Y_pred))

    # Confusion matrix
    conf_mat = confusion_matrix(Y_test, Y_pred)
    indexing = ["No Diabetes", "Diabetes"]
    column_names = ["No Diabetes", "Diabetes"]
    conf_dataframe = pd.DataFrame(conf_mat, index=indexing, columns=column_names)
    plt.figure(figsize=(6, 6))
    color_map = sns.color_palette("Greens", 12)
    sns.heatmap(conf_dataframe, annot=True, annot_kws={'size': 15}, fmt='g', cmap=color_map)
    plt.ylabel("True label", fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)
    plt.title("Confusion Matrix Heatmap", fontsize=18)
    plt.show()



def KNN_stuff_fetal(X, Y):
    train_size_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    accuracy_train_error = np.empty(len(train_size_array))
    accuracy_test_error = np.empty(len(train_size_array))


    for i, training_size in enumerate(train_size_array):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=training_size, random_state=0)
        model = KNeighborsClassifier()
        model.fit(X_train, Y_train)
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        accuracy_train_error[i] = accuracy_score(Y_train, Y_pred_train) * 100
        accuracy_test_error[i] = accuracy_score(Y_test, Y_pred_test) * 100

        plt.title("KNN accuracy with Varying Training sizes", fontsize=16)
        plt.plot(training_size, accuracy_test_error, label="Test Accuracy", color='green')
        plt.plot(training_size, accuracy_train_error, label='Training accuracy')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.xlabel("Training size", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.show()





    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("First we get", accuracy)
    # 87.93103448275862

    # After scaling, we get the same
    scaled_information = StandardScaler()
    scaled_information.fit(X_train)
    transformed_data_train = scaled_information.transform(X_train)
    transformed_data_test = scaled_information.transform(X_test)
    index_training = X_train.index
    columns_training = X_train.columns
    index_testing = X_test.index
    columns_testing = X_test.columns
    X_train = pd.DataFrame(transformed_data_train, index=index_training, columns=columns_training)
    X_test = pd.DataFrame(transformed_data_test, index=index_testing, columns=columns_testing)

    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("After scaling X data, we get", accuracy)
    # 87.93103448275862

    # Classification Report
    print(classification_report(Y_test, Y_pred))

    # Confusion matrix
    conf_mat = confusion_matrix(Y_test, Y_pred)
    indexing = ["Normal", "Suspicious", "Pathological"]
    column_names = ["Normal", "Suspicious", "Pathological"]
    conf_dataframe = pd.DataFrame(conf_mat, index=indexing, columns=column_names)
    plt.figure(figsize=(6, 6))
    color_map = sns.color_palette("Greens", 12)
    sns.heatmap(conf_dataframe, annot=True, annot_kws={'size': 15}, fmt='g', cmap=color_map)
    plt.ylabel("True label", fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)
    plt.title("Confusion Matrix Heatmap", fontsize=18)
    plt.show()

    # Hypertuning manually


    # Hypertuning manually - N_Neighbors

    # We can create a graph that shows the accuracy for varying values of K
    neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    length_of_neighbors = len(neighbors)
    # accuracy_array_training = np.empty(length_of_neighbors)
    accuracy_array_testing = np.empty(length_of_neighbors)

    for i, neighbor_number in enumerate(neighbors):

        model = KNeighborsClassifier(n_neighbors=neighbor_number)
        model.fit(X_train, Y_train)
        # accuracy_array_training[i] = model.score(X_train, Y_train)
        accuracy_array_testing[i] = model.score(X_test, Y_test)

    plt.title("KNN with Varying N_Neighbor Hyperparameter", fontsize=16)
    plt.plot(neighbors, accuracy_array_testing, label="Test Accuracy", color='green')
    # plt.plot(neighbors, accuracy_array_training, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("N Neighbors", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.show()

    # It's max at 12, so let's do it only at 12
    start_time = time.time()
    model = KNeighborsClassifier(n_neighbors=12)
    model.fit(X_train, Y_train)
    end_time = time.time()
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("The best supposed accuracy after hypertuning N_neighbors is ", accuracy)
    # maximum
    # 90.28213166144201
    print("The wall clock time for KNN in milliseconds is", (end_time - start_time)*1000)

    # Hypertuning with Cross Validation
    parameters_to_hypertune = {"n_neighbors": np.arange(1, 18)}
    model = KNeighborsClassifier()
    model_cross_val = GridSearchCV(model, parameters_to_hypertune, cv=5)
    model_cross_val.fit(X, Y)
    accuracy_CV = model_cross_val.best_score_ * 100
    print("The best possible accuracy score from CV ", accuracy_CV)
    print("The best score through CV was obtained using this parameter", model_cross_val.best_params_)

def Neural_Network_stuff(X_train, Y_train, X_test, Y_test):
    model = MLPClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print(accuracy)
    # 87.46081504702194
    # Works better with scaled data
    # 92.16300940438872
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))


def SVM_stuff(X_train, Y_train, X_test, Y_test):
    model = SVC()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print(accuracy)
    # 83.85579937304075
    # Better with scaling
    # 90.75235109717869
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))


# EDA_fetal()
# X_fetal, Y_fetal, X_train_fetal, X_test_fetal, Y_train_fetal, Y_test_fetal = fetal_preprocess()
# decision_Tree_stuff_fetal(X_train_fetal, Y_train_fetal, X_test_fetal, Y_test_fetal)
# Boost_stuff(X_train_fetal, Y_train_fetal, X_test_fetal, Y_test_fetal)
# KNN_stuff_fetal(X_fetal, Y_fetal, X_train_fetal, Y_train_fetal, X_test_fetal, Y_test_fetal)
# Neural_Network_stuff(X_train_fetal, Y_train_fetal, X_test_fetal, Y_test_fetal)
# SVM_stuff(X_train_fetal, Y_train_fetal, X_test_fetal, Y_test_fetal)

X_diabetes, Y_diabetes, X_train_diabetes, X_test_diabetes, Y_train_dabetes, Y_test_diabetes = diabetes_preprocess()

