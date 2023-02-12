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
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import time
from sklearn.model_selection import learning_curve

import warnings

fetal = pd.read_csv("fetal_health.csv")
eda_fetal = fetal.copy(deep=True)

diabetes = pd.read_csv("diabetes.csv")
eda_diabetes = diabetes.copy(deep=True)


# EDA
def EDA_diabetes():
    print("Diabetes top 5 rowa", diabetes.head())
    print("Shape of diabetes dataset", diabetes.shape)
    print("Diabetes dtypes", diabetes.dtypes)
    print("Description of the diabetes dataset",diabetes.describe())
    print("Infor on the diabetes dataset", diabetes.info)

    # Let's take a look at the frequency of each of the labels, 0(No diabetes), 1(Diabetes) or the class distribution
    number_of_occurences = diabetes["Outcome"].value_counts()
    print("Class imbalance for diabetes are", number_of_occurences)

    print("Check for any empty data cells",diabetes.isnull().any())

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
    print("Fetal top 5 rows", fetal.head())
    print("Shape of Fetal dataset", fetal.shape)
    print("Fetal dtypes", fetal.dtypes)
    print("Description of the Fetal dataset", fetal.describe())
    print("Infor on the Fetal dataset",fetal.info)
    # Let's take a look at the frequency of each of the labels, 1(Normal), 2(Suspect)or 3(Pathological) or the class distribution
    number_of_occurences = fetal["fetal_health"].value_counts()
    print("Class imbalance for Fetal are",number_of_occurences)
    # There are clearly way more Normal(1655), than Suspect(295) and then Pathological(176)

    # Let's check for missing values
    print("Check for any empty data cellsin fetal", fetal.isnull().any())
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
    new = eda_fetal.drop(["fetal_health", "histogram_mode", "histogram_mean", "histogram_median", "histogram_min", "histogram_width"], axis=1)
    correlations = new.corr()
    # print(correlations)

    plt.figure(figsize=(20, 20))
    sns.heatmap(correlations, cmap='RdBu_r', annot=True, vmax=1, vmin=-1)
    plt.show()

def fetal_preprocess():
    X = eda_fetal.drop("fetal_health", axis=1)
    #  "histogram_mode", "histogram_mean", "histogram_median", "histogram_min", "histogram_width"
    Y = eda_fetal["fetal_health"]

    return X, Y
def diabetes_preprocess():
    X = eda_diabetes.drop("Outcome", axis=1)
    Y = eda_diabetes["Outcome"]

    return X, Y

def DT_fetal(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    train_size_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracy_train_error = np.empty(len(train_size_array))
    accuracy_test_error = np.empty(len(train_size_array))

    for i, training_size in enumerate(train_size_array):
        new_X_train = X_train.sample(frac=training_size, replace=False, random_state=0)
        new_Y_train = Y_train[new_X_train.index]

        model = DecisionTreeClassifier()
        model.fit(new_X_train, new_Y_train)
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        accuracy_train_error[i] = accuracy_score(Y_train, Y_pred_train) * 100
        accuracy_test_error[i] = accuracy_score(Y_test, Y_pred_test) * 100

    plt.title("DT Accuracy with Varying Training sizes for Fetal Health", fontsize=15)
    plt.plot(train_size_array, accuracy_test_error, label="Test Accuracy", color='green')
    plt.plot(train_size_array, accuracy_train_error, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Training size", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.savefig("DT_accuracy_with_training_size_FETAL.png")
    plt.close()
    # plt.show()

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

    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    model.fit(X_train, Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    accuracy_score_test = accuracy_score(Y_test, Y_pred_test) * 100
    accuracy_score_train = accuracy_score(Y_train, Y_pred_train) * 100
    print("The accuracy is now", accuracy_score_test)
    print("The difference is now", accuracy_score_train - accuracy_score_test)


    model = DecisionTreeClassifier()
    cost_complexity_pruning_path = model.cost_complexity_pruning_path(X_train, Y_train)
    cost_complexity_pruning_alphas, cost_complexity_pruning_impurities = cost_complexity_pruning_path.ccp_alphas, cost_complexity_pruning_path.impurities

    fig, ax = plt.subplots()
    ax.plot(cost_complexity_pruning_alphas[:-1], cost_complexity_pruning_impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.show()

    print("Pruning has begun")
    cost_pruning_fs = []
    for ccp_alpha in cost_complexity_pruning_alphas:
        model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        model.fit(X_train, Y_train)
        cost_pruning_fs.append(model)

    cost_pruning_fs = cost_pruning_fs[:-1]
    cost_complexity_pruning_alphas = cost_complexity_pruning_alphas[:-1]

    print("pruning has ended")

    count_of_nodes = [clf.tree_.node_count for clf in cost_pruning_fs]
    depth_ranges = [clf.tree_.max_depth for clf in cost_pruning_fs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(cost_complexity_pruning_alphas, count_of_nodes, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(cost_complexity_pruning_alphas, depth_ranges, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.show()

    print("Graphs done")

    train_scores = [clf.score(X_train, Y_train) for clf in cost_pruning_fs]
    test_scores = [clf.score(X_test, Y_test) for clf in cost_pruning_fs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(cost_complexity_pruning_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(cost_complexity_pruning_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()


    alphas = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    accuracy_test_error = np.empty(len(alphas))
    accuracy_train_error = np.empty(len(alphas))

    model = DecisionTreeClassifier(ccp_alpha=0.005)
    model.fit(X_train, Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    accuracy_score_train = accuracy_score(Y_train, Y_pred_train) * 100
    accuracy_score_test = accuracy_score(Y_test, Y_pred_test) * 100


    print("The accuracy is now", accuracy_score_test)
    print("The difference is now", accuracy_score_train - accuracy_score_test)

    for i, alpha in enumerate(alphas):
        model = DecisionTreeClassifier(ccp_alpha=alpha)
        model.fit(X_train, Y_train)
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        accuracy_train_error[i] = accuracy_score(Y_train, Y_pred_train) * 100
        accuracy_test_error[i] = accuracy_score(Y_test, Y_pred_test) * 100

    plt.title("DT accuracy with Varying alphas", fontsize=16)
    plt.plot(alphas, accuracy_test_error, label="Test Accuracy", color='green')
    plt.plot(alphas, accuracy_train_error, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Alpha", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(50, 100)
    # plt.show()
    plt.savefig("DT_accuracy_with_varying_alphas_FETAL.png")
    plt.close()
def DT_diabetes(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    train_size_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracy_train_error = np.empty(len(train_size_array))
    accuracy_test_error = np.empty(len(train_size_array))

    for i, training_size in enumerate(train_size_array):
        new_X_train = X_train.sample(frac=training_size, replace=False, random_state=0)
        new_Y_train = Y_train[new_X_train.index]

        model = DecisionTreeClassifier()
        model.fit(new_X_train, new_Y_train)
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        accuracy_train_error[i] = accuracy_score(Y_train, Y_pred_train) * 100
        accuracy_test_error[i] = accuracy_score(Y_test, Y_pred_test) * 100

    plt.title("DT accuracy with Varying Training sizes", fontsize=16)
    plt.plot(train_size_array, accuracy_test_error, label="Test Accuracy", color='green')
    plt.plot(train_size_array, accuracy_train_error, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Training size", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    # plt.show()
    plt.savefig("DT_accuracy_with_training_size_DIABETES.png")
    plt.close()
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

    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    model.fit(X_train, Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    accuracy_score_test = accuracy_score(Y_test, Y_pred_test) * 100
    accuracy_score_train = accuracy_score(Y_train, Y_pred_train) * 100
    print("The accuracy is now", accuracy_score_test)
    print("The difference is now", accuracy_score_train - accuracy_score_test)

    model = DecisionTreeClassifier()
    cost_complexity_pruning_path = model.cost_complexity_pruning_path(X_train, Y_train)
    cost_complexity_pruning_alphas, cost_complexity_pruning_impurities = cost_complexity_pruning_path.ccp_alphas, cost_complexity_pruning_path.impurities

    fig, ax = plt.subplots()
    ax.plot(cost_complexity_pruning_alphas[:-1], cost_complexity_pruning_impurities[:-1], marker="o",
            drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.show()

    print("Pruning has begun")
    cost_pruning_fs = []
    for ccp_alpha in cost_complexity_pruning_alphas:
        model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        model.fit(X_train, Y_train)
        cost_pruning_fs.append(model)

    cost_pruning_fs = cost_pruning_fs[:-1]
    cost_complexity_pruning_alphas = cost_complexity_pruning_alphas[:-1]

    print("pruning has ended")

    count_of_nodes = [clf.tree_.node_count for clf in cost_pruning_fs]
    depth_ranges = [clf.tree_.max_depth for clf in cost_pruning_fs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(cost_complexity_pruning_alphas, count_of_nodes, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(cost_complexity_pruning_alphas, depth_ranges, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.show()

    print("Graphs done")

    train_scores = [clf.score(X_train, Y_train) for clf in cost_pruning_fs]
    test_scores = [clf.score(X_test, Y_test) for clf in cost_pruning_fs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(cost_complexity_pruning_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(cost_complexity_pruning_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()

    alphas = [0.0, 0.005, 0.0075, 0.01, 0.02, 0.05]
    accuracy_test_error = np.empty(len(alphas))
    accuracy_train_error = np.empty(len(alphas))
    model = DecisionTreeClassifier(ccp_alpha=0.0075)
    model.fit(X_train, Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    accuracy_score_train = accuracy_score(Y_train, Y_pred_train) * 100
    accuracy_score_test = accuracy_score(Y_test, Y_pred_test) * 100

    print("The accuracy is now", accuracy_score_test)
    print("The difference is now", accuracy_score_train - accuracy_score_test)

    for i, alpha in enumerate(alphas):
        model = DecisionTreeClassifier(ccp_alpha=alpha)
        model.fit(X_train, Y_train)
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        accuracy_train_error[i] = accuracy_score(Y_train, Y_pred_train) * 100
        accuracy_test_error[i] = accuracy_score(Y_test, Y_pred_test) * 100

    plt.title("DT accuracy with Varying alphas", fontsize=16)
    plt.plot(alphas, accuracy_test_error, label="Test Accuracy", color='green')
    plt.plot(alphas, accuracy_train_error, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Alpha", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(50, 100)
    plt.savefig("DT_accuracy_with_varying_alphas_DIABETES.png")
    plt.close()
    # plt.show()


def KNN_fetal(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    train_size_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracy_train_error = np.empty(len(train_size_array))
    accuracy_test_error = np.empty(len(train_size_array))

    for i, training_size in enumerate(train_size_array):
        new_X_train = X_train.sample(frac=training_size, replace=False, random_state=0)
        new_Y_train = Y_train[new_X_train.index]

        model = KNeighborsClassifier()
        model.fit(new_X_train, new_Y_train)
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        accuracy_train_error[i] = accuracy_score(Y_train, Y_pred_train) * 100
        accuracy_test_error[i] = accuracy_score(Y_test, Y_pred_test) * 100

    plt.title("KNN accuracy with Varying Training sizes", fontsize=16)
    plt.plot(train_size_array, accuracy_test_error, label="Test Accuracy", color='green')
    plt.plot(train_size_array, accuracy_train_error, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Training size", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(50, 100)
    plt.savefig("KNN accuracy with Varying Training sizes_FETAL.png")
    plt.close()
    # plt.show()

    # model = KNeighborsClassifier()
    # train_size, train_score, test_score = learning_curve(model, X, Y, train_sizes=train_size_array)
    # for train_size, cv_train_scores, cv_test_scores in zip(train_size, train_score, test_score):
    #     print(f"{train_size} samples were used to train the model")
    #     print(f"The average train accuracy is {cv_train_scores.mean():.2f}")
    #     print(f"The average test accuracy is {cv_test_scores.mean():.2f}")

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



    # Hypertuning manually

    # Hypertuning manually - N_Neighbors

    # We can create a graph that shows the accuracy for varying values of K
    neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    length_of_neighbors = len(neighbors)
    accuracy_array_training = np.empty(length_of_neighbors)
    accuracy_array_testing = np.empty(length_of_neighbors)

    for i, neighbor_number in enumerate(neighbors):
        model = KNeighborsClassifier(n_neighbors=neighbor_number)
        model.fit(X_train, Y_train)
        accuracy_array_training[i] = model.score(X_train, Y_train)
        accuracy_array_testing[i] = model.score(X_test, Y_test)

    plt.title("KNN with Varying N_Neighbor Hyperparameter", fontsize=16)
    plt.plot(neighbors, accuracy_array_testing, label="Test Accuracy", color='green')
    plt.plot(neighbors, accuracy_array_training, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("N Neighbors", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.savefig("KNN with Varying N_Neighbor Hyperparameter_FETAL.png")
    plt.close()
    # plt.show()

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
    print("The wall clock time for KNN in milliseconds is", (end_time - start_time) * 1000)

    # Classification Report
    print(classification_report(Y_test, Y_pred))

    # Confusion matrix
    conf_mat = confusion_matrix(Y_test, Y_pred)
    indexing = ["Normal", "Suspect", "Pathological"]
    column_names = ["Normal", "Suspect", "Pathological"]
    conf_dataframe = pd.DataFrame(conf_mat, index=indexing, columns=column_names)
    plt.figure(figsize=(6, 6))
    color_map = sns.color_palette("Greens", 12)
    sns.heatmap(conf_dataframe, annot=True, annot_kws={'size': 15}, fmt='g', cmap=color_map)
    plt.ylabel("True label", fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)
    plt.title("Confusion Matrix Heatmap", fontsize=18)
    plt.show()

    # Hypertuning with Cross Validation
    parameters_to_hypertune = {"n_neighbors": np.arange(1, 18)}
    model = KNeighborsClassifier()
    model_cross_val = GridSearchCV(model, parameters_to_hypertune, cv=5)
    model_cross_val.fit(X, Y)
    accuracy_CV = model_cross_val.best_score_ * 100
    print("The best possible accuracy score from CV ", accuracy_CV)
    print("The best score through CV was obtained using this parameter", model_cross_val.best_params_)
def KNN_diabetes(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    train_size_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracy_train_error = np.empty(len(train_size_array))
    accuracy_test_error = np.empty(len(train_size_array))

    for i, training_size in enumerate(train_size_array):
        new_X_train = X_train.sample(frac=training_size, replace=False, random_state=0)
        new_Y_train = Y_train[new_X_train.index]

        model = KNeighborsClassifier()
        model.fit(new_X_train, new_Y_train)
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        accuracy_train_error[i] = accuracy_score(Y_train, Y_pred_train) * 100
        accuracy_test_error[i] = accuracy_score(Y_test, Y_pred_test) * 100

    plt.title("KNN accuracy with Varying Training sizes", fontsize=16)
    plt.plot(train_size_array, accuracy_test_error, label="Test Accuracy", color='green')
    plt.plot(train_size_array, accuracy_train_error, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Training size", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(50, 100)
    plt.savefig("KNN accuracy with Varying Training sizes_DIABETES.png")
    plt.close()
    # plt.show()

    # model = KNeighborsClassifier()
    # train_size, train_score, test_score = learning_curve(model, X, Y, train_sizes=train_size_array)
    # for train_size, cv_train_scores, cv_test_scores in zip(train_size, train_score, test_score):
    #     print(f"{train_size} samples were used to train the model")
    #     print(f"The average train accuracy is {cv_train_scores.mean():.2f}")
    #     print(f"The average test accuracy is {cv_test_scores.mean():.2f}")

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

    # Hypertuning manually

    # Hypertuning manually - N_Neighbors
    # We can create a graph that shows the accuracy for varying values of K
    neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    length_of_neighbors = len(neighbors)
    accuracy_array_training = np.empty(length_of_neighbors)
    accuracy_array_testing = np.empty(length_of_neighbors)

    for i, neighbor_number in enumerate(neighbors):
        model = KNeighborsClassifier(n_neighbors=neighbor_number)
        model.fit(X_train, Y_train)
        accuracy_array_training[i] = model.score(X_train, Y_train)
        accuracy_array_testing[i] = model.score(X_test, Y_test)

    plt.title("KNN with Varying N_Neighbor Hyperparameter", fontsize=16)
    plt.plot(neighbors, accuracy_array_testing, label="Test Accuracy", color='green')
    plt.plot(neighbors, accuracy_array_training, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("N Neighbors", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.savefig("KNN with Varying N_Neighbor Hyperparameter_DIABETES.png")
    plt.close()
    # plt.show()

    # It's max at 13, so let's do it only at 13
    start_time = time.time()
    model = KNeighborsClassifier(n_neighbors=13)
    model.fit(X_train, Y_train)
    end_time = time.time()
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("The best supposed accuracy after hypertuning N_neighbors is ", accuracy)
    # maximum
    # 90.28213166144201
    print("The wall clock time for KNN in milliseconds is", (end_time - start_time) * 1000)

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

    # Hypertuning with Cross Validation
    parameters_to_hypertune = {"n_neighbors": np.arange(1, 18)}
    model = KNeighborsClassifier()
    model_cross_val = GridSearchCV(model, parameters_to_hypertune, cv=5)
    model_cross_val.fit(X, Y)
    accuracy_CV = model_cross_val.best_score_ * 100
    print("The best possible accuracy score from CV ", accuracy_CV)
    print("The best score through CV was obtained using this parameter", model_cross_val.best_params_)

def Boost_fetal(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    train_size_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracy_train_error = np.empty(len(train_size_array))
    accuracy_test_error = np.empty(len(train_size_array))

    for i, training_size in enumerate(train_size_array):
        new_X_train = X_train.sample(frac=training_size, replace=False, random_state=0)
        new_Y_train = Y_train[new_X_train.index]

        model_dt = DecisionTreeClassifier(ccp_alpha=0.005)

        model = AdaBoostClassifier(estimator=model_dt)
        model.fit(new_X_train, new_Y_train)
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        accuracy_train_error[i] = accuracy_score(Y_train, Y_pred_train) * 100
        accuracy_test_error[i] = accuracy_score(Y_test, Y_pred_test) * 100

    plt.title("Adaboost accuracy with Varying Training sizes", fontsize=16)
    plt.plot(train_size_array, accuracy_test_error, label="Test Accuracy", color='green')
    plt.plot(train_size_array, accuracy_train_error, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Training size", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(60, 100)
    plt.savefig("Adaboost_accuracy_with_Varying_Training_sizes_FETAL.png")
    plt.close()
    model_dt = DecisionTreeClassifier(ccp_alpha=0.005)
    model = AdaBoostClassifier(estimator=model_dt)
    # train_size, train_score, test_score = learning_curve(model, X, Y, train_sizes=train_size_array)
    # for train_size, cv_train_scores, cv_test_scores in zip(train_size, train_score, test_score):
    #     print(f"{train_size} samples were used to train the model")
    #     print(f"The average train accuracy is {cv_train_scores.mean():.2f}")
    #     print(f"The average test accuracy is {cv_test_scores.mean():.2f}")

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

    model_dt = DecisionTreeClassifier(ccp_alpha=0.005)
    model = AdaBoostClassifier(estimator=model_dt)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("After scaling X data, we get", accuracy)

    # # Classification Report
    # print(classification_report(Y_test, Y_pred))
    #
    # # Confusion matrix
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

    # # Hyperparameter Tuning manually - n_estimators
    #
    # rates = [10, 20, 30, 40, 50, 60, 100, 150, 200, 300, 500, 800]
    # length_of_rates = len(rates)
    # accuracy_array_training = np.empty(length_of_rates)
    # accuracy_array_testing = np.empty(length_of_rates)
    #
    # for i, rate_number in enumerate(rates):
    #
    #     model_dt = DecisionTreeClassifier(ccp_alpha=0.005)
    #     model = AdaBoostClassifier(n_estimators=rate_number, estimator=model_dt)
    #     model.fit(X_train, Y_train)
    #     accuracy_array_training[i] = model.score(X_train, Y_train)
    #     accuracy_array_testing[i] = model.score(X_test, Y_test)
    #
    # plt.title("AdaBoost with Varying N_Estimator Hyperparameter", fontsize=16)
    # plt.plot(rates, accuracy_array_testing, label="Test Accuracy", color='green')
    # plt.plot(rates, accuracy_array_training, label='Training accuracy')
    # plt.legend()
    # plt.grid(alpha=0.5)
    # plt.xlabel("N Estimator", fontsize=14)
    # plt.ylabel("Accuracy", fontsize=14)
    # plt.show()
    # plt.savefig("AdaBoost_with_Varying_N_Estimator_Hyperparameter")


    #     Hyperparameter Tuning manually - Learning Rate

    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    length_of_rates = len(rates)
    accuracy_array_training = np.empty(length_of_rates)
    accuracy_array_testing = np.empty(length_of_rates)

    for i, rate_number in enumerate(rates):

        model_dt = DecisionTreeClassifier(ccp_alpha=0.005)
        model = AdaBoostClassifier(learning_rate=rate_number, estimator=model_dt)
        model.fit(X_train, Y_train)
        accuracy_array_training[i] = model.score(X_train, Y_train)
        accuracy_array_testing[i] = model.score(X_test, Y_test)

    plt.title("AdaBoost with Varying Learning Rate Hyperparameter", fontsize=16)
    plt.plot(rates, accuracy_array_testing, label="Test Accuracy", color='green')
    plt.plot(rates, accuracy_array_training, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Learning Rate", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    # plt.show()
    plt.savefig("AdaBoost_with_Varying_Learning_Rate_Hyperparameter_FETAL.png")
    plt.close()

    #     The maximum is at 1.4, so with learning rate 1.4 we get
    model_dt = DecisionTreeClassifier(ccp_alpha=0.005)
    model = AdaBoostClassifier(learning_rate=0.6, estimator=model_dt, n_estimators=300)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("With learning rate hyperparameter tuning, we get", accuracy)
    # 89.34169278996865

    #     Hyperparameter Tuning with CV
    # print("Now the CV begins")
    # parameters_to_hypertune = {"n_estimators": [10, 20, 30, 40, 50],
    #                            "learning_rate": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}
    # model = AdaBoostClassifier()
    # model_cross_val = GridSearchCV(model, parameters_to_hypertune, cv=5)
    # model_cross_val.fit(X, Y)
    # accuracy_CV = model_cross_val.best_score_ * 100
    # print("The best possible accuracy score from CV ", accuracy_CV)
    # print("The best score through CV was obtained using this parameter", model_cross_val.best_params_)
def Boost_diabetes(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    train_size_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracy_train_error = np.empty(len(train_size_array))
    accuracy_test_error = np.empty(len(train_size_array))

    for i, training_size in enumerate(train_size_array):
        new_X_train = X_train.sample(frac=training_size, replace=False, random_state=0)
        new_Y_train = Y_train[new_X_train.index]

        model_dt = DecisionTreeClassifier(ccp_alpha=0.0075)
        model = AdaBoostClassifier(estimator=model_dt)
        model.fit(new_X_train, new_Y_train)
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        accuracy_train_error[i] = accuracy_score(Y_train, Y_pred_train) * 100
        accuracy_test_error[i] = accuracy_score(Y_test, Y_pred_test) * 100

    plt.title("Adaboost accuracy with Varying Training sizes", fontsize=16)
    plt.plot(train_size_array, accuracy_test_error, label="Test Accuracy", color='green')
    plt.plot(train_size_array, accuracy_train_error, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Training size", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    # plt.show()
    plt.savefig("Adaboost_accuracy_with_Varying_Training_sizes_DIABETES.png")
    plt.close()

    # model_dt = DecisionTreeClassifier(ccp_alpha=0.0075)
    # model = AdaBoostClassifier(estimator=model_dt)
    # train_size, train_score, test_score = learning_curve(model, X, Y, train_sizes=train_size_array)
    # for train_size, cv_train_scores, cv_test_scores in zip(train_size, train_score, test_score):
    #     print(f"{train_size} samples were used to train the model")
    #     print(f"The average train accuracy is {cv_train_scores.mean():.2f}")
    #     print(f"The average test accuracy is {cv_test_scores.mean():.2f}")

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

    model_dt = DecisionTreeClassifier(ccp_alpha=0.0075)
    model = AdaBoostClassifier(estimator=model_dt)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("After scaling X data, we get", accuracy)

    # # Classification Report
    # print(classification_report(Y_test, Y_pred))
    #
    # # Confusion matrix
    # conf_mat = confusion_matrix(Y_test, Y_pred)
    # indexing = ["No Diabetes", "Diabetes"]
    # column_names = ["No Diabetes", "Diabetes"]
    # conf_dataframe = pd.DataFrame(conf_mat, index=indexing, columns=column_names)
    # plt.figure(figsize=(6, 6))
    # color_map = sns.color_palette("Greens", 12)
    # sns.heatmap(conf_dataframe, annot=True, annot_kws={'size': 15}, fmt='g', cmap=color_map)
    # plt.ylabel("True label", fontsize=14)
    # plt.xlabel("Predicted label", fontsize=14)
    # plt.title("Confusion Matrix Heatmap", fontsize=18)
    # plt.show()

    # Hyperparameter Tuning manually - n_estimators

    # rates = [10, 20, 30, 40, 50, 60, 100, 150, 200, 300, 400, 500, 600]
    # length_of_rates = len(rates)
    # accuracy_array_training = np.empty(length_of_rates)
    # accuracy_array_testing = np.empty(length_of_rates)
    #
    # for i, rate_number in enumerate(rates):
    #     model_dt = DecisionTreeClassifier(ccp_alpha=0.0075)
    #     model = AdaBoostClassifier(n_estimators=rate_number, estimator=model_dt)
    #     model.fit(X_train, Y_train)
    #     accuracy_array_training[i] = model.score(X_train, Y_train)
    #     accuracy_array_testing[i] = model.score(X_test, Y_test)
    #
    # plt.title("AdaBoost with Varying N_Estimator Hyperparameter", fontsize=16)
    # plt.plot(rates, accuracy_array_testing, label="Test Accuracy", color='green')
    # plt.plot(rates, accuracy_array_training, label='Training accuracy')
    # plt.legend()
    # plt.grid(alpha=0.5)
    # plt.xlabel("N Estimator", fontsize=14)
    # plt.ylabel("Accuracy", fontsize=14)
    # plt.show()

    #     Hyperparameter Tuning manually - Learning Rate

    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    length_of_rates = len(rates)
    accuracy_array_training = np.empty(length_of_rates)
    accuracy_array_testing = np.empty(length_of_rates)

    for i, rate_number in enumerate(rates):
        model_dt = DecisionTreeClassifier(ccp_alpha=0.0075)
        model = AdaBoostClassifier(learning_rate=rate_number, estimator=model_dt)
        model.fit(X_train, Y_train)
        accuracy_array_training[i] = model.score(X_train, Y_train)
        accuracy_array_testing[i] = model.score(X_test, Y_test)

    plt.title("AdaBoost with Varying Learning Rate Hyperparameter", fontsize=16)
    plt.plot(rates, accuracy_array_testing, label="Test Accuracy", color='green')
    plt.plot(rates, accuracy_array_training, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Learning Rate", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.savefig("AdaBoost_with_Varying_Learning_Rate_Hyperparameter_DIABETES.png")
    plt.close()
    # plt.show()

    print("It's done")

    #     Hyperparameter Tuning with CV
    # print("Now the CV begins")
    # parameters_to_hypertune = {"n_estimators": [10, 20, 30, 40, 50],
    #                            "learning_rate": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]}
    # model = AdaBoostClassifier()
    # model_cross_val = GridSearchCV(model, parameters_to_hypertune, cv=5)
    # model_cross_val.fit(X, Y)
    # accuracy_CV = model_cross_val.best_score_ * 100
    # print("The best possible accuracy score from CV ", accuracy_CV)
    # print("The best score through CV was obtained using this parameter", model_cross_val.best_params_)

def NN_fetal(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    train_size_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracy_train_error = np.empty(len(train_size_array))
    accuracy_test_error = np.empty(len(train_size_array))

    for i, training_size in enumerate(train_size_array):
        new_X_train = X_train.sample(frac=training_size, replace=False, random_state=0)
        new_Y_train = Y_train[new_X_train.index]

        model = MLPClassifier()
        model.fit(new_X_train, new_Y_train)
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        accuracy_train_error[i] = accuracy_score(Y_train, Y_pred_train) * 100
        accuracy_test_error[i] = accuracy_score(Y_test, Y_pred_test) * 100

    plt.title("NN accuracy with Varying Training sizes", fontsize=16)
    plt.plot(train_size_array, accuracy_test_error, label="Test Accuracy", color='green')
    plt.plot(train_size_array, accuracy_train_error, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Training size", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.show()

    # model = MLPClassifier()
    # train_size, train_score, test_score = learning_curve(model, X, Y, train_sizes=train_size_array)
    # for train_size, cv_train_scores, cv_test_scores in zip(train_size, train_score, test_score):
    #     print(f"{train_size} samples were used to train the model")
    #     print(f"The average train accuracy is {cv_train_scores.mean():.2f}")
    #     print(f"The average test accuracy is {cv_test_scores.mean():.2f}")

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

    model = MLPClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("After scaling X data, we get", accuracy)

    # Hypertuning with GridSearch
    model = MLPClassifier(max_iter=100)
    parameters_to_tune = {
        'hidden_layer_sizes': [(10, 30, 10), (20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    model_cross_val = GridSearchCV(model, parameters_to_tune, cv=5)
    model_cross_val.fit(X, Y)
    accuracy_CV = model_cross_val.best_score_ * 100
    print("The best possible accuracy score from CV ", accuracy_CV)
    print("The best score through CV was obtained using this parameter", model_cross_val.best_params_)

    # # Classification Report
    # print(classification_report(Y_test, Y_pred))
    #
    # # Confusion matrix
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
def NN_diabetes(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    train_size_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracy_train_error = np.empty(len(train_size_array))
    accuracy_test_error = np.empty(len(train_size_array))

    for i, training_size in enumerate(train_size_array):
        new_X_train = X_train.sample(frac=training_size, replace=False, random_state=0)
        new_Y_train = Y_train[new_X_train.index]

        model = MLPClassifier()
        model.fit(new_X_train, new_Y_train)
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        accuracy_train_error[i] = accuracy_score(Y_train, Y_pred_train) * 100
        accuracy_test_error[i] = accuracy_score(Y_test, Y_pred_test) * 100

    plt.title("NN accuracy with Varying Training sizes", fontsize=16)
    plt.plot(train_size_array, accuracy_test_error, label="Test Accuracy", color='green')
    plt.plot(train_size_array, accuracy_train_error, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Training size", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.show()

    model = MLPClassifier()
    train_size, train_score, test_score = learning_curve(model, X, Y, train_sizes=train_size_array)
    for train_size, cv_train_scores, cv_test_scores in zip(train_size, train_score, test_score):
        print(f"{train_size} samples were used to train the model")
        print(f"The average train accuracy is {cv_train_scores.mean():.2f}")
        print(f"The average test accuracy is {cv_test_scores.mean():.2f}")

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

    model = MLPClassifier()
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

def SVM_fetal(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    train_size_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracy_train_error = np.empty(len(train_size_array))
    accuracy_test_error = np.empty(len(train_size_array))

    for i, training_size in enumerate(train_size_array):
        new_X_train = X_train.sample(frac=training_size, replace=False, random_state=0)
        new_Y_train = Y_train[new_X_train.index]

        model = SVC()
        model.fit(new_X_train, new_Y_train)
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        accuracy_train_error[i] = accuracy_score(Y_train, Y_pred_train) * 100
        accuracy_test_error[i] = accuracy_score(Y_test, Y_pred_test) * 100

    plt.title("SVC accuracy with Varying Training sizes", fontsize=16)
    plt.plot(train_size_array, accuracy_test_error, label="Test Accuracy", color='green')
    plt.plot(train_size_array, accuracy_train_error, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Training size", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(50, 100)
    plt.savefig("SVC accuracy with Varying Training sizes_FETAL.png")
    plt.close()
    # plt.show()

    # model = SVC()
    # train_size, train_score, test_score = learning_curve(model, X, Y, train_sizes=train_size_array)
    # for train_size, cv_train_scores, cv_test_scores in zip(train_size, train_score, test_score):
    #     print(f"{train_size} samples were used to train the model")
    #     print(f"The average train accuracy is {cv_train_scores.mean():.2f}")
    #     print(f"The average test accuracy is {cv_test_scores.mean():.2f}")

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

    model = SVC(kernel='linear')
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("After scaling X data, we get", accuracy)



#     Hyperparameter tuning
    kernel_types = ["linear", "rbf", "poly"]
    length_of_kernels = len(kernel_types)
    accuracy_array_testing = np.empty(length_of_kernels)
    accuracy_array_training = np.empty(length_of_kernels)

    for i, kernels in enumerate(kernel_types):
        model = SVC(kernel=kernels)
        model.fit(X_train, Y_train)
        accuracy_array_training[i] = model.score(X_train, Y_train)
        accuracy_array_testing[i] = model.score(X_test, Y_test)

    plt.title("SVC with Varying Kernel Hyperparameter", fontsize=16)
    plt.plot(kernel_types, accuracy_array_testing, label="Test Accuracy", color='green')
    # plt.plot(kernel_types, accuracy_array_training, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Kernel Types", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.savefig("SVC with Varying Kernel Hyperparameter_FETAL.png")
    plt.close()
    # plt.show()

#     RBF gives the best accuracy
#     Now let us look at gamma
#     gammas = [0.1, 1, 10, 100]
#     length_of_gammas = len(gammas)
#     accuracy_array_testing = np.empty(length_of_gammas)
#     accuracy_array_training = np.empty(length_of_gammas)
#
#     for i, gamma in enumerate(gammas):
#         model = SVC(kernel="rbf", gamma=gamma)
#         model.fit(X_train, Y_train)
#         accuracy_array_training[i] = model.score(X_train, Y_train)
#         accuracy_array_testing[i] = model.score(X_test, Y_test)
#
#
#     plt.title("SVC with Varying Gamma Hyperparameter", fontsize=16)
#     plt.plot(gammas, accuracy_array_testing, label="Test Accuracy", color='green')
#     plt.plot(gammas, accuracy_array_training, label='Training accuracy')
#     plt.legend()
#     plt.grid(alpha=0.5)
#     plt.xlabel("Kernel Types", fontsize=14)
#     plt.ylabel("Accuracy", fontsize=14)
#     plt.savefig("SVC with Varying Gamma Hyperparameter_FETAL.png")
#     plt.close()
    # plt.show()
#     Gamma is best at 0

    model = SVC(kernel="rbf")
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("After hypertuning we get", accuracy)

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
def SVM_diabetes(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    train_size_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracy_train_error = np.empty(len(train_size_array))
    accuracy_test_error = np.empty(len(train_size_array))

    for i, training_size in enumerate(train_size_array):
        new_X_train = X_train.sample(frac=training_size, replace=False, random_state=0)
        new_Y_train = Y_train[new_X_train.index]

        model = SVC()
        model.fit(new_X_train, new_Y_train)
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        accuracy_train_error[i] = accuracy_score(Y_train, Y_pred_train) * 100
        accuracy_test_error[i] = accuracy_score(Y_test, Y_pred_test) * 100

    plt.title("SVC accuracy with Varying Training sizes", fontsize=16)
    plt.plot(train_size_array, accuracy_test_error, label="Test Accuracy", color='green')
    plt.plot(train_size_array, accuracy_train_error, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Training size", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(50, 100)
    plt.savefig("SVC accuracy with Varying Training sizes_DIABETES.png")
    plt.close()
    # plt.show()

    # model = SVC()
    # train_size, train_score, test_score = learning_curve(model, X, Y, train_sizes=train_size_array)
    # for train_size, cv_train_scores, cv_test_scores in zip(train_size, train_score, test_score):
    #     print(f"{train_size} samples were used to train the model")
    #     print(f"The average train accuracy is {cv_train_scores.mean():.2f}")
    #     print(f"The average test accuracy is {cv_test_scores.mean():.2f}")

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

    model = SVC()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("After scaling X data, we get", accuracy)

    #     Hyperparameter tuning
    kernel_types = ["linear", "rbf", "poly"]
    length_of_kernels = len(kernel_types)
    accuracy_array_testing = np.empty(length_of_kernels)
    accuracy_array_training = np.empty(length_of_kernels)

    for i, kernels in enumerate(kernel_types):
        model = SVC(kernel=kernels)
        model.fit(X_train, Y_train)
        accuracy_array_training[i] = model.score(X_train, Y_train)
        accuracy_array_testing[i] = model.score(X_test, Y_test)

    plt.title("SVC with Varying Kernel Hyperparameter", fontsize=16)
    plt.plot(kernel_types, accuracy_array_testing, label="Test Accuracy", color='green')
    # plt.plot(kernel_types, accuracy_array_training, label='Training accuracy')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel("Kernel Types", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.savefig("SVC with Varying Kernel Hyperparameter_DIABETES.png")
    plt.close()
    # plt.show()

    #     Linear gives the best accuracy
    #     Now let us look at gamma
    # gammas = [0.1, 1, 10, 100]
    # length_of_gammas = len(gammas)
    # accuracy_array_testing = np.empty(length_of_gammas)
    # accuracy_array_training = np.empty(length_of_gammas)
    #
    # for i, gamma in enumerate(gammas):
    #     model = SVC(kernel="linear", gamma=gamma)
    #     model.fit(X_train, Y_train)
    #     accuracy_array_training[i] = model.score(X_train, Y_train)
    #     accuracy_array_testing[i] = model.score(X_test, Y_test)
    #
    # plt.title("SVC with Varying Gamma Hyperparameter", fontsize=16)
    # plt.plot(gammas, accuracy_array_testing, label="Test Accuracy", color='green')
    # plt.plot(gammas, accuracy_array_training, label='Training accuracy')
    # plt.legend()
    # plt.grid(alpha=0.5)
    # plt.xlabel("Kernel Types", fontsize=14)
    # plt.ylabel("Accuracy", fontsize=14)
    # plt.show()

    model = SVC(kernel="linear")
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("After hypertuning we get", accuracy)

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


# EDA_fetal()
X_fetal, Y_fetal = fetal_preprocess()
KNN_fetal(X_fetal, Y_fetal)
# DT_fetal(X_fetal, Y_fetal)
# Boost_fetal(X_fetal, Y_fetal)
# NN_fetal(X_fetal, Y_fetal)
# SVM_fetal(X_fetal, Y_fetal)
# EDA_diabetes()
X_diabetes, Y_diabetes = diabetes_preprocess()
KNN_diabetes(X_diabetes, Y_diabetes)
# DT_diabetes(X_diabetes, Y_diabetes)
# Boost_diabetes(X_diabetes, Y_diabetes)
# NN_diabetes(X_diabetes, Y_diabetes)
# SVM_diabetes(X_diabetes, Y_diabetes)

