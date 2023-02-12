# model = DecisionTreeClassifier()
# train_size, train_score, test_score = learning_curve(model, X, Y, train_sizes=train_size_array)
#
# train_scores_mean = -train_score.mean(axis=1)
# test_scores_mean = -test_score.mean(axis=1)
#
# plt.plot(train_size, train_scores_mean, label='Training error')
# plt.plot(train_size, test_scores_mean, label='Test error')
# plt.ylabel('Accuracy', fontsize=14)
# plt.xlabel('Training set size', fontsize=14)
# plt.title('Learning curves for a Decision Tree model', fontsize=18, y=1.03)
# plt.legend()
# plt.show()
#
# for train_size, cv_train_scores, cv_test_scores in zip(train_size, train_score, test_score):
#     print(f"{train_size} samples were used to train the model")
#     print(f"The average train accuracy is {cv_train_scores.mean():.2f}")
#     print(f"The average test accuracy is {cv_test_scores.mean():.2f}")

#
# # After scaling, we get the same
# scaled_information = StandardScaler()
# scaled_information.fit(X_train)
# transformed_data_train = scaled_information.transform(X_train)
# transformed_data_test = scaled_information.transform(X_test)
# index_training = X_train.index
# columns_training = X_train.columns
# index_testing = X_test.index
# columns_testing = X_test.columns
# X_train = pd.DataFrame(transformed_data_train, index=index_training, columns=columns_training)
# X_test = pd.DataFrame(transformed_data_test, index=index_testing, columns=columns_testing)
#
# model = DecisionTreeClassifier()
# model.fit(X_train, Y_train)
# Y_pred = model.predict(X_test)
# accuracy = accuracy_score(Y_test, Y_pred) * 100
# print("After scaling X data, we get", accuracy)
#
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
