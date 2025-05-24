import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Ensure the directory for plots exists
plots_dir = 'plots'
if not os.path.exists(plots_dir):
	os.makedirs(plots_dir)

datasets = ['train.csv', 'test.csv']

for file in datasets:
	# Drop any rows with missing values
	df = pd.read_csv(file)
	df = df.dropna()

	# Descriptive statistics
	output_path = file.replace('.csv', '_descriptive.txt')
	with open(output_path, 'w') as out:
		out.write(f"Descriptive Analysis for {file}\n\n")

		# Numeric descriptive statistics
		numeric_cols = df.select_dtypes(include=[np.number]).columns
		out.write("Numeric features summary:\n")
		out.write(df[numeric_cols].describe().to_string())
		out.write("\n")

		# Categorical descriptive statistics
		categorical_cols = df.select_dtypes(include=['object', 'category']).columns
		out.write("Categorical features summary:\n")
		out.write(df[categorical_cols].describe().to_string())
		out.write("\n")

		out.write(f"End of analysis for {file}\n")

	# Generate histograms for numeric columns
	for col in numeric_cols:
		plt.figure()
		df[col].hist(bins=30)
		plt.title(f'Histogram of {col} - {file}')
		plt.xlabel(col)
		plt.ylabel('Frequency')
		plt.tight_layout()
		plt.savefig(os.path.join(plots_dir, f"{file.replace('.csv', '')}_{col}_hist.png"))
		plt.close()

	# Generate bar plots for categorical columns
	for col in categorical_cols:
		plt.figure()
		df[col].value_counts().plot(kind='bar')
		plt.title(f'Countplot of {col} - {file}')
		plt.xlabel(col)
		plt.ylabel('Count')
		plt.tight_layout()
		plt.savefig(os.path.join(plots_dir, f"{file.replace('.csv', '')}_{col}_bar.png"))
		plt.close()
		
	# Generate boxplots for numeric columns (to spot outliers)
	for col in numeric_cols:
		plt.figure()
		df.boxplot(column=col)
		plt.title(f'Boxplot of {col} - {file}')
		plt.ylabel(col)
		plt.tight_layout()
		plt.savefig(os.path.join(plots_dir, f"{file.replace('.csv', '')}_{col}_boxplot.png"))
		plt.close()
	
	# Generate correlation heatmap for numeric columns
	corr = df[numeric_cols].corr()
	plt.figure()
	plt.imshow(corr, interpolation='none', aspect='auto')
	plt.colorbar()
	plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
	plt.yticks(range(len(numeric_cols)), numeric_cols)
	plt.title(f'Correlation heatmap - {file}')
	plt.tight_layout()
	plt.savefig(os.path.join(plots_dir, f"{file.replace('.csv', '')}_correlation_heatmap.png"))
	plt.close()

	# Violin plots for numeric features by fraud class
	if 'fraud' in df.columns:
		for col in numeric_cols:
			plt.figure()
			data0 = df[df['fraud'] == 0][col]
			data1 = df[df['fraud'] == 1][col]
			plt.violinplot([data0, data1], showmeans=True)
			plt.xticks([1, 2], ['Normal', 'Fraud'])
			plt.title(f'Violin plot of {col} by Fraud - {file}')
			plt.ylabel(col)
			plt.tight_layout()
			plt.savefig(os.path.join(plots_dir, f"{file.replace('.csv', '')}_{col}_violin_vs_fraud.png"))
			plt.close()


train_df = pd.read_csv('train.csv').dropna()
test_df  = pd.read_csv('test.csv').dropna()

# Separate features/target
X_train = train_df.drop(columns=['transaction_id', 'fraud'])
y_train = train_df['fraud']
X_test  = test_df.drop(columns=['transaction_id', 'fraud'])
y_test  = test_df['fraud']

# One-hot encode categorical variables consistent with labs
X_train = pd.get_dummies(X_train, drop_first=True)
X_test  = pd.get_dummies(X_test, drop_first=True)
# Align columns of test to train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Initialize and fit random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
classes = ['Normal', 'Fraud']
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted')
plt.ylabel('True')
for i in range(cm.shape[0]):
	for j in range(cm.shape[1]):
		plt.text(j, i, cm[i, j], ha='center', va='center')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
plt.close()

# Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X_train.columns
plt.figure(figsize=(8,6))
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'feature_importances.png'))
plt.close()
