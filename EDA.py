import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
