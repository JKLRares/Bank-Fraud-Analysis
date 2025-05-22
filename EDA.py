import pandas as pd
import numpy as np

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
		if len(numeric_cols) > 0:
			out.write("Numeric features summary:\n")
			out.write(df[numeric_cols].describe().to_string())
			out.write("\n\n")

		# Categorical descriptive statistics
		categorical_cols = df.select_dtypes(include=['object', 'category']).columns
		if len(categorical_cols) > 0:
			out.write("Categorical features summary:\n")
			out.write(df[categorical_cols].describe().to_string())
			out.write("\n\n")

		out.write(f"End of analysis for {file}\n")
