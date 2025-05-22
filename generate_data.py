import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Parameters
n_rows = 1000 
output_csv = 'all_data.csv'

# Seed for reproducibility
np.random.seed(12345)

# Time range
start_date = datetime.now() - timedelta(days = 365)

def random_date_time(start, end):
	delta = end - start
	n_seconds = int(delta.total_seconds())
	random_second = np.random.randint(0, n_seconds)
	time = start + timedelta(seconds = random_second)
	return time.replace(microsecond = 0)

cities = ['Slatina', 'Bucharest', 'New York', 'Los Angeles', 'Chicago', 'Craiova', 'Cluj', 'Miami', 'London', 'Berlin', 'Paris', 'Tokyo', 'Sydney', 'Constanta']
devices = ['phone', 'computer', 'laptop']
transaction_types = ['online', 'in-person']

# Generate columns
transaction_id = np.arange(1, n_rows + 1)
user_id = np.random.randint(1, 250, size = n_rows)
amount = np.round(np.random.uniform(1, 3000, size = n_rows), 2)
date_time = [random_date_time(start_date, datetime.now()) for _ in range(n_rows)]
location = np.random.choice(cities, size = n_rows)
ip_add = ['.'.join(str(np.random.randint(0, 255)) for _ in range(4)) for _ in range(n_rows)]
device = np.random.choice(devices, size = n_rows)
transaction_type = np.random.choice(transaction_types, size = n_rows)
freq = np.random.poisson(lam = 2, size = n_rows)

# Bias higher amounts slightly more likely to be fraud
fraud_probs = np.where(amount > 2000, 0.1, 0.02)
labels = np.random.binomial(1, fraud_probs)

# Create Dataframe
df = pd.DataFrame({
	'transaction_id': transaction_id,
	'user_id': user_id,
	'amount': amount,
	'date_time': date_time,
	'location': location,
	'ip_address': ip_add,
	'device': device,
	'transaction_type': transaction_type,
	'frequency_last_24h': freq,
	'fraud': labels,
})

# Save to CSV
df.to_csv(output_csv, index = False)
print(f"Generated {n_rows} transactions and saved to '{output_csv}'")
