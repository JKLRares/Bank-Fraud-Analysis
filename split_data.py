import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    in_f = 'all_data.csv'
    train_f = 'train.csv'
    test_f = 'test.csv'
    test_ratio = 0.25
    seed = 12345

    df = pd.read_csv(in_f)

    # Split into train/test
    train_df, test_df = train_test_split(
        df,
        test_size = test_ratio,
        random_state = seed,
        shuffle = True
    )

    # Save splits
    train_df.to_csv(train_f, index = False)
    test_df.to_csv(test_f, index = False)

    print(
        f"Data set split complete: train = {len(train_df)} rows into '{train_f}', test = {len(test_df)} rows into '{test_f}'"
    )
