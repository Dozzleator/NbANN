import pandas as pd
from pathlib import Path
import os

def main() -> None:
    data_path = Path(__file__).parent.parent / 'data' / 'job_salary.csv'
    data_df = pd.read_csv(data_path)
    data = data_df.to_numpy()

    print(data)

    return None

if __name__ == '__main__':
    main()
