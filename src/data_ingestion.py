import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# Ensure the logs directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

consoloe_handler = logging.StreamHandler()
consoloe_handler.setLevel('DEBUG')

file_handler = logging.FileHandler(os.path.join(log_dir, 'data_ingestion.log'))
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoloe_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(consoloe_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param data_url: URL or path to the CSV file.
    :return: DataFrame containing the loaded data.
    """
    try:
        logger.debug(f"Loading data from {data_url}")
        data = pd.read_csv(data_url)
        logger.debug("Data loaded successfully")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and encoding categorical variables.
    
    :param data: DataFrame containing the raw data.
    :return: Preprocessed DataFrame.
    """
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug("Data preprocessing completed")
        return df
    except KeyError as e:
        logger.error(f"Error during preprocessing: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save the train and test datasets to the specified directory.
    
    :param train_data: DataFrame containing the training data.
    :param test_data: DataFrame containing the testing data.
    :param output_dir: Directory where the datasets will be saved.
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug(f"Train and test datasets saved successfully {raw_data_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.2
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error(f"Failed to complete the data ingestion process: {e}")
        raise

if __name__ == "__main__":
    main()
