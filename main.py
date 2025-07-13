from data_cleaning import process_dataset
from inverted_index import create_inverse_index_catalogue
import os

if __name__ == '__main__':
    if not os.path.isfile("cleaned_data.csv"):
        process_dataset()

    create_inverse_index_catalogue()

