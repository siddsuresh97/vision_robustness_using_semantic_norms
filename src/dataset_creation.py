#import all the required libraries
import os
import shutil
from tqdm import tqdm
import camelot
import pandas as pd


def load_tables_from_pdf(pdf_path, pages):
    tables = camelot.read_pdf(pdf_path, pages=pages, flavor='stream')
    return tables

def transform_tables_to_df(tables):
    df = pd.DataFrame()
    for table in tables:
        df = df.append(table.df)
        
    #ignore the first two rowns and use the third row as column names
    df.columns = df.iloc[1]
    df = df.iloc[2:]
    df = df.reset_index(drop=True) #reset index
    return df

# taje in a text file and return a df
def read_leuven_norms_as_df(path):
    if 'ANIMALS' not in path:
        #read txt file with utf-8 encoding
        df = pd.read_csv(path, sep='\t', encoding='ISO-8859-1"')
    else:
        df = pd.read_csv(path, sep='\t')
    return df


# write a function to create a dataset from the ecoset dataset
# the function takes in the path to the ecoset dataset and the path to the new dataset storage location
# the function also takes in a list of concepts to be included in the dataset

def create_dataset(dataset_dir, new_dataset_dir, concepts, make_dataset = True):
    if make_dataset:
        # create a new directory to store the dataset
        if not os.path.exists(new_dataset_dir):
            os.makedirs(new_dataset_dir)
        
        # create a dir called 'train', 'val' and 'test' in the new_dataset_dir
        for split in ['train', 'val', 'test']:
            if not os.path.exists(os.path.join(new_dataset_dir, split)):
                os.makedirs(os.path.join(new_dataset_dir, split))
            # for each split copy the images from concepts from dataset_dir to the new_dataset_dir
            for concept in tqdm(concepts):
                if not os.path.exists(os.path.join(new_dataset_dir, split, concept)):
                    os.makedirs(os.path.join(new_dataset_dir, split, concept))
                for class_name in os.listdir(os.path.join(dataset_dir, split)):
                    if concept in class_name:
                        if not os.path.exists(os.path.join(new_dataset_dir, split, concept)):
                            os.makedirs(os.path.join(new_dataset_dir, split, concept))
                        for image in os.listdir(os.path.join(dataset_dir, split, class_name)):
                            shutil.copy(os.path.join(dataset_dir, split, class_name, image), os.path.join(new_dataset_dir, split, concept, image))
                    