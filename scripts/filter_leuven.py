import pandas as pd

def convert_csv_to_dataframe(csv_file, classes):
    #read the csv file
    df = pd.read_csv(csv_file)
    
    #filter out based on classes
    df = df[df['class'].isin(classes)]
    
    # convert to 1s and 0s
    df = df.applymap(lambda x: 0 if x == 0.0 else 1 if isinstance(x, float) and x != 0 else x)

    #transpose rows as columns and vice versa
    df = df.transpose()
    
    #export the csv
    df.to_csv('updated_csv.csv')
    
    return df

def main():
    classes = ["airplane","car","sparrow","cat","deer","dog","frog","horse","boat","truck"]
    csv_file = '../data/leuven_full.csv'
    convert_csv_to_dataframe(csv_file, classes)

if __name__ == '__main__':
    main()