import csv
import os
import pandas as pd
def read_csv(path):
    df = pd.read_csv(path, header=0)
    return df

def add_data_csv(path, data):
    lines = read_csv(path)
    with open('test.csv','w')as f:
        f_csv = csv.writer(f)
        f_csv.writerows(lines)
        print(f"save file {path}")

def save_csv(path, content):
    with open(path,"r",encoding="utf-8") as f:
        f_csv = csv.writer(f)
        f_csv.writerows(content)
        print(f"save file {path}")

def exist_file(path):
    if os.path.exists(path):
        return True
    return False
