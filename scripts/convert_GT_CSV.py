import pandas as pd
import glob
import os
from pathlib import Path

def txt_to_csv_GT(path, filename):

    """ This function converts .txt file to csv files in batch and also appends each files with the file name in the csv file for future mapping"""
    files_folder = path
    files = []
    files = [pd.read_csv(file, delimiter='\t', names =['startX','startY','endX','endY','class']) for file in glob.glob(os.path.join(files_folder ,"*.txt"))]
    files_df = pd.concat(files)
    gt_paths = [path.parts[-1:] for path in Path(path).rglob('*.txt')]
    df = pd.DataFrame(data=gt_paths, columns=['gt_img_id'])
    files_df = files_df.reset_index(drop=True)
    gt_df = pd.concat([df, files_df], axis=1)  
    gt_df.to_csv(filename+'.csv')
    return gt_df

if __name__=='__main__':

    GT_path = '../data/ucf_action/Kicking/001/gt/'
    df = txt_to_csv_GT(GT_path, 'Kicking')
