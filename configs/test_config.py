
import argparse
import numpy as np

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
MODEL = 'RefineNetNew'
DATA_DIRECTORY ='./data'
DATA_CITY_PATH = './dataset/cityscapes_list/clear_lindau.txt'
DATA_DIRECTORY_CITY = './data/Cityscapes'
DATA_LIST_PATH_EVAL = './data/Foggy_Zurich/lists_file_names/RGB_testv2_filenames.txt'
DATA_LIST_PATH_EVAL_FD ='./lists_file_names/leftImg8bit_testall_filenames.txt'
DATA_LIST_PATH_EVAL_ACDC = './lists_file_names/rgb_fog_val_filenames.txt'
DATA_DIR_EVAL = './data'
DATA_DIR_EVAL_FD = './data/Foggy_Driving'
NUM_CLASSES = 19 
RESTORE_FROM = 'no model'
SNAPSHOT_DIR = f'./data/snapshots/Base_model'
GT_DIR_FZ = './data/Foggy_Zurich'
GT_DIR_FD = './data/Foggy_Driving'
GT_DIR_CLINDAU = './data/Cityscapes/gtFine'
GT_DIR_ACDC = './data/ACDC'
SET = 'val'

MODEL = 'RefineNetNew'

def get_arguments():
    parser = argparse.ArgumentParser(description="Evlauation")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--data-city-list", type=str, default = DATA_CITY_PATH)
    parser.add_argument("--data-list-eval", type=str, default=DATA_LIST_PATH_EVAL)
    parser.add_argument("--data-list-eval-fd", type=str, default=DATA_LIST_PATH_EVAL_FD)  
    parser.add_argument("--data-list-eval-acdc", type=str, default=DATA_LIST_PATH_EVAL_ACDC)               
    parser.add_argument("--data-dir-city", type=str, default=DATA_DIRECTORY_CITY)
    parser.add_argument("--data-dir-eval", type=str, default=DATA_DIR_EVAL)
    parser.add_argument("--data-dir-eval-fd", type=str, default=DATA_DIR_EVAL_FD)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM)    
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--set", type=str, default=SET)
    parser.add_argument("--file-name", type=str, required=True)
    parser.add_argument("--gt-dir-fz", type=str, default=GT_DIR_FZ)
    parser.add_argument("--gt-dir-fd", type=str, default=GT_DIR_FD)
    parser.add_argument("--gt-dir-clindau", type=str, default=GT_DIR_CLINDAU)
    parser.add_argument("--gt-dir-acdc", type=str, default=GT_DIR_ACDC)
    parser.add_argument("--devkit-dir-fz", default='./data/Foggy_Zurich/lists_file_names') 
    parser.add_argument("--devkit-dir-fd", default='./lists_file_names') 
    parser.add_argument("--devkit-dir-clindau", default='./dataset/cityscapes_list')
    parser.add_argument("--devkit-dir-acdc", default='./lists_file_names') 
    return parser.parse_args()
