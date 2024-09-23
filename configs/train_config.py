import argparse
import numpy as np

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
BATCH_SIZE = 4
NUM_WORKERS = 4
DATA_DIRECTORY_CW = './data/Cityscapes'
DATA_LIST_PATH_CW = './dataset/cityscapes_list/train_origin.txt'
DATA_LIST_RF = './lists_file_names/RGB_sum_filenames.txt'
DATA_DIR = './data'
NUM_CLASSES = 19 
NUM_STEPS = 100000 
NUM_STEPS_STOP = 60000  # early stopping
RANDOM_SEED = np.random.randint(0, 99999)
RESTORE_FROM = 'no_model'
SAVE_PRED_EVERY = 100
ENERGY_VALUE = -15.0
LOSS_WEIGHT = 1.0
ALPHA = 0.999
SET = 'train'

def get_arguments():

    parser = argparse.ArgumentParser(description="Baseline framework")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--data-dir-cw", type=str, default=DATA_DIRECTORY_CW)
    parser.add_argument("--data-list-cw", type=str, default=DATA_LIST_PATH_CW)
    parser.add_argument("--data-dir-rf", type=str, default=DATA_DIR)
    parser.add_argument("--data-list-rf", type=str, default=DATA_LIST_RF)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM)
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--set", type=str, default=SET)
    parser.add_argument("--file-name", type=str, required=True)
    parser.add_argument("--energy-threshold", type=float, default=ENERGY_VALUE)
    parser.add_argument("--pseudo-weight", type=float, default=LOSS_WEIGHT)
    parser.add_argument("--alpha", type=float, default=ALPHA)    
    return parser.parse_args()

args = get_arguments()


