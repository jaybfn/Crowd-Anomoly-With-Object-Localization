# importing all the libraries
import os

# defining the paths 

BASE_PATH = '../data/ucf_action'
IMAGE_PATH = os.path.sep.join([BASE_PATH,'images'])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "Annotations"])

# define the path to the base output directory
BASE_OUTPUT = "../Model_Output"
# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, "lb.pickle"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 150
BATCH_SIZE = 16