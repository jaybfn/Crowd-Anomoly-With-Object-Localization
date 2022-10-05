# importing all the libraries
import os

# defining the paths 

BASE_PATH = '../data/ucf_action/Running/001'
BASE_PATH_IMG = '../data/ucf_action/Running'
IMAGE_PATH = os.path.sep.join([BASE_PATH_IMG,'001'])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "Running.csv"])

# define the path to the base output directory
BASE_OUTPUT = "Model_Output"
# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32