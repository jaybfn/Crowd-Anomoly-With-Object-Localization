import os

def create_dir(path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path, exist_ok = False)
            print("New directory is created")

if __name__ == '__main__':

    path ='Model_Output/plots'
    create_dir(path)
    