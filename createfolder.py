import os

def create_dir(path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path, exist_ok = False)
            print("New directory is created")

if __name__ == '__main__':

    path ='data'
    create_dir(path)
    path ='Model_Output'
    create_dir(path)
    path ='scripts'
    create_dir(path)
    path ='notebooks'
    create_dir(path)
