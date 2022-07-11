
from Essential import global_params as gp
from Essential import path_handler as ph
from Processing_Data import get_data as gd

def runner():
    # Init Directories
    ph.InitDataDirectories()
    # Get The Data
    gd.scan()
    # Process the data


    # Train Models 

    # Inference 


if __name__=='__main__':
    runner()
    