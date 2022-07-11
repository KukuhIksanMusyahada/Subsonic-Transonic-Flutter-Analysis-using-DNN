
from Essential import global_params as gp
from Essential import path_handler as ph
from Processing_Data import get_data as gd
from Processing_Data import data_processing as dp

def runner():
    # Init Directories
    ph.InitDataDirectories()
    # Get The Data
    gd.scan()
    # Process the data
    ## Process flutter classification data
    X_class_train, X_class_val, y_class_train, y_class_val = dp.process_data_classification()
    ## Process flutter  data

    ## Process non flutter  data

    ## Process transonic  data

    # Train Models 

    # Inference 


if __name__=='__main__':
    runner()
    