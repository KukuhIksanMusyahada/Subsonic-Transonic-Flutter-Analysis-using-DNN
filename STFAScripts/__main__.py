
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
    X_flutter_train, X_flutter_val, y_flutter_train, y_flutter_val = dp.process_data_flutter()
    ## Process non flutter  data
    X_non_flutter_train, X_non_flutter_val, y_non_flutter_train, y_non_flutter_val = dp.process_data_non_flutter()
    ## Process transonic  data
    X_transonic_train, X_transonic_val, y_transonic_train, y_transonic_val = dp.process_data_transonic()
    # Train Models 

    # Inference 


if __name__=='__main__':
    runner()
    