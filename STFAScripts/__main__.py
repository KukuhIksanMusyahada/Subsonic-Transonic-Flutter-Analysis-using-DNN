from Essential import global_params as gp
from Essential import path_handler as ph
from Processing_Data import get_data as gd
from Processing_Data import data_processing as dp
from Training.models import set_low_tf_verbose
from class_trainer_inferencer import classification_trainer
from flutter_trainer_inferencer import f_trainer
from non_flutter_trainer_inferencer import nf_trainer
from transonic_trainer_inferencer import transonic_trainer

def runner():
    set_low_tf_verbose()
    # Init Directories
    ph.InitDataDirectories()
    # Get The Data
    max_row, max_row_transonic = gd.scan()
    # Process the data
    ## Process flutter classification data
    X_class_train, X_class_val, y_class_train, y_class_val = dp.process_data_classification()
    ## Process flutter  data
    X_flutter_train, X_flutter_val, y_flutter_train, y_flutter_val = dp.process_data_flutter(max_row)
    ## Process non flutter  data
    X_non_flutter_train, X_non_flutter_val, y_non_flutter_train, y_non_flutter_val = dp.process_data_non_flutter(max_row)
    ## Process transonic  data
    X_transonic_train, X_transonic_val, y_transonic_train, y_transonic_val = dp.process_data_transonic(max_row_transonic)
    # Train Models 
    model_class, history_class, time_class = classification_trainer(X_class_train, X_class_val, y_class_train, y_class_val)
    model_flutter, history_flutter, time_flutter = f_trainer(X_flutter_train, X_flutter_val, y_flutter_train, y_flutter_val, max_row)
    model_non_flutter, history_non_flutter, time_non_flutter = nf_trainer(X_non_flutter_train, X_non_flutter_val, y_non_flutter_train, y_non_flutter_val, max_row)
    model_transonic, history_transonic, time_transonic = transonic_trainer(X_transonic_train, X_transonic_val, y_transonic_train, y_transonic_val,max_row_transonic)
    total_time = time_class + time_flutter + time_non_flutter + time_transonic
    print(f'TOTAL TIME FOR TRAINING = {total_time}')
    # Inference 


if __name__=='__main__':
    runner()
    