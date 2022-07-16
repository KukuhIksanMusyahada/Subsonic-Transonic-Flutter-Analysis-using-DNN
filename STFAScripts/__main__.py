import os
import pandas as pd
import matplotlib.pyplot as plt


from Essential import global_params as gp
from Essential import path_handler as ph
from Processing_Data import get_data as gd
from Processing_Data import data_processing as dp
from Training.models import set_low_tf_verbose
from Training.class_trainer_inferencer import *
from Training.flutter_trainer_inferencer import *
from Training.non_flutter_trainer_inferencer import *
from Training.transonic_trainer_inferencer import *


def runner():
    set_low_tf_verbose()
    # Init Directories
    ph.InitDataDirectories()
    # Input Model
    mach = float(input('please input mach number: '))
    vf = float(input('please input Flutter Speed: '))
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
    total_time = 0
    if len(os.listdir(ph.get_models_classification()))==0:
        model_class, history_class, time_class = classification_trainer(X_class_train, X_class_val, y_class_train, y_class_val)
        total_time+=time_class
    if len(os.listdir(ph.get_models_flutter()))==0:
        model_flutter, history_flutter, time_flutter = f_trainer(X_flutter_train, X_flutter_val, y_flutter_train, y_flutter_val, max_row)
        total_time+=time_flutter
    if len(os.listdir(ph.get_models_non_flutter()))==0:
        model_non_flutter, history_non_flutter, time_non_flutter = nf_trainer(X_non_flutter_train, X_non_flutter_val, y_non_flutter_train, y_non_flutter_val, max_row)
        total_time+=time_non_flutter
    if len(os.listdir(ph.get_models_transonic()))==0:
        model_transonic, history_transonic, time_transonic = transonic_trainer(X_transonic_train, X_transonic_val, y_transonic_train, y_transonic_val,max_row_transonic)
        total_time+=time_transonic
    
    print(f'TOTAL TIME FOR TRAINING = {total_time}')

    # Inference 
    class_pred = classification_inferencer(mach, vf)
    if class_pred == 0:
        output_str='Airfoil will not experience flutter'
    else:
        output_str='Airfoil will experience flutter'
    if mach < 0.9:
        ## Class Prediction
        ## Feature Prediction
        if class_pred == 0:
            feature_pred = nf_inferencer(mach, vf)
        else:
            feature_pred = f_inferencer(mach, vf)
    else:
        feature_pred = transonic_inferencer(mach, vf)
    print(f'Predicting done with the result /n')
    print(output_str)
    print()
    print(f'The properties (CL, CD, Plunge, Pitch) is at {feature_pred.shape}')
    test = pd.read_csv(os.path.join(ph.get_transonic_data(),'M_0.9_VF_1.0.csv'), nrows=90)
    test['CL'].plot()
    test['CD'].plot()
    test['plunge_airfoil'].plot()
    test['pitch_airfoil'].plot()
    plt.plot(feature_pred[:,0], label='CL_Pred')
    plt.plot(feature_pred[:,1], label='CD_Pred')
    plt.plot(feature_pred[:,2], label='Plunge_Pred')
    plt.plot(feature_pred[:,3], label='Pitch_Pred')
    plt.legend()
    plt.show()

    
if __name__=='__main__':
    runner()
    