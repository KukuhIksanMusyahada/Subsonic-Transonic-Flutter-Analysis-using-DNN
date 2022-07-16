from Essential import path_handler as ph
from Training.trainer_inferencer import *

# Columns
COLUMNS1 = ['CD', 'CL', 'plunge(airfoil)', 'pitch(airfoil)']
COLUMNS2 = ['CD', 'CL', 'plunge_airfoil','pitch_airfoil']


# Training and Inferencing
TRAIN_RATIO = 0.9
CASE = ["ModelFlutterClassification", "ModelFlutterPrediction",
            "ModelNonFlutterPrediction", "ModelTransonicPrediction",
            "MasterModel"]



# Directories
DATA_DIRECTORIES = [
        "Data_Source",
        "Data_Source/Models",
        "Data_Source/Models/Model_Classification",
        "Data_Source/Models/Model_Non_Flutter",
        "Data_Source/Models/Model_Flutter",
        "Data_Source/Models/Model_Transonic",
        # "Data_Source/Models/Model_Master",
        "Data_Source/Raw",
        "Data_Source/Results",
        "Data_Source/Processed",
        "Data_Source/Processed/Transonic_Data",
        "Data_Source/Processed/Flutter_Data",
        "Data_Source/Processed/Non_Flutter_Data",
        "Data_Source/Processed/Flutter_Classification_Data"
        
    ]