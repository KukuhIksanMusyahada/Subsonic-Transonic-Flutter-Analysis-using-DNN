import os


from Essential import global_params as gp


def get_this_dir():
    return os.path.dirname( os.path.abspath(__file__) )


def get_data_source():
    return os.path.join(get_this_dir(), os.pardir, 'Data_Source')


def get_raw_data():
    return os.path.join(get_data_source(), 'Raw')


def get_processed_data():
    return os.path.join(get_data_source(), 'Processed')

def get_transonic_data():
    return os.path.join(get_processed_data(),'Transonic_Data')

def get_flutter_data():
    return os.path.join(get_processed_data(),'Flutter_Data')

def get_non_flutter_data():
    return os.path.join(get_processed_data(),'Non_Flutter_Data')

def get_flutter_class_data():
    return os.path.join(get_processed_data(),'Flutter_Classification_Data')


def get_result_data():
    return os.path.join(get_data_source(), 'Results')

def get_models_result():
    return os.path.join(get_result_data(),'Models_Result')


def get_models_history():
    return os.path.join(get_models_result(),'Models_History')


def get_models_prediction():
    return os.path.join(get_models_result(),'Models_Prediction')


def get_models_data():
    return os.path.join(get_data_source(), 'Models')


def get_models_classification():
    return os.path.join(get_models_data(), 'Model_Classification')


def get_models_non_flutter():
    return os.path.join(get_models_data(), 'Model_Non_Flutter')


def get_models_flutter():
    return os.path.join(get_models_data(), 'Model_Flutter')


def get_models_transonic():
    return os.path.join(get_models_data(), 'Model_Transonic')


def get_models_master():
    return os.path.join(get_models_data(), 'Model_Master')







def InitDataDirectories():
    important_dir = gp.DATA_DIRECTORIES

    for dir in important_dir:
        full_path = os.path.join (get_this_dir(), os.pardir, dir)
        full_path = os.path.abspath (full_path)

        if not os.path.exists(full_path):
            os.makedirs(full_path)