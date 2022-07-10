import os


from 


def get_this_dir():
    return os.path.dirname( os.path.abspath(__file__) )


def get_data_source():
    return os.path.join(get_this_dir(), 'Data_Source')


def get_raw_data():
    return os.path.join(get_data_source(), 'Raw')


def get_processed_data():
    return os.path.join(get_data_source(), 'Processed')


def get_result_data():
    return os.path.join(get_data_source(), 'Result')


def get_models_data():
    return os.path.join(get_data_source(), 'Models')







def InitDataDirectories():
    important_dir = None

    for dir in important_dir:
        full_path = os.path.join (GetThisDir(), os.pardir, os.pardir, dir)
        full_path = os.path.abspath (full_path)

        if not os.path.exists(full_path):
            os.makedirs(full_path)