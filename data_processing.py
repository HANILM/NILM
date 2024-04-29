import pandas as pd
import numpy as np
from nilmtk import DataSet


def open_dataset(file_path, train_start, train_end, test_start, test_end, building):
    data = DataSet(file_path)
    data.set_window(start=train_start, end=train_end)
    train_elec = data.buildings[building].elec
    data.set_window(start=test_start, end=test_end)
    test_elec = data.buildings[building].elec
    return train_elec, test_elec


def get_power_series(elec, meter_keys, sample_period):

    meters= [elec[key] for key in meter_keys]

    return [key.power_series(sample_period=sample_period) for key in meters]

def get_power_value( meter_keys):
    return[ next(key)for key in meter_keys]


def pad_and_window(data, pad_width, window_size):
    padded_data = np.pad(data, (pad_width, pad_width), 'constant', constant_values=(0, 0))
    return np.array([padded_data[i:i + window_size] for i in range(len(padded_data) - window_size + 1)])

def data_window(data, window_size):

    return np.array([data[i:i + window_size] for i in range(len(data) - window_size + 1)])

def normalize(data, mean, std):
    return (data - mean) / std

def preprocess_series(series, window_size,mean, std, n_load, train=True):
    series_mean, series_std = mean, std
    pad_width = n_load // 2 #if train else window_size // 2
    windowed_data = pad_and_window(series, pad_width, window_size )
    normalized_data = normalize(windowed_data, series_mean, series_std)
    return pd.DataFrame(normalized_data), series_mean, series_std

def data_preprocess_series(series, window_size,mean, std):
    series_mean, series_std = mean, std

    windowed_data = data_window(series, window_size )
    normalized_data = normalize(windowed_data, series_mean, series_std)
    return pd.DataFrame(normalized_data), series_mean, series_std


def get_H5_power_values(file_path, meter_keys, aggregate_power_key,building):
    with pd.HDFStore(file_path, mode='r') as store:
        power_values = []
        for key in meter_keys:
            power_values.append(store.get(f'/building3/{key}').values.squeeze())
        main_values = store.get(f'/building3/{aggregate_power_key}').values.squeeze()

    return power_values, main_values

def split_H5_data_sequential(data, split_ratio=0.9):

    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

def closed_call_preprocessing(mains_lst, submeters_lsts,train_main_mean, train_main_std,train_meter_means,train_meter_stds ,  window_size, batch_size, train=True):
    sequence_reduction = window_size-35
    n_load = 35
    mains_df, mains_mean, mains_std = preprocess_series(mains_lst, window_size,train_main_mean, train_main_std, sequence_reduction, train=train)

    units_to_pad_load=n_load//2
    rec_mains = np.array(mains_lst)[units_to_pad_load:len(mains_lst) - units_to_pad_load]
    rec_mains = (rec_mains - mains_mean) / mains_std


    appliance_dfs = []
    for submeter_lst, meter_mean, meter_std in zip(submeters_lsts, train_meter_means, train_meter_stds):
        appliance_df, _, _ = data_preprocess_series(submeter_lst, n_load, meter_mean, meter_std)
        appliance_dfs.append(appliance_df)
    appliance_dfs.append(pd.DataFrame(rec_mains))
    return mains_df, appliance_dfs

def ddpg_call_preprocessing(mains_lst, submeter_lst, main_mean,main_std,  meter_mean,meter_std, window_size, batch_size, train=True):
    sequence_length = window_size
    n_load = 1
    mains_df, mains_mean, mains_std = preprocess_series(mains_lst, window_size, main_mean, main_std,sequence_length, train=train)
    appliance_df, appliance_mean, appliance_std = data_preprocess_series(submeter_lst, n_load, meter_mean,meter_std)

    return mains_df, appliance_df, appliance_mean, appliance_std






def train_test_split(processed_data, n_load=35, train=True):
    mains = processed_data.pop('main')
    appliances = {meter: data.reshape((-1, n_load if train else 1)) for meter, data in processed_data.items()}
    return mains, appliances


def load_data(train_start, train_end, test_start, test_end, building, meter_keys,
              window_size, batch_size, method,sample_period):
    train_elec, test_elec = open_dataset('D:/Git code/daima/zijixiede/ukdale.h5', train_start, train_end, test_start, test_end, building)
    train_series = get_power_series(train_elec, meter_keys, sample_period)
    test_series = get_power_series(test_elec, meter_keys, sample_period)
    train_meters_values=get_power_value(train_series)
    test_meters_values=get_power_value(test_series)

    train_elec_socker_main = train_elec.select_using_appliances(
        type=meter_keys).power_series(sample_period=sample_period)
    test_elec_socker_main = test_elec.select_using_appliances(
        type=meter_keys).power_series(sample_period=sample_period)



    train_main_values=next(train_elec_socker_main)
    test_main_values=next(test_elec_socker_main)
    train_main_values.fillna(0, inplace=True)
    test_main_values.fillna(0, inplace=True)
    ######load your own data
    # file_path = "D:/Git code/Datamerge/merge.h5"
    # meter_keys = ['kettle', 'microwave', 'washing dryer', 'television', 'stove']
    # aggregate_power_key = 'aggregate power'
    # train_meters_values, train_main_values = get_H5_power_values(file_path, meter_keys, aggregate_power_key,building)
    #
    # train_meters_values, train_main_values = get_H5_power_values(file_path, meter_keys, aggregate_power_key, building)
    #
    # # 将数据按顺序分割成训练集和测试集，比例为9:1
    # train_meters_values, test_meters_values= split_H5_data_sequential(train_meters_values)
    # train_main_values, test_main_values = split_H5_data_sequential(train_main_values)

    train_main_mean, train_main_std = train_main_values.mean(), train_main_values.std()
    train_meter_means = [series.mean() for series in train_meters_values]
    train_meter_stds = [series.std() for series in train_meters_values]


    train_mains_df, train_appliance_dfs = closed_call_preprocessing(train_main_values, train_meters_values,train_main_mean, train_main_std,train_meter_means,train_meter_stds , window_size, batch_size, train=True)
    test_mains_df, test_appliance_dfs = closed_call_preprocessing(test_main_values, test_meters_values,train_main_mean, train_main_std,train_meter_means,train_meter_stds , window_size, batch_size, train=True)
    train_ddpg_mains_df, train_ddpg_appliance_df, _, _ = ddpg_call_preprocessing(train_main_values, train_meters_values[0],train_main_mean, train_main_std, train_meter_means[0], train_meter_means[0],window_size,
                                                                     batch_size, train=True)

    test_ddpg_mains_df, test_ddpg_appliance_df, _, _ = ddpg_call_preprocessing(test_main_values, test_meters_values[0],
                                                                     train_main_mean, train_main_std,
                                                                     train_meter_means[0], train_meter_means[0],
                                                                     window_size,
                                                                     batch_size, train=True)




    if method=='closed':
        train_mains = train_mains_df.values.reshape((-1, 1, window_size))
        test_mains = test_mains_df.values.reshape((-1, 1, window_size))
        train_appliance_data = [
            df.values.reshape((-1, 35)) if i != len(train_appliance_dfs) - 1 else df.values.reshape((-1, 1)) for i, df
            in enumerate(train_appliance_dfs)]
        test_appliance_data = [
            df.values.reshape((-1, 35)) if i != len(test_appliance_dfs) - 1 else df.values.reshape((-1, 1)) for i, df
            in enumerate(test_appliance_dfs)]

        return train_mains, train_appliance_data, test_mains, test_appliance_data,train_meter_means, train_meter_stds

    else:
        train_ddpg_mains_df= train_ddpg_mains_df.values.reshape((-1, 1, window_size))
        test_ddpg_mains_df= test_ddpg_mains_df.values.reshape((-1, 1, window_size))

        train_ddpg_appliance_df= train_ddpg_appliance_df.values.reshape((-1, 1))
        test_ddpg_appliance_df= test_ddpg_appliance_df.values.reshape((-1, 1))

        return train_ddpg_mains_df, train_ddpg_appliance_df,test_ddpg_mains_df,  test_ddpg_appliance_df



