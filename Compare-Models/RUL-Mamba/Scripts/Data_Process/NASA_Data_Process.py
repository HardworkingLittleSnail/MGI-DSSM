import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.io
import torch
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NASA_DATA_DIR = PROJECT_ROOT / 'Data' / 'NASA_Data'
NASA_CACHE_PATH = NASA_DATA_DIR / 'NASA.npy'
NASA_REAL_DATA_TEMPLATE = 'Results/Capacity_{test_name}_Real_Data.pth'


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data/NASA_Data/', help='path of the data file')
    parser.add_argument('--battery_cache_path', type=str, default='Data/NASA_Data/NASA.npy', help='path of the cached battery data file')
    parser.add_argument('--Battery_list', nargs='+', default=['B0005', 'B0006', 'B0007', 'B0018'], help='Battery data')
    parser.add_argument('--Rated_Capacity', type=float, default=2.0, help='Rate Capacity')
    parser.add_argument('--test_name', type=str, default='B0005', help='Battery data used for test')
    parser.add_argument('--start_point_list', nargs='+', type=int, default=[50, 70, 90], help='The cycle when prediction gets started.')
    return parser


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_runtime_args(config=None):
    if config is None:
        return SimpleNamespace(
            data_dir='Data/NASA_Data/',
            battery_cache_path='Data/NASA_Data/NASA.npy',
            Battery_list=['B0005', 'B0006', 'B0007', 'B0018'],
            Train_Battery_list=['B0006', 'B0007', 'B0018'],
            Rated_Capacity=2.0,
            test_name='B0005',
            start_point_list=[50, 70, 90],
            seq_len=16,
        )
    dataset = config['dataset']
    window = config.get('window', {})
    return SimpleNamespace(
        data_dir=dataset.get('data_dir', 'Data/NASA_Data/'),
        battery_cache_path=dataset.get('battery_cache_path', 'Data/NASA_Data/NASA.npy'),
        Battery_list=dataset.get('battery_list', ['B0005', 'B0006', 'B0007', 'B0018']),
        Train_Battery_list=dataset.get('train_battery_list', ['B0006', 'B0007', 'B0018']),
        Rated_Capacity=dataset['rated_capacity'],
        test_name=dataset['test_name'],
        start_point_list=dataset['start_points'],
        seq_len=window.get('seq_len', 16),
    )


def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = Path(matfile).stem
    col = data[filename]
    col = col[0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0]
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(convert_to_time(col[i][2][0])), d2
        data.append(d1)

    return data


def getBatteryCapacityData(Battery, name):
    elem_list = []
    i = 1
    for Bat in Battery:
        elem = []
        if Bat['type'] == 'discharge':
            elem.append(name)
            elem.append(i)
            elem.append(Bat['data']['Capacity'][0])
            i += 1
            elem_list.append(elem)
    return elem_list


def DataRead(Battery_list, dir_path):
    battery_data = []
    resolved_dir = _resolve_project_path(dir_path)
    for name in Battery_list:
        print('Load Dataset ' + name + '.mat ...')
        path = resolved_dir / f'{name}.mat'
        data = loadMat(str(path))
        battery_data += getBatteryCapacityData(data, name)
    return battery_data


def BatteryDataRead(args):
    battery_data = DataRead(args.Battery_list, args.data_dir)
    cache_path = _resolve_project_path(args.battery_cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    battery_data_array = np.array([battery_data], dtype=object)
    np.save(cache_path, battery_data_array, allow_pickle=True)
    return battery_data


def LoadBatteryCache(cache_path):
    resolved_cache_path = _resolve_project_path(cache_path)
    with open(resolved_cache_path, 'rb') as cache_file:
        if cache_file.read(42).startswith(b'version https://git-lfs.github.com/spec'):
            raise RuntimeError(
                f'{resolved_cache_path} is a Git LFS pointer, not NASA data. '
                'Run `git lfs pull` or regenerate it from the four .mat files.'
            )
    battery_data = np.load(resolved_cache_path, allow_pickle=True)
    if isinstance(battery_data, np.ndarray):
        if battery_data.ndim == 3 and battery_data.shape[0] == 1:
            return battery_data[0].tolist()
        if battery_data.shape == ():
            return battery_data.item()
        if battery_data.size == 1:
            first_item = battery_data.reshape(-1)[0]
            return first_item.item() if isinstance(first_item, np.ndarray) and first_item.shape == () else first_item
        return battery_data.tolist()
    return battery_data


def _clean_capacity_data(BatteryData):
    df = pd.DataFrame(BatteryData, columns=['BatteryName', 'Cycle', 'Capacity'])
    df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
    df.sort_values(['BatteryName', 'Cycle'], inplace=True)

    # Paper Sec. 2.6: linearly interpolate missing values, then remove values
    # outside the 2-sigma interval.  Treat removed outlier values as missing and
    # interpolate them so that the cycle index remains continuous.
    df['Capacity'] = df.groupby('BatteryName', sort=False)['Capacity'].transform(
        lambda values: values.interpolate(method='linear', limit_direction='both')
    )
    grouped = df.groupby('BatteryName', sort=False)['Capacity']
    means = grouped.transform('mean')
    stds = grouped.transform(lambda values: values.std(ddof=0))
    outlier_mask = stds.gt(0) & df['Capacity'].sub(means).abs().gt(2.0 * stds)
    df.loc[outlier_mask, 'Capacity'] = np.nan
    df['Capacity'] = df.groupby('BatteryName', sort=False)['Capacity'].transform(
        lambda values: values.interpolate(method='linear', limit_direction='both')
    )
    if df['Capacity'].isna().any():
        raise ValueError('NASA capacity data still contain missing values after interpolation.')
    return df


def _prepare_frames(df, test_name, train_battery_list, rated_capacity):
    if test_name in train_battery_list:
        raise ValueError(f'Test battery {test_name} must not occur in the training battery list.')
    available = set(df['BatteryName'])
    missing_train_batteries = set(train_battery_list) - available
    if missing_train_batteries:
        raise ValueError(f'Missing NASA training batteries: {sorted(missing_train_batteries)}')

    df['Capacity'] /= rated_capacity
    df['constant'] = 0
    df['target'] = df['Capacity']
    columns = ['BatteryName', 'constant', 'Cycle', 'Capacity', 'target']
    df_all = df.loc[df['BatteryName'] == test_name, columns].copy()
    df_train = df.loc[df['BatteryName'].isin(train_battery_list), columns].copy()

    # Fit Min-Max parameters on training batteries only, then apply them to the
    # held-out battery.  This implements Sec. 2.6 without test-set leakage.
    min_val = df_train['Capacity'].min()
    max_val = df_train['Capacity'].max()
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
        raise ValueError('Cannot Min-Max normalize NASA capacities from the training data.')
    df_train['Capacity'] = (df_train['Capacity'] - min_val) / (max_val - min_val)
    df_all['Capacity'] = (df_all['Capacity'] - min_val) / (max_val - min_val)

    # Separate group identifiers prevent windows from crossing cell boundaries.
    df_train['time_idx'] = df_train.groupby('BatteryName', sort=False).cumcount()
    df_all['time_idx'] = df_all.groupby('BatteryName', sort=False).cumcount()
    df_train.reset_index(drop=True, inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    df_train.attrs['capacity_min'] = float(min_val)
    df_train.attrs['capacity_max'] = float(max_val)
    df_all.attrs.update(df_train.attrs)

    return df_train, df_all


def _prepare_author_public_frames(BatteryData, test_name, start_point, rated_capacity):
    """Reproduce the data flow currently published in the author repository.

    This path is intentionally isolated from the paper-faithful default.  It is
    used only for protocol forensics: the public implementation adds the
    observed test-cell prefix to supervised training and assigns one continuous
    time index to all training cells.
    """
    df = pd.DataFrame(BatteryData, columns=['BatteryName', 'Cycle', 'Capacity'])
    df['Capacity'] = pd.to_numeric(df['Capacity'], errors='raise') / rated_capacity
    df['constant'] = 0
    df['target'] = df['Capacity']
    columns = ['BatteryName', 'constant', 'Cycle', 'Capacity', 'target']
    df_all = df.loc[df['BatteryName'] == test_name, columns].copy()
    df_train = df.loc[
        (df['BatteryName'] != test_name)
        | ((df['BatteryName'] == test_name) & (df['Cycle'] < start_point)),
        columns,
    ].copy()

    min_val = df_train['Capacity'].min()
    max_val = df_train['Capacity'].max()
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
        raise ValueError('Cannot Min-Max normalize NASA capacities from the public protocol.')
    df_train['Capacity'] = (df_train['Capacity'] - min_val) / (max_val - min_val)
    df_all['Capacity'] = (df_all['Capacity'] - min_val) / (max_val - min_val)

    df_train['time_idx'] = np.arange(len(df_train), dtype=np.int64)
    df_all['time_idx'] = np.arange(len(df_all), dtype=np.int64)
    df_train.reset_index(drop=True, inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    df_train.attrs['capacity_min'] = float(min_val)
    df_train.attrs['capacity_max'] = float(max_val)
    df_all.attrs.update(df_train.attrs)
    return df_train, df_all


def DataProcess(BatteryData, test_name, start_point, rated_capacity):
    df = _clean_capacity_data(BatteryData)
    train_battery_list = [name for name in df['BatteryName'].unique() if name != test_name]
    df_train, df_all = _prepare_frames(df, test_name, train_battery_list, rated_capacity)
    df_test = df_all[['BatteryName', 'constant', 'Cycle', 'Capacity', 'target', 'time_idx']].copy()
    return df_train, df_test


def BatteryDataProcess(BatteryData, test_name, start_point, args):
    data_protocol = getattr(args, 'Data_Protocol', 'paper')
    if data_protocol == 'author_public':
        df_train, df_all = _prepare_author_public_frames(
            BatteryData,
            test_name,
            start_point,
            args.Rated_Capacity,
        )
    elif data_protocol == 'paper':
        df = _clean_capacity_data(BatteryData)
        train_battery_list = getattr(
            args,
            'Train_Battery_list',
            [name for name in df['BatteryName'].unique() if name != test_name],
        )
        df_train, df_all = _prepare_frames(
            df,
            test_name,
            train_battery_list,
            args.Rated_Capacity,
        )
    else:
        raise ValueError(f'Unsupported NASA data protocol: {data_protocol}')
    history_start = start_point - args.seq_len
    df_test = df_all.loc[
        df_all['Cycle'] >= history_start,
        ['BatteryName', 'time_idx', 'constant', 'Cycle', 'Capacity', 'target'],
    ].copy()
    df_test.attrs.update(df_all.attrs)
    return df_train, df_test, df_all


def main():
    args = build_parser().parse_args()
    battery_data = BatteryDataRead(args)
    runtime_args = SimpleNamespace(Rated_Capacity=args.Rated_Capacity, seq_len=16)
    _, _, df_all = BatteryDataProcess(battery_data, args.test_name, args.start_point_list[0], runtime_args)
    real_data = df_all['target'].values * args.Rated_Capacity
    results_path = _resolve_project_path(NASA_REAL_DATA_TEMPLATE.format(test_name=args.test_name))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(real_data, results_path)


if __name__ == '__main__':
    main()
