import ast
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.integrate import cumulative_trapezoid
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
import random
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class BatteryDataset(Dataset):
    def __init__(self, data, context=10, horizon=50, capacity_length=10,
                 scaler_features=None, rated_capacity=3500):
        min_discharge_capacity = data['Discharge_Capacity'].min()
        if min_discharge_capacity > 2850:
            raise ValueError(f"min_discharge_capacity{min_discharge_capacity}")
        self.data = data
        self.context = context
        self.horizon = horizon
        self.capacity_length = capacity_length
        self.rated_capacity = float(rated_capacity)
        self.capacity_increment_list = []
        self.rul_values = []
        threshold = 0.8 * self.rated_capacity
        total_cycles = len(data)
        end_of_life_idx = None
        for i in range(len(data)):
            if data.iloc[i]['Discharge_Capacity'] < threshold:
                end_of_life_idx = i
                break
        if end_of_life_idx is None:
            end_of_life_idx = total_cycles - 1
        for i in range(len(data)):
            current_rul = max(0, end_of_life_idx - i)  
            self.rul_values.append(current_rul)

        for i in range(len(data)):
            cap_incr = data.iloc[i]['Capacity_Increment']
            if isinstance(cap_incr, str):
                cap_incr = ast.literal_eval(cap_incr)
            # Section 3.1: capacity increments are divided by rated capacity.
            self.capacity_increment_list.append(
                np.asarray(cap_incr, dtype=np.float32) / self.rated_capacity
            )
            
        self.capacity_increment_list = np.array(self.capacity_increment_list) 
        self.rul_values = np.array(self.rul_values)  
    def __len__(self):
        return len(self.data) - self.context - self.horizon + 1

    def __getitem__(self, idx):
        history_increments = self.capacity_increment_list[idx:idx+self.context]  
        increment_tensor = torch.tensor(history_increments, dtype=torch.float32)
        inputs = increment_tensor
        his_capacities = self.data.iloc[
            idx:idx+self.context
        ]['Discharge_Capacity'].values / self.rated_capacity
        future_capacities = self.data.iloc[
            idx+self.context : idx+self.context+self.horizon
        ]['Discharge_Capacity'].values / self.rated_capacity
        current_cycle = idx + self.context - 1 
        current_rul = self.rul_values[current_cycle]  
        future_capacities = torch.tensor(future_capacities, dtype=torch.float32)
        his_capacities = torch.tensor(his_capacities, dtype=torch.float32)
        current_rul = torch.tensor(current_rul, dtype=torch.float32)
        outputs = (future_capacities, his_capacities, current_rul)
        return inputs, outputs
    
class BatteryDataset1(BatteryDataset):
    """Dataset for the 2500 mAh NCM+NCA cells described in Section 2."""

    def __init__(self, data, context=10, horizon=50, capacity_length=10,
                 scaler_features=None):
        min_discharge_capacity = data['Discharge_Capacity'].min()
        if min_discharge_capacity > 2050:
            raise ValueError(f"min_discharge_capacity{min_discharge_capacity}")
        super().__init__(
            data=data,
            context=context,
            horizon=horizon,
            capacity_length=capacity_length,
            scaler_features=scaler_features,
            rated_capacity=2500,
        )


class NASABatteryDataset(Dataset):
    """IC2ML windows built from one NASA battery's charge/discharge cycles."""

    rated_capacity = 2.0
    eol_capacity = 1.4

    def __init__(self, capacity_increments, capacities, context=10, horizon=50):
        self.capacity_increments = np.asarray(capacity_increments, dtype=np.float32)
        self.capacities = np.asarray(capacities, dtype=np.float32)
        self.context = context
        self.horizon = horizon
        if self.capacity_increments.ndim != 2 or self.capacity_increments.shape[1] != 10:
            raise ValueError("NASA capacity increments must have shape [cycles, 10]")
        if len(self.capacity_increments) != len(self.capacities):
            raise ValueError("NASA inputs and capacities must contain the same cycles")
        if len(self.capacities) < context + horizon:
            raise ValueError(
                f"NASA battery has {len(self.capacities)} usable cycles, but "
                f"context + horizon is {context + horizon}"
            )

        eol_indices = np.flatnonzero(self.capacities <= self.eol_capacity)
        self.eol_index = int(eol_indices[0]) if len(eol_indices) else len(self.capacities) - 1

    def __len__(self):
        return len(self.capacities) - self.context - self.horizon + 1

    def __getitem__(self, idx):
        history_end = idx + self.context
        future_end = history_end + self.horizon
        inputs = torch.tensor(
            self.capacity_increments[idx:history_end] / self.rated_capacity,
            dtype=torch.float32,
        )
        history = torch.tensor(
            self.capacities[idx:history_end] / self.rated_capacity,
            dtype=torch.float32,
        )
        future = torch.tensor(
            self.capacities[history_end:future_end] / self.rated_capacity,
            dtype=torch.float32,
        )
        current_index = history_end - 1
        rul = torch.tensor(
            max(0, self.eol_index - current_index), dtype=torch.float32
        )
        return inputs, (future, history, rul)


def _nasa_charge_increment(charge_data, voltage_range=(3.6, 3.7), num_points=10):
    """Integrate charge current and interpolate Q(V) on the paper's 0.1 V segment."""

    voltage = np.asarray(charge_data.Voltage_measured, dtype=np.float64).reshape(-1)
    current = np.asarray(charge_data.Current_measured, dtype=np.float64).reshape(-1)
    time = np.asarray(charge_data.Time, dtype=np.float64).reshape(-1)
    valid = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(time)
    voltage, current, time = voltage[valid], current[valid], time[valid]
    if len(voltage) < 2:
        return None

    # The NASA records contain a few connection transients. The CC charging
    # samples are positive and near 1.5 A; 0.5 A excludes those transients and CV.
    charging = current > 0.5
    voltage, current, time = voltage[charging], current[charging], time[charging]
    if len(voltage) < 2 or voltage.min() > voltage_range[0] or voltage.max() < voltage_range[1]:
        return None

    order = np.argsort(time)
    voltage, current, time = voltage[order], current[order], time[order]
    increasing_time = np.r_[True, np.diff(time) > 0]
    voltage, current, time = (
        voltage[increasing_time],
        current[increasing_time],
        time[increasing_time],
    )
    if len(time) < 2:
        return None

    charge_capacity = cumulative_trapezoid(current, time, initial=0.0) / 3600.0
    voltage_order = np.argsort(voltage)
    voltage, charge_capacity = voltage[voltage_order], charge_capacity[voltage_order]
    voltage, unique_indices = np.unique(voltage, return_index=True)
    charge_capacity = charge_capacity[unique_indices]
    if voltage[0] > voltage_range[0] or voltage[-1] < voltage_range[1]:
        return None

    sample_voltage = np.linspace(voltage_range[0], voltage_range[1], num_points)
    sampled_capacity = np.interp(sample_voltage, voltage, charge_capacity)
    return sampled_capacity - sampled_capacity[0]


def load_nasa_cycle_data(mat_path, voltage_range=(3.6, 3.7)):
    """Return one 10-point charge feature for every NASA discharge cycle.

    Missing charge segments are causally forward-filled within the same cell so
    cycle numbering remains identical to the MGI-DSSM summary protocol.
    """
    battery_name = os.path.splitext(os.path.basename(mat_path))[0]
    battery = loadmat(mat_path, squeeze_me=True, struct_as_record=False)[battery_name]
    latest_increment = None
    increments, capacities, cycles = [], [], []
    for operation in np.atleast_1d(battery.cycle):
        if operation.type == 'charge':
            candidate = _nasa_charge_increment(
                operation.data, voltage_range=voltage_range
            )
            if candidate is not None:
                latest_increment = candidate
        elif operation.type == 'discharge':
            capacity = float(operation.data.Capacity)
            if np.isfinite(capacity):
                increments.append(
                    None if latest_increment is None else latest_increment.copy()
                )
                capacities.append(capacity)
                cycles.append(len(cycles) + 1)
            latest_increment = None

    last_valid = None
    for index, increment in enumerate(increments):
        if increment is not None:
            last_valid = increment
        elif last_valid is not None:
            increments[index] = last_valid.copy()
    if all(value is None for value in increments):
        raise ValueError(f"No valid NASA charge segments in {mat_path}")
    for index, increment in enumerate(increments):
        if increment is None:
            # A future cycle must never be used to construct an earlier input.
            increments[index] = np.zeros(10, dtype=np.float64)
        else:
            break
    return (
        np.asarray(increments, dtype=np.float32),
        np.asarray(capacities, dtype=np.float32),
        np.asarray(cycles, dtype=np.int64),
    )


def _load_nasa_battery(mat_path, context, horizon, voltage_range=(3.6, 3.7)):
    increments, capacities, _ = load_nasa_cycle_data(
        mat_path, voltage_range=voltage_range
    )
    return NASABatteryDataset(increments, capacities, context=context, horizon=horizon)


def NASA_trainloader(args):
    """Cell-level split: two train cells, one validation cell, one test cell."""

    battery_names = ['B0005', 'B0006', 'B0007', 'B0018']
    test_battery = args.condition
    if test_battery not in battery_names:
        raise ValueError(
            f"NASA condition must be one of {battery_names}; got {test_battery!r}"
        )
    remaining = [name for name in battery_names if name != test_battery]
    preferred_validation = 'B0007' if test_battery != 'B0007' else 'B0006'
    validation_battery = preferred_validation
    train_batteries = [name for name in remaining if name != validation_battery]

    root = os.path.join('data', 'NASA data')
    voltage_range = (args.nasa_voltage_start, args.nasa_voltage_end)
    if not np.isclose(voltage_range[1] - voltage_range[0], 0.1):
        raise ValueError("NASA voltage interval must be exactly 0.1 V")
    datasets = {
        name: _load_nasa_battery(
            os.path.join(root, f'{name}.mat'),
            args.context,
            args.horizon,
            voltage_range=voltage_range,
        )
        for name in battery_names
    }
    train_samples = [
        datasets[name][index]
        for name in train_batteries
        for index in range(len(datasets[name]))
    ]
    validation_samples = [
        datasets[validation_battery][index]
        for index in range(len(datasets[validation_battery]))
    ]
    test_samples = [
        datasets[test_battery][index]
        for index in range(len(datasets[test_battery]))
    ]
    print(
        f'NASA {voltage_range[0]:.1f}-{voltage_range[1]:.1f} V | '
        f'train={train_batteries} ({len(train_samples)} windows), '
        f'validation={validation_battery} ({len(validation_samples)} windows), '
        f'test={test_battery} ({len(test_samples)} windows)'
    )
    return (
        DataLoader(train_samples, batch_size=args.batch_size, shuffle=True),
        DataLoader(validation_samples, batch_size=args.batch_size, shuffle=False),
        DataLoader(test_samples, batch_size=args.batch_size, shuffle=False),
    )


def NCA_trainloader(args):
    train_samples = []
    train_features_list = []
    train_outputs = []
    val_samples = []
    val_outputs = []
    test_samples = []
    test_outputs = []
    if args.condition == 'CY45-05_1':
        train_files = [
            'CY45-05_1-#1.csv', 'CY45-05_1-#2.csv', 'CY45-05_1-#3.csv', 'CY45-05_1-#4.csv',
            'CY45-05_1-#5.csv', 'CY45-05_1-#6.csv', 'CY45-05_1-#7.csv', 'CY45-05_1-#8.csv',
            'CY45-05_1-#9.csv', 'CY45-05_1-#10.csv', 'CY45-05_1-#11.csv', 'CY45-05_1-#12.csv',
            'CY45-05_1-#13.csv', 'CY45-05_1-#14.csv', 'CY45-05_1-#15.csv', 'CY45-05_1-#16.csv',
            'CY45-05_1-#17.csv'
        ]
        val_files = [
            'CY45-05_1-#28.csv', 'CY45-05_1-#25.csv'
        ]
        test_files = [
            'CY45-05_1-#24.csv', 'CY45-05_1-#26.csv', 'CY45-05_1-#27.csv', 'CY45-05_1-#22.csv',
            'CY45-05_1-#23.csv'
        ]
    elif args.condition == 'CY25-05_1':
        train_files = [
            'CY25-05_1-#2.csv', 'CY25-05_1-#3.csv', 'CY25-05_1-#4.csv',
            'CY25-05_1-#5.csv', 'CY25-05_1-#6.csv', 'CY25-05_1-#7.csv', 'CY25-05_1-#8.csv',
            'CY25-05_1-#9.csv', 'CY25-05_1-#10.csv', 'CY25-05_1-#11.csv', 'CY25-05_1-#13.csv'
        ]
        val_files = [
            'CY25-05_1-#18.csv', 'CY25-05_1-#19.csv'
        ]
        test_files = [
             'CY25-05_1-#1.csv', 'CY25-05_1-#14.csv', 'CY25-05_1-#15.csv', 'CY25-05_1-#16.csv',
            'CY25-05_1-#17.csv', 'CY25-05_1-#12.csv'
        ]
    elif args.condition == 'CY25-025_1':
        train_files = [
            'CY25-025_1-#1.csv', 'CY25-025_1-#2.csv', 'CY25-025_1-#3.csv'
        ]
        val_files = [
            'CY25-025_1-#7.csv'
        ]
        test_files = [
            'CY25-025_1-#5.csv', 'CY25-025_1-#6.csv', 'CY25-025_1-#4.csv'
        ]
    elif args.condition == 'CY25-1_1':
        train_files = [
            'CY25-1_1-#1.csv', 'CY25-1_1-#2.csv', 'CY25-1_1-#3.csv', 'CY25-1_1-#4.csv', 'CY25-1_1-#5.csv'
        ]
        val_files = [
            'CY25-1_1-#6.csv'
        ]
        test_files = [
            'CY25-1_1-#7.csv', 'CY25-1_1-#8.csv', 'CY25-1_1-#9.csv'
        ]
    elif args.condition == 'CY35-05_1':
        train_files = [
            'CY35-05_1-#1.csv'
        ]
        val_files = [
            'CY35-05_1-#2.csv'
        ]
        test_files = [
            'CY35-05_1-#3.csv'
        ]
    else:
        raise ValueError(f"Unsupported condition: {args.condition}")
    

    
    input_folders = [
        'dataset/NCA/V3.6-3.7/',
    ]

    if hasattr(args, 'dataaccess'):
        if args.dataaccess == 100:
            train_files = train_files.copy()
        else:
            num_train = max(1, math.ceil(len(train_files) * args.dataaccess / 100))
            train_files = random.sample(train_files, num_train)
    else:
        train_files = train_files.copy()

    for input_folder in input_folders:
        for file_name in train_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue  
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(ast.literal_eval)
            try:
                battery_dataset = BatteryDataset(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                train_samples.append(inputs)

                train_outputs.append(outputs)
    for input_folder in input_folders:  
        for file_name in val_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(ast.literal_eval)
            try:
                battery_dataset = BatteryDataset(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                val_samples.append(inputs)
                val_outputs.append(outputs)

    for input_folder in input_folders:
        for file_name in test_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(ast.literal_eval)
            try:
                battery_dataset = BatteryDataset(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                test_samples.append(inputs)
                test_outputs.append(outputs)

    train_loader = DataLoader(
        list(zip(train_samples, train_outputs)),
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        list(zip(val_samples, val_outputs)),
        batch_size=args.batch_size, 
        shuffle=False  
    )
    test_loader = DataLoader(
        list(zip(test_samples, test_outputs)),
        batch_size=args.batch_size, 
        shuffle=False
    )

    return train_loader, val_loader, test_loader

def NCM_trainloader(args):
    train_samples = []
    train_features_list = []
    train_outputs = []
    val_samples = []
    val_outputs = []
    test_samples = []
    test_outputs = []

    if args.condition == 'CY45-05_1':
        train_files = [
            'CY45-05_1-#1.csv', 'CY45-05_1-#2.csv', 'CY45-05_1-#3.csv', 'CY45-05_1-#4.csv',
            'CY45-05_1-#5.csv', 'CY45-05_1-#6.csv', 'CY45-05_1-#7.csv', 'CY45-05_1-#8.csv',
            'CY45-05_1-#9.csv', 'CY45-05_1-#10.csv', 'CY45-05_1-#11.csv', 'CY45-05_1-#12.csv',
            'CY45-05_1-#13.csv', 'CY45-05_1-#14.csv', 'CY45-05_1-#15.csv', 'CY45-05_1-#16.csv'
        ]
        val_files = [
            'CY45-05_1-#28.csv','CY45-05_1-#17.csv'
        ]
        test_files = [
            'CY45-05_1-#24.csv', 'CY45-05_1-#26.csv', 'CY45-05_1-#27.csv', 'CY45-05_1-#22.csv',
            'CY45-05_1-#23.csv'
        ]
    elif args.condition == 'CY25-05_1':
        train_files = [
            'CY25-05_1-#1.csv', 'CY25-05_1-#2.csv', 'CY25-05_1-#3.csv', 'CY25-05_1-#4.csv',
            'CY25-05_1-#6.csv', 'CY25-05_1-#7.csv', 'CY25-05_1-#8.csv',
            'CY25-05_1-#9.csv', 'CY25-05_1-#10.csv', 'CY25-05_1-#11.csv', 'CY25-05_1-#12.csv',
            'CY25-05_1-#13.csv', 'CY25-05_1-#15.csv', 'CY25-05_1-#16.csv'
        ]
        val_files = [
            'CY25-05_1-#5.csv','CY25-05_1-#17.csv'
        ]
        test_files = [
            'CY25-05_1-#18.csv', 'CY25-05_1-#19.csv', 'CY25-05_1-#20.csv', 'CY25-05_1-#21.csv',
            'CY25-05_1-#22.csv', 'CY25-05_1-#23.csv'
        ]
    elif args.condition == 'CY35-05_1':
        train_files = [
            'CY35-05_1-#1.csv'
        ]
        val_files = [
            'CY35-05_1-#2.csv'
        ]
        test_files = [
            'CY35-05_1-#3.csv', 'CY35-05_1-#4.csv'
        ]
    else:
        raise ValueError(f"Unsupported condition: {args.condition}")
    input_folders = [
        'dataset/NCM/V3.6-3.7/',
    ]

    if hasattr(args, 'dataaccess'):
        if args.dataaccess == 100:
            train_files = train_files.copy()
        else:
            num_train = max(1, math.ceil(len(train_files) * args.dataaccess / 100))
            train_files = random.sample(train_files, num_train)
    else:
        train_files = train_files.copy()

    for input_folder in input_folders:
        for file_name in train_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue  
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(ast.literal_eval)
            try:
                battery_dataset = BatteryDataset(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                train_samples.append(inputs)

                train_outputs.append(outputs)

    for input_folder in input_folders:  
        for file_name in val_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(ast.literal_eval)
            try:
                battery_dataset = BatteryDataset(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                val_samples.append(inputs)
                val_outputs.append(outputs)

    for input_folder in input_folders:
        for file_name in test_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(ast.literal_eval)
            try:
                battery_dataset = BatteryDataset(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                test_samples.append(inputs)
                test_outputs.append(outputs)

    train_loader = DataLoader(
        list(zip(train_samples, train_outputs)),
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        list(zip(val_samples, val_outputs)),
        batch_size=args.batch_size, 
        shuffle=False  
    )
    test_loader = DataLoader(
        list(zip(test_samples, test_outputs)),
        batch_size=args.batch_size, 
        shuffle=False
    )

    return train_loader, val_loader, test_loader

def NCMNCA_trainloader(args):
    train_samples = []
    train_features_list = []
    train_outputs = []
    val_samples = []
    val_outputs = []
    test_samples = []
    test_outputs = []
    if args.condition == 'CY25-05_1':
        train_files = [
            'CY25-05_1-#1.csv'
        ]
        val_files = [
            'CY25-05_1-#2.csv'
        ]
        test_files = [
            'CY25-05_1-#3.csv'
        ]

    elif args.condition == 'CY25-05_2':
        train_files = [
            'CY25-05_2-#1.csv'
        ]
        val_files = [
            'CY25-05_2-#2.csv'
        ]
        test_files = [
            'CY25-05_2-#3.csv'
        ]

    elif args.condition == 'CY25-05_4':
        train_files = [
            'CY25-05_4-#1.csv'
        ]
        val_files = [
            'CY25-05_4-#2.csv'
        ]
        test_files = [
            'CY25-05_4-#3.csv'
        ]

    else:
        raise ValueError(f"Unsupported condition: {args.condition}")
    input_folders = [
        'dataset/NCMNCA/V3.6-3.7/',
    ]

    if hasattr(args, 'dataaccess'):
        if args.dataaccess == 100:
            train_files = train_files.copy()
        else:
            num_train = max(1, math.ceil(len(train_files) * args.dataaccess / 100))
            train_files = random.sample(train_files, num_train)
    else:
        train_files = train_files.copy()

    for input_folder in input_folders:
        for file_name in train_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue 
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(ast.literal_eval)
            try:
                battery_dataset = BatteryDataset1(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                train_samples.append(inputs)

                train_outputs.append(outputs)

    for input_folder in input_folders: 
        for file_name in val_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(ast.literal_eval)
            try:
                battery_dataset = BatteryDataset1(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                val_samples.append(inputs)
                val_outputs.append(outputs)

    for input_folder in input_folders:
        for file_name in test_files:
            file_path = os.path.join(input_folder, file_name)
            if not os.path.exists(file_path):
                continue
                
            data = pd.read_csv(file_path)
            data['Capacity_Increment'] = data['Capacity_Increment'].apply(ast.literal_eval)
            try:
                battery_dataset = BatteryDataset1(data, context=args.context, horizon=args.horizon)
            except ValueError as e:
                print(f" {file_path}：{e}")
                continue
            
            for idx in range(len(battery_dataset)):
                inputs, outputs = battery_dataset[idx]
                test_samples.append(inputs)
                test_outputs.append(outputs)
    train_loader = DataLoader(
        list(zip(train_samples, train_outputs)),
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        list(zip(val_samples, val_outputs)),
        batch_size=args.batch_size, 
        shuffle=False  
    )
    test_loader = DataLoader(
        list(zip(test_samples, test_outputs)),
        batch_size=args.batch_size, 
        shuffle=False
    )

    return train_loader, val_loader, test_loader
