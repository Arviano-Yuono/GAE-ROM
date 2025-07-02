from sklearn import preprocessing
import torch
import joblib

def scaler_functions(k):
    if k==1:
        sc_fun = preprocessing.MinMaxScaler()
        sc_name = "minmax"
    elif k==2:
        sc_fun = preprocessing.RobustScaler()
        sc_name = "robust"
    elif k==3:
        sc_fun = preprocessing.StandardScaler()
        sc_name = "standard"
    return sc_fun, sc_name


def tensor_scaling(tensor: torch.Tensor, scaling_type: int, scaler_name: str):
    if scaler_name == 'standard':
        scaling_fun_1 = preprocessing.StandardScaler()
        scaling_fun_2 = preprocessing.StandardScaler()
    elif scaler_name == 'minmax':
        scaling_fun_1 = preprocessing.MinMaxScaler()
        scaling_fun_2 = preprocessing.MinMaxScaler()
    elif scaler_name == 'robust':
        scaling_fun_1 = preprocessing.RobustScaler()
        scaling_fun_2 = preprocessing.RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {scaler_name}")
    
    if scaling_type==1:
        # print("SAMPLE SCALING")
        scale = scaling_fun_1.fit(tensor)
        scaled_data = torch.unsqueeze(torch.tensor(scale.transform(tensor)), 0).permute(2, 1, 0)
    elif scaling_type==2:
        # print("FEATURE SCALING")
        scale = scaling_fun_1.fit(torch.t(tensor))
        scaled_data = torch.unsqueeze(torch.tensor(scale.transform(torch.t(tensor))), 0).permute(1, 2, 0)
    elif scaling_type==3:
        # print("FEATURE-SAMPLE SCALING")
        scaler_f = scaling_fun_1.fit(torch.t(tensor))
        temp = torch.tensor(scaler_f.transform(torch.t(tensor)))
        scaler_s = scaling_fun_2.fit(temp)
        scaled_data = torch.unsqueeze(torch.tensor(scaler_s.transform(temp)), 0).permute(1, 2, 0)
        scale = [scaler_f, scaler_s]
    elif scaling_type==4:
        # print("SAMPLE-FEATURE SCALING")
        # For SAMPLE-FEATURE scaling with single feature:
        # Simply calculate mean and std of the entire tensor
        
        if tensor.shape[1] == 1:  # Single feature
            # Calculate mean and std of the entire tensor
            tensor_mean = torch.mean(tensor, dim=0)  # [1]
            tensor_std = torch.std(tensor, dim=0)    # [1]
            
            # Create scalers with the calculated parameters
            scaler_s = scaling_fun_1
            scaler_s.mean_ = tensor_mean.numpy().flatten()  # Shape (1,)
            scaler_s.scale_ = tensor_std.numpy().flatten()  # Shape (1,)
            scaler_s.n_features_in_ = 1
            
            scaler_f = scaling_fun_2
            scaler_f.mean_ = tensor_mean.numpy().flatten()  # Shape (1,)
            scaler_f.scale_ = tensor_std.numpy().flatten()  # Shape (1,)
            scaler_f.n_features_in_ = 1
            
            # Apply scaling
            temp_s = (tensor - scaler_s.mean_) / scaler_s.scale_
            temp_f = (temp_s.T - scaler_f.mean_.reshape(-1, 1)) / scaler_f.scale_.reshape(-1, 1)
            scaled_data = torch.unsqueeze(torch.t(temp_f), 0).permute(2, 1, 0)
            
        else:
            # Multiple features - use the original approach
            scaler_s = scaling_fun_1.fit(tensor)
            temp = torch.t(torch.tensor(scaler_s.transform(tensor)))
            scaler_f = scaling_fun_2.fit(temp)
            scaled_data = torch.unsqueeze(torch.t(torch.tensor(scaler_f.transform(temp))), 0).permute(2, 1, 0)
        
        scale = [scaler_s, scaler_f]
    return scale, scaled_data

def load_scaler(scaler_name: str):
    scaler = joblib.load(f'artifacts/{scaler_name}.joblib')
    return scaler

def inverse_scaling(tensor, scale, scaling_type):
    if scaling_type==1:
        # print("SAMPLE SCALING")
        rescaled_data = torch.tensor(scale.inverse_transform(torch.t(torch.tensor(tensor[:, :, 0].detach().numpy().squeeze()))))
    elif scaling_type==2:
        # print("FEATURE SCALING")
        rescaled_data = torch.tensor(torch.t(torch.tensor(scale.inverse_transform(tensor[:, :, 0].detach().numpy().squeeze()))))
    elif scaling_type==3:
        # print("FEATURE-SAMPLE SCALING")
        scaler_f = scale[0]
        scaler_s = scale[1]
        rescaled_data = torch.t(torch.tensor(scaler_f.inverse_transform(torch.tensor(scaler_s.inverse_transform(tensor[:, :, 0].detach().numpy().squeeze())))))
    elif scaling_type==4:
        # print("SAMPLE-FEATURE SCALING")
        scaler_s = scale[0]
        scaler_f = scale[1]
        rescaled_data = torch.tensor(scaler_s.inverse_transform(torch.t(torch.tensor(scaler_f.inverse_transform(tensor[:, :, 0].detach().numpy().squeeze())))))
    return rescaled_data 
