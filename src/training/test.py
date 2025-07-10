import torch
import torch_geometric
from src.model.gae import GAE
import pyvista as pv

def test(model: GAE,
         test_loader: torch_geometric.loader.DataLoader,
         device: torch.device,
         error_func: list[str],
         save_path: str = None,
         save_results: bool = False,
         mesh_path: str = "dataset\full\flow_Re_100000_alpha_1.vtu",
         verbose: bool = False):
    """
    Test the model on the test dataset.
    Returns:
        full_test_results: list of full test model predictions, from input params to output x (mapping + linear decoder + graph decoder)
        full_total_error: dict of full test model total errors per error function, from input params to output x
        full_total_error_list: dict of full test model all error types lists per parameter for mapping and prediction, from input params to output x
        ae_test_results: list of AE test model predictions, from input features to output x (graph encoder + linear autoencoder + graph decoder + mapping)
        ae_total_error: dict of AE test model total errors per error function, from input features to output x
        ae_total_error_list: dict of AE test model all error types lists per parameter, from input features to output x
    """
    model = model.to(device)
    model.eval()
    params_list = []
    full_pred_total_error = {f"total_{error.__name__}": {i: 0 for i in range(test_loader.dataset[0].y.shape[1])} for error in error_func}
    full_mapping_total_error = {f"total_{error.__name__}": {i: 0 for i in range(test_loader.dataset[0].y.shape[1])} for error in error_func}
    full_pred_error_list = {f"{error.__name__}": {i: [] for i in range(test_loader.dataset[0].y.shape[1])} for error in error_func}
    full_mapping_error_list = {f"{error.__name__}": {i: [] for i in range(test_loader.dataset[0].y.shape[1])} for error in error_func}
    full_test_results = []
    full_estimated_latent_var_list = []

    ae_total_error = {f"total_{error.__name__}": {i: 0 for i in range(test_loader.dataset[0].y.shape[1])} for error in error_func}
    ae_error_list = {f"{error.__name__}": {i: [] for i in range(test_loader.dataset[0].y.shape[1])} for error in error_func}
    ae_test_results = []
    ae_latent_var_list = []

    with torch.no_grad():
        for data in test_loader:
            params = data.params.float().to(device)
            data = data.to(device)

            # AE test model
            ae_pred, ae_latent_var, ae_estimated_latent_var = model(data, params)
            if model.linear_autoencoder is not None:
                ae_latent_var_list.append(ae_latent_var.cpu().numpy())
                ae_test_results.append(ae_pred.cpu().numpy())

            # Full test model
            if model.linear_autoencoder is not None:
                decoded_x = model.linear_autoencoder.decoder(ae_estimated_latent_var)
                decoder_input_shape = model.graph_decoder.config['convolution_layers']['hidden_channels'][0]
                full_pred = model.graph_decoder(data, decoded_x.reshape([data.x.shape[0], decoder_input_shape]), is_verbose=verbose)

                full_test_results.append(full_pred.cpu().numpy())
                full_estimated_latent_var_list.append(ae_estimated_latent_var.cpu().numpy())

            # Calculate error
            params_key = tuple(map(float, params.cpu().numpy()))
            params_list.append(params_key)

            for error in error_func:
                for i in range(ae_pred.shape[1]):
                    if model.linear_autoencoder is not None:
                        full_pred_total_error[f"total_{error.__name__}"][i] += error(pred=full_pred[:, i], target=data.y[:, i]).item()
                        full_mapping_total_error[f"total_{error.__name__}"][i] += error(pred=ae_estimated_latent_var, target=ae_latent_var).item()
                        full_pred_error_list[f"{error.__name__}"][i].append(error(pred=full_pred[:, i], target=data.y[:, i]).item())
                        full_mapping_error_list[f"{error.__name__}"][i].append(error(pred=ae_estimated_latent_var, target=ae_latent_var).item())
                    
                    ae_error_list[f"{error.__name__}"][i].append(error(pred=ae_pred[:, i], target=data.y[:, i]).item())
                    ae_total_error[f"total_{error.__name__}"][i] += error(pred=ae_pred[:, i], target=data.y[:, i]).item()
    if model.linear_autoencoder is not None:
        for error, value in full_pred_total_error.items():
            full_pred_total_error[error] = {i: value[i] / len(test_loader.dataset) for i in range(test_loader.dataset[0].y.shape[1])}
        for error, value in full_mapping_total_error.items():
            full_mapping_total_error[error] = {i: value[i] / len(test_loader.dataset) for i in range(test_loader.dataset[0].y.shape[1])}
    for error, value in ae_total_error.items():
        ae_total_error[error] = {i: value[i] / len(test_loader.dataset) for i in range(test_loader.dataset[0].y.shape[1])}

    return params_list, (full_test_results, full_pred_total_error, full_mapping_total_error, full_pred_error_list, full_mapping_error_list, full_estimated_latent_var_list), (ae_test_results, ae_total_error, ae_error_list, ae_latent_var_list)

def single_test(model: GAE,
                test_params: torch.Tensor,
                test_data: torch_geometric.data.Data,
                device: torch.device,
                save_path: str = None,
                save_results: bool = False):
    
    model = model.to(device)
    test_params = test_params.to(device)
    model.eval()
    with torch.no_grad():
        latent_var = model.mapping(test_params)
        decoded_x = model.linear_autoencoder.decoder(latent_var)
        decoder_input_shape = model.graph_decoder.config['convolution_layers']['hidden_channels'][0]
        pred_x = model.graph_decoder(test_data, decoded_x.reshape([test_data.x.shape[0], decoder_input_shape]))

    return pred_x