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
    
    model = model.to(device)
    model.eval()
    total_error = {f"total_{error.__name__}": 0 for error in error_func}
    test_results = []
    with torch.no_grad():
        index = 0
        for data in test_loader:
            params = data.params.float().to(device)
            data = data.to(device)
            latent_var = model.mapping(params)
            decoded_x = model.linear_autoencoder.decoder(latent_var)
            decoder_input_shape = model.graph_decoder.config['convolution_layers']['hidden_channels'][0]
            pred = model.graph_decoder(data, decoded_x.reshape([data.x.shape[0], decoder_input_shape]), is_verbose=verbose)
            test_results.append(pred)
            index += 1
            for error in error_func:
                total_error[f"total_{error.__name__}"] += error(pred, data.y).item()
                
    for error, value in total_error.items():
        total_error[error] = value / (index + 1)

    # save to vtu files
    if save_results:
        mesh = pv.read(mesh_path)
        mesh.clear_points()
        for i, pred in enumerate(test_results):
            pred.write_vtu(save_path + f"test_results_{i}.vtu")
        torch.save(test_results, save_path + "test_results.pt")

    return test_results, total_error

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

    if save_results:
        torch.save(test_data, save_path + f"test_data_Re{test_params[0]}_alpha{test_params[1]}.pt")

    return pred_x