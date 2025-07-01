import torch
import torch_geometric
from src.model.gae import GAE

def test(model: GAE,
         test_params: torch.Tensor,
         test_loader: torch_geometric.loader.DataLoader,
         device: torch.device,
         save_path: str = None,
         save_results: bool = False):
    
    model = model.to(device)
    test_params = test_params.to(device)
    model.eval()
    test_results = []
    with torch.no_grad():
        index = 0
        for data in test_loader:
            params = test_params[index]
            data = data.to(device)
            latent_var = model.mapping(params)
            decoded_x = model.linear_autoencoder.decoder(latent_var)
            data.x = model.graph_decoder(data, decoded_x)
            test_results.append(data)
            index += 1

    if save_results:
        torch.save(test_results, save_path + "test_results.pt")

    return test_results

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
        pred_x = model.graph_decoder(test_data, decoded_x)

    if save_results:
        torch.save(test_data, save_path + f"test_data_Re{test_params[0]}_alpha{test_params[1]}.pt")

    return pred_x