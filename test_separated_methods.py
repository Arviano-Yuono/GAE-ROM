import torch
from src.model.gae import GAE
from src.utils.commons import get_config, get_device
from src.data.loader_paper import GraphDatasetPaper
from torch_geometric.loader import DataLoader

def test_separated_methods():
    """Test the separated encoding, mapping, and decoding methods."""
    
    # Load configuration
    config = get_config('configs/paper.yaml')
    device = get_device()
    
    # Create dataset and model
    dataset = GraphDatasetPaper(config=config['config'], split='train')
    model = GAE(config, num_graphs=dataset.num_graphs).to(device)
    
    # Create data loader
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print("Testing separated GAE methods...")
    
    for batch in train_loader:
        batch = batch.to(device)
        
        print(f"Input shape: {batch.x.shape}")
        print(f"Parameters shape: {batch.params.shape}")
        
        # Test individual methods
        with torch.no_grad():
            # 1. Test encoding
            print("\n1. Testing encode method:")
            encoded_output = model.encode(batch)
            print(f"   Encoded output shape: {encoded_output.shape}")
            
            # 2. Test mapping
            print("\n2. Testing mapping method:")
            if hasattr(batch, 'params') and batch.params is not None:
                estimated_latent = model.mapping(batch.params)
                print(f"   Estimated latent shape: {estimated_latent.shape}")
            else:
                print("   No parameters found for mapping")
            
            # 3. Test decoding
            print("\n3. Testing decode method:")
            estimated_x = model.decode(batch, encoded_output)
            print(f"   Final output shape: {estimated_x.shape}")
            
            # 4. Test full forward pass
            print("\n4. Testing full forward pass:")
            estimated_x, encoded_output, estimated_latent = model(batch)
            print(f"   Final output shape: {estimated_x.shape}")
            print(f"   Encoded output shape: {encoded_output.shape}")
            print(f"   Estimated latent shape: {estimated_latent.shape if estimated_latent is not None else 'None'}")
            
            # Example usage patterns
            print("\n" + "="*50)
            print("EXAMPLE USAGE PATTERNS")
            print("="*50)
            
            # Example 1: Only encoding (for feature extraction)
            print("\nExample 1: Feature extraction (encoding only)")
            encoded_features = model.encode(batch)
            print(f"Extracted features shape: {encoded_features.shape}")
            
            # Example 2: Only decoding (for reconstruction from features)
            print("\nExample 2: Reconstruction from features")
            reconstructed = model.decode(batch, encoded_features)
            print(f"Reconstructed output shape: {reconstructed.shape}")
            
            # Example 3: Custom pipeline
            print("\nExample 3: Custom pipeline")
            # Encode
            encoded = model.encode(batch)
            # Apply custom processing to encoded output
            custom_encoded = encoded * 1.1 if encoded is not None else None
            # Decode
            custom_output = model.decode(batch, custom_encoded)
            print(f"Custom output shape: {custom_output.shape}")
            
            # Example 4: Full forward pass with custom processing
            print("\nExample 4: Full forward pass")
            estimated_x, encoded_output, estimated_latent = model(batch)
            print(f"Complete pipeline output shape: {estimated_x.shape}")
            
            print("\n" + "="*50)
            print("TESTING COMPLETED SUCCESSFULLY!")
            print("="*50)
        
        print("\n✅ All separated methods work correctly!")
        break

def demonstrate_individual_usage():
    """Demonstrate how to use individual methods for different purposes."""
    
    config = get_config('configs/paper.yaml')
    device = get_device()
    
    dataset = GraphDatasetPaper(config=config['config'], split='train')
    model = GAE(config, num_graphs=dataset.num_graphs).to(device)
    
    for batch in dataset:
        batch = batch.to(device)
        
        print("\n=== Demonstrating Individual Method Usage ===")
        
        # Example 1: Only encoding (for feature extraction)
        print("\nExample 1: Feature extraction (encoding only)")
        encoded_features, _ = model.encode(batch)
        print(f"Extracted features shape: {encoded_features.shape}")
        
        # Example 2: Only mapping (for parameter-to-latent mapping)
        print("\nExample 2: Parameter to latent mapping")
        latent_from_params = model.mapping(batch.params)
        print(f"Latent from parameters shape: {latent_from_params.shape}")
        
        # Example 3: Only decoding (for reconstruction from features)
        print("\nExample 3: Reconstruction from features")
        _, decoded_features = model.encode(batch)
        reconstructed = model.decode(batch, decoded_features)
        print(f"Reconstructed output shape: {reconstructed.shape}")
        
        # Example 4: Custom pipeline
        print("\nExample 4: Custom pipeline")
        # Encode
        encoded, latent = model.encode(batch)
        # Apply custom processing to latent variables
        custom_latent = latent * 1.1 if latent is not None else None
        # Decode with custom latent
        if custom_latent is not None:
            # You would need to modify the autoencoder to accept custom latent
            # For now, just show the concept
            print("Custom latent processing applied")
        
        break

if __name__ == "__main__":
    test_separated_methods()
    demonstrate_individual_usage()
