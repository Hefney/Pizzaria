import torch


def save_model_state(model, path):
    """
    Saves only the model's parameters

    Args:
        model: PyTorch model
        path: Path where to save the model
    """
    import os

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        # Save model to CPU first to avoid potential GPU memory issues
        model_state = model.cpu().state_dict() if hasattr(model, 'cpu') else model.state_dict()
        torch.save(model_state, path)
    except Exception as e:
        raise Exception(f"Error saving model to {path}: {str(e)}")


def load_model_state(model, path, device=None):
    """
    Loads the model's parameters into a pre-defined architecture

    Args:
        model: PyTorch model
        path: Path to the saved model state
        device: Device to load the model to (default: None, will use CUDA if available)
    """
    try:
        # Determine device if not specified
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load state dict
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)

        # Move model to device and set to eval mode
        model = model.to(device)
        model.eval()

        return model
    except Exception as e:
        raise Exception(f"Error loading model from {path}: {str(e)}")
