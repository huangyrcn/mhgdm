import sys
import pathlib
import yaml
import ml_collections
import torch
import numpy as np
import networkx as nx # Keep networkx if needed for direct graph inspection later

# Ensure the project root is in the Python path
project_root = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(project_root))

from utils.data_utils import MyDataset # Import MyDataset for data loading
from sampler import Sampler           # Import the Sampler class

def load_config(config_path):
    """Loads configuration from a YAML file."""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        config = ml_collections.ConfigDict(config_dict)
    print("Configuration loaded successfully.")
    return config

def configure_for_sampling(config):
    """Adds or modifies configuration specific to the Sampler."""
    print("Configuring for sampling...")
    # --- Sampler Specific Configuration ---
    # NOTE: Adjust the checkpoint path as needed.
    config.ckpt = "checkpoints/ENZYMES/proto_guide_score_model/proto_guide_score_model_20250425_045147.pth"
    config.exp_name = "playground_sampling_demo"

    # Ensure wandb section exists and disable it for the demo
    if 'wandb' not in config:
        config.wandb = ml_collections.ConfigDict()
    config.wandb.no_wandb = True
    config.wandb.online = False
    config.wandb.project = "playground_demo"
    config.wandb.wandb_usr = "your_wandb_username" # Optional: Set your username

    # Ensure sampler section exists and configure it
    if 'sampler' not in config:
        config.sampler = ml_collections.ConfigDict()
    config.sampler.name = 'pc' # Example: Predictor-Corrector sampler
    config.sampler.predictor = 'reverse_diffusion'
    config.sampler.corrector = 'langevin'
    config.sampler.snr_x = 0.16
    config.sampler.snr_A = 0.16
    config.sampler.scale_eps_x = 1.0
    config.sampler.scale_eps_A = 1.0

    # Ensure sample section exists and configure it
    if 'sample' not in config:
        config.sample = ml_collections.ConfigDict()
    config.sample.use_ema = True
    config.sample.seed = 42
    config.sample.probability_flow = False
    config.sample.noise_removal = True
    config.sample.num_steps = 500 # Adjust as needed
    config.sample.eps = 1e-5
    print("Sampling configuration applied.")
    # --- End Sampler Configuration ---
    return config

def demonstrate_dataset_loading(config):
    """Loads the dataset using MyDataset and demonstrates DataLoader."""
    print("\n--- Starting Dataset Loading Demonstration ---")
    dataset_handler = None
    train_loader = None
    test_loader = None
    try:
        # Instantiate MyDataset
        # This internally calls load_from_file and sets relevant config values
        dataset_handler = MyDataset(config)
        print(f"Dataset '{config.data.name}' loaded via MyDataset.")
        # Access max values calculated/updated by MyDataset
        print(f"Max nodes (from config after MyDataset): {config.data.max_node_num}")
        print(f"Max features (from config after MyDataset): {config.data.max_feat_num}")
        print(f"Tagset size (feature dim if degree_as_tag=True): {len(dataset_handler.tagset)}")


        # Get DataLoaders
        train_loader, test_loader = dataset_handler.get_loaders()
        print("Train and Test DataLoaders obtained.")

        # Optional: Demonstrate iterating through a few batches
        print("\n--- Demonstrating DataLoader Iteration (Train Loader) ---")
        batch_count = 0
        if train_loader:
            for batch in train_loader:
                # Assuming batch structure is (x, adj, labels) based on MyDataset.get_loaders
                # Adjust if the structure is different
                if len(batch) == 3:
                     x_batch, adj_batch, labels_batch = batch
                     print(f"Batch {batch_count + 1}:")
                     print(f"  Node Features Shape (x): {x_batch.shape}")
                     print(f"  Adjacency Matrix Shape (adj): {adj_batch.shape}")
                     print(f"  Labels Shape: {labels_batch.shape}")
                else:
                     print(f"Batch {batch_count + 1}: Unexpected batch structure - {len(batch)} items")

                batch_count += 1
                if batch_count >= 2: # Show first 2 batches
                    break
            if batch_count == 0:
                print("Train loader yielded no batches.")
        else:
            print("Train loader is None.")

    except FileNotFoundError as e:
         print(f"\nError loading dataset: {e}")
         print("Please ensure the dataset file and necessary split files exist at the expected locations.")
    except Exception as e:
         print(f"\nAn unexpected error occurred during dataset loading: {e}")
         import traceback
         traceback.print_exc() # Print detailed traceback for debugging
    finally:
        print("--- Finished Dataset Loading Demonstration ---")
        # Return necessary objects, config might have been updated by MyDataset
        return config, dataset_handler, train_loader, test_loader

def demonstrate_sampling(config):
    """Instantiates and runs the Sampler."""
    print("\n--- Starting Sampler Demonstration ---")
    try:
        # Ensure the checkpoint path is set
        if not config.get('ckpt'):
             print("Error: Checkpoint path (config.ckpt) is not set.")
             return

        print(f"Using checkpoint: {config.ckpt}")
        sampler = Sampler(config)
        print("Sampler initialized.")

        # Run sampling
        # independent=True handles wandb init internally if needed (though disabled here)
        sampling_results = sampler.sample(independent=True)
        print("\nSampling finished.")
        print("Sampling Results (e.g., MMD metrics):")
        # Print results nicely if it's a dictionary or similar
        if isinstance(sampling_results, dict):
            for key, value in sampling_results.items():
                print(f"  {key}: {value}")
        else:
            print(sampling_results)

    except FileNotFoundError as e:
        print(f"\nError during sampling: {e}")
        print(f"Critical: Could not find the checkpoint file specified in config.ckpt: {config.ckpt}")
        print("Please verify the path and ensure the file exists.")
    except AttributeError as e:
        print(f"\nError during sampling: Missing configuration? {e}")
        print("Please ensure all required configuration fields for the Sampler are set.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during sampling: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        print("--- Finished Sampler Demonstration ---")


if __name__ == "__main__":
    # 1. Load Base Configuration
    base_config_path = project_root / "configs" / "enzymes_configs" / "enzymes_train_score.yaml"
    config = load_config(base_config_path)

    # 2. Load Dataset and Demonstrate Loaders
    # MyDataset might modify the config object (e.g., max_node_num)
    config, dataset_handler, train_loader, test_loader = demonstrate_dataset_loading(config)

    # 3. Configure for Sampling (potentially using updated config from dataset loading)
    config = configure_for_sampling(config)

    # 4. Demonstrate Sampling
    demonstrate_sampling(config)

    print("\nPlayground script finished.")