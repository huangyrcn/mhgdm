import os
import sys

def setup_configs():
    # Get the absolute path to the HGDM directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define source and target paths
    source_dir = os.path.join(base_dir, 'config')
    target_dir = os.path.join(base_dir, 'configs')
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return False
    
    # Create symbolic link if it doesn't exist
    if not os.path.exists(target_dir):
        try:
            # Create symbolic link
            os.symlink(source_dir, target_dir, target_is_directory=True)
            print(f"Created symbolic link from '{source_dir}' to '{target_dir}'")
            return True
        except OSError as e:
            print(f"Error creating symbolic link: {e}")
            return False
    else:
        print(f"'{target_dir}' already exists. No action taken.")
        return True

if __name__ == "__main__":
    if setup_configs():
        print("Setup completed successfully.")
    else:
        print("Setup failed.")
        sys.exit(1)
