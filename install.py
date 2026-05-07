import launch
import os
import sys
import subprocess

current_dir = os.path.dirname(os.path.realpath(__file__))
req_file = os.path.join(current_dir, "requirements.txt")

# Install regular dependencies
with open(req_file) as f:
    for lib in f:
        lib = lib.strip()
        # Skip empty lines and comments
        if not lib or lib.startswith("#"):
            continue
        if not launch.is_installed(lib):
            launch.run_pip(
                f"install {lib}",
                f"sd-webui-segment-anything3 requirement: {lib}"
            )

# Now handle sam3 separately
try:
    import sam3
    print("SAM3 already installed, skipping installation.")
except ImportError:
    print("SAM3 not found. Installing from official repository...")
    # Option 1: pip install directly from GitHub
    launch.run_pip(
        "install git+https://github.com/facebookresearch/sam3.git",
        "sd-webui-segment-anything3 requirement: sam3"
    )
    # Verify
    try:
        import sam3
        print("SAM3 installed successfully.")
    except ImportError:
        print(
            "ERROR: SAM3 installation failed. Please manually clone the repo and run 'pip install -e .' inside it."
        )