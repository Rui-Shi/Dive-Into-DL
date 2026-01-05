import torch
import torchvision
import sys
import subprocess
import platform

def check_versions():
    """Prints the versions of PyTorch, Torchvision, and CUDA."""
    
    print(f"Python Version: {sys.version}")
    print("-" * 30)
    
    # Check PyTorch
    print(f"PyTorch Version: {torch.__version__}")
    print("-" * 30)

    # Check Torchvision
    print(f"Torchvision Version: {torchvision.__version__}")
    print("-" * 30)
    
    print("Reading hardware info...")
    
    # --- 1. Get CPU Name ---
    # We use 'wmic' (Windows command) because platform.processor() is often vague
    try:
        command = "wmic cpu get name"
        # Run command and decode output
        output = subprocess.check_output(command, shell=True).decode().strip()
        # Output comes as "Name \n CPU_Model", so we take the last line
        cpu_name = output.split('\n')[-1].strip()
    except:
        cpu_name = platform.processor() # Fallback

    # --- 2. Get RAM ---
    try:
        command = "wmic computersystem get TotalPhysicalMemory"
        output = subprocess.check_output(command, shell=True).decode().strip()
        # Convert bytes to GB
        total_bytes = float(output.split('\n')[-1].strip())
        ram_gb = round(total_bytes / (1024**3), 2) # 1024^3 = Bytes to GB
    except:
        ram_gb = "Unknown"

    print("-" * 40)
    print(f"CPU Model:  {cpu_name}")
    print(f"Total RAM:  {ram_gb} GB")
    print("-" * 40)

    # --- 3. Check CUDA ---
    print("--- CUDA Information ---")
    if torch.cuda.is_available():
        print("CUDA is AVAILABLE")
        # This is the CUDA version PyTorch was compiled with
        print(f"PyTorch-linked CUDA Version: {torch.version.cuda}")
        
        # Get details for each available GPU
        device_count = torch.cuda.device_count()
        print(f"Detected {device_count} CUDA-capable device(s).")
        for i in range(device_count):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Total memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    else:
        print("CUDA is NOT available.")
        print("PyTorch is running in CPU-only mode.")
    print("-" * 30)

if __name__ == "__main__":
    check_versions()
