import sys
import os

print(f"Python version: {sys.version}")
print(f"Executable: {sys.executable}")

print("\n--- Checking numpy ---")
try:
    import numpy
    print(f"numpy version: {numpy.__version__}")
    print(f"numpy path: {numpy.__file__}")
except ImportError as e:
    print(f"Failed to import numpy: {e}")

print("\n--- Checking onnxruntime ---")
try:
    import onnxruntime
    print(f"onnxruntime version: {onnxruntime.__version__}")
    print(f"onnxruntime path: {onnxruntime.__file__}")
except ImportError as e:
    print(f"Failed to import onnxruntime: {e}")

print("\n--- Checking rembg ---")
try:
    import rembg
    print(f"rembg version: {rembg.__version__}")
    print(f"rembg path: {rembg.__file__}")
    
    from rembg import remove
    print("Successfully imported 'remove' from rembg")
except ImportError as e:
    print(f"Failed to import rembg: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"An error occurred while importing rembg: {e}")
    import traceback
    traceback.print_exc()
