# Was having major issues finding the dataset so I created this

import os

print("--- DEBUGGING PATHS ---")
print(f"1. Current Terminal Location: {os.getcwd()}")

# Check if 'data' folder exists relative to here
expected_data_path = '../data'
if os.path.exists(expected_data_path):
    print(f"2. Found '../data' folder? YES")
    print(f"3. Files inside '../data': {os.listdir(expected_data_path)}")
else:
    print(f"2. Found '../data' folder? NO")
    print("   (This means you are running the script from the wrong folder!)")
    
    # Check if 'data' exists in the current folder (Root)
    if os.path.exists('data'):
        print("   But I found a 'data' folder right here! You should remove '../' from your path.")