import os

def main() -> None:
    """Quick diagnostic to verify where the script is running from and data folder visibility."""
    print("--- DEBUGGING PATHS ---")
    print(f"1. Current Terminal Location: {os.getcwd()}")

    expected_data_path = "../data"
    if os.path.exists(expected_data_path):
        print("2. Found '../data' folder? YES")
        print(f"3. Files inside '../data': {os.listdir(expected_data_path)}")
    else:
        print("2. Found '../data' folder? NO")
        print("   (This usually means the script is running from a different folder.)")
        if os.path.exists("data"):
            print("   A 'data' folder exists in the current directory. Consider using 'data/...'.")


if __name__ == "__main__":
    main()