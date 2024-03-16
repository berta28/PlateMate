import subprocess
import argparse

def setup(conda_env_name: str = "plate8"):
    # Create the Anaconda environment
    subprocess.run(["conda", "create", "-n", conda_env_name, "python=3.9"], check=True)

    # Activate the Anaconda environment
    subprocess.run(["conda", "activate", conda_env_name], check=True)

    # Install the packages from requirements.txt
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda_env_name", type=str, default="plate8", help="Name of the Anaconda environment")
    args = parser.parse_args()

    setup(args.conda_env_name)