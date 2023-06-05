import subprocess

def execute_python_file(file_path):
    try:
        # Execute the Python file as a separate process
        subprocess.run(['python', file_path], check=True)
    except subprocess.CalledProcessError as e:
        # Handle any errors that occur during the execution
        print(f"Error executing {file_path}: {e}")