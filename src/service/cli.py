import subprocess

def main():
    subprocess.run(['sudo', 'systemctl', 'start', 'sower-agent.service'])
    print("Service started.")