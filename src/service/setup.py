import os
from setuptools import setup, find_packages

def create_service_file():
    service_content = """[Unit]
Description=Sower Agent Service
After=network.target

[Service]
ExecStart=sower start
Restart=always

[Install]
WantedBy=multi-user.target
"""
    os.makedirs('systemd', exist_ok=True)
    with open('systemd/sower-agent.service', 'w') as f:
        f.write(service_content)

create_service_file()

setup(
    name='sower-agent',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'agent-start = cli:main',
        ],
    },
    install_requires=[
        'paho-mqtt',  # MQTT library dependency
        'docker', # docker library for python
    ],
    data_files=[('/etc/systemd/system', ['systemd/sower-agent.service'])],
)
