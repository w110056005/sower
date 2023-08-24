from setuptools import setup, find_packages

setup(
    name='sower_client',
    version='1.0',
    package_dir = {'service': 'sower_client'},
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'sower = sower_client.script:main',
        ],
    },
    install_requires=[
        'paho-mqtt',  # MQTT library dependency
        'docker', # docker library for python
    ],
)
