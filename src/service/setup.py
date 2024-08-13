from setuptools import setup, find_packages

setup(
    name='sower_agent',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'sower = sower_agent.script:main',
        ],
    },
    install_requires=[
        'paho-mqtt',  # MQTT library dependency
        'docker', # docker library for python
    ],
)
