from setuptools import setup, find_packages

setup(
    name='sower_client',
    version='1.0',
    package_dir = {'service': 'src'},
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'sower-start = sower_client.script:main',
        ],
    },
    install_requires=[
        'pyzmq',  # ZeroMQ library dependency
    ],
)
