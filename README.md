# Sower

## Before we start
1. Make sure you install Docker ([https://www.docker.com/](https://www.docker.com/)) in all Client/Server environments.
2. Make sure you have a GitHub account and create a Docker Hub account using your GitHub account.
3. Fork this repository to your own account.

## In Server
### 1. pull Platform images
```bash
docker pull <your-account>/sower_platform:latest
```
### 2. run platform
```bash
docker run -d --name sower_platform_container -p 5555:5555  -p 8000:8000 -p 8080:8080 sower_platform:latest
```

## In all clients

### To install client services, run the following commands:

```bash
wget https://github.com/<your-account>/Sower/releases/download/latest/sower_client-1.0.tar.gz
```

### Then, install the Sower Client package:

```bash
pip install sower_client-1.0.tar.gz
```

This step installs `sower_client` with its dependencies. The `pip install` command installs the package from the specified `.tar.gz` file, which was created in the previous step during the package build process.


### Last, setup the client as a service:

```bash
sudo cp sower_client.service /etc/systemd/system/

sudo systemctl enable sower_client
sudo systemctl start sower_client

sudo systemctl status sower_client
```

## Start Training
### Access the platform by
<http://localhost:8000/admin>

### Click the "Start Training button"
