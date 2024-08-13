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
docker run -d --name sower_platform_container -p 8000:8000 -p 8080:8080 <your-account>/sower_platform:latest
```

## In all nodes

### To install agent services, run the following commands:

```bash
wget https://github.com/<your-account>/Sower/releases/download/latest/sower_agent-1.0.tar.gz
```

### Then, install the Sower Agent package:

```bash
pip install sower_agent-1.0.tar.gz
```

This step installs `sower_agent` with its dependencies. The `pip install` command installs the package from the specified `.tar.gz` file, which was created in the previous step during the package build process.


### Last, setup the agent as a service:

```bash
sower start
```

## Start Training
### Access the platform by
<http://localhost:8000/admin>

### Click the "Start Training button"
