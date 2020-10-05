# Using Docker

### Note

The following works for users on MacOS; I have not tested it on other platforms. 

I've converted the PyQt4 code to PyQt5 in order to make this app easily runnable with Python 3. 

Please follow this guide ([https://sourabhbajaj.com/blog/2017/02/07/gui-applications-docker-mac/](https://sourabhbajaj.com/blog/2017/02/07/gui-applications-docker-mac/)) to install & configure XQuartz (and Docker if you haven't already!)

### Instructions
Assuming you've git-cloned this repository and are currently in the `ideepcolor/` repository:

    cd docker
    bash models/pytorch/fetch_model.sh # Fetches the PyTorch model
    docker build -t colorize .
    docker image ls

You should see a list of Docker images, one of them named `colorize`. 

Then:

    xhost + 127.0.0.1
    docker run -e DISPLAY=host.docker.internal:0 colorize
    
in order to run the app!
