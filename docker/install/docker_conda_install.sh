apt update
apt-get -y install libgl1-mesa-glx
apt-get -y install libqt5x11extras5
conda install -n env -c anaconda protobuf  ## photobuf
conda install -n env -c anaconda scikit-learn ## scikit-learn
conda install -n env -c anaconda scikit-image  ## scikit-image
conda install -n env -c menpo opencv   ## opencv
conda install -n env pyqt ## qt5
conda install -n env -c pytorch pytorch torchvision cudatoolkit=9.0 
