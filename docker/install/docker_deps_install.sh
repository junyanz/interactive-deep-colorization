
wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.7.10-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# apt-get update
# apt-get install python3-pip python-opencv 
# pip install scikit-image scikit-learn qdarkstyle
