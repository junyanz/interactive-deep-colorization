
# Interactive Deep Colorization

### [[Project Page]](https://richzhang.github.io/ideepcolor/) [[Paper]]() [[Demo Video]](https://youtu.be/eL5ilZgM89Q) [[Talk]](https://www.youtube.com/watch?v=FTzcFsz2xqw&feature=youtu.be&t=992)
<img src='imgs/demo.gif' width=600>

<!-- This repository contains a Python implementation for user-guided image colorization technique described in the following paper: -->
[Richard Zhang](https://richzhang.github.io/)\*, [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/)\*, [Phillip Isola](http://people.eecs.berkeley.edu/~isola/), [Xinyang Geng](http://young-geng.xyz/), Angela S. Lin, Tianhe Yu, and [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/).
**Real-Time User-Guided Image Colorization with Learned Deep Priors.**
In ACM Transactions on Graphics (SIGGRAPH 2017).
(\*indicates equal contribution)

We first describe the system <b>(0) Prerequisities</b> and steps for <b>(1) Getting started</b>. We then describe the interactive colorization demo <b>(2) Interactive Colorization (Local Hints Network)</b>. There are two demos: (a) a "barebones" version in iPython notebook and (b) the full GUI we used in our paper. We then provide an example of the <b>(3) Global Hints Network</b>.

<img src='https://richzhang.github.io/ideepcolor/index_files/imagenet_showcase_small.jpg' width=800>

### (0) Prerequisites
- Linux or OSX
- [Caffe](http://caffe.berkeleyvision.org/installation.html)
- CPU or NVIDIA GPU + CUDA CuDNN.

### (1) Getting Started
- Clone this repo:
```bash
git clone https://github.com/junyanz/interactive-deep-colorization ideepcolor
cd ideepcolor
```

- Download the reference model
```
bash ./models/fetch_reference_model.sh
```

- Install [Caffe](http://caffe.berkeleyvision.org/installation.html) and Python libraries ([OpenCV](http://opencv.org/))

### (2) Interactive Colorization (Local Hints Network)
<img src='imgs/teaser_v3.jpg' width=800>

We provide a "barebones" demo in iPython notebook, which does not require QT. We also provide our full GUI demo.

#### 2(a) Barebones Interactive Colorization Demo

- Run `ipython notebook` and click on [`DemoInteractiveColorization.ipynb`](./DemoInteractiveColorization.ipynb).

#### 2(b) Full Demo GUI

- Install [Qt4](https://wiki.python.org/moin/PyQt4) and [QDarkStyle](https://github.com/ColinDuquesnoy/QDarkStyleSheet). (See [Requirements](## (A) Requirements))

- Run the UI: `bash python ideepcolor.py --gpu [GPU_ID]`. Arguments are described below:
```
--win_size    [512] GUI window size
--gpu         [0] GPU number
--image_file  ['./tesT_imgs/mortar_pester.jpg'] path to the image file
```

- User interactions

<img src='./imgs/pad.jpg' width=800>

- <b>Adding points</b>: Left-click somewhere on the input pad
- <b>Moving points</b>: Left-click and hold on a point on the input pad, drag to desired location, and let go
- <b>Changing colors</b>: For currently selected point, choose a recommended color (middle-left) or choose a color on the ab color gamut (top-left)
- <b>Removing points</b>: Right-click on a point on the input pad
- <b>Changing patch size</b>: Mouse wheel changes the patch size from 1x1 to 9x9
- <b>Load image</b>: Click the load image button and choose desired image
- <b>Restart</b>: Click on the restart button. All points on the pad will be removed.
- <b>Save result</b>: Click on the save button. This will save the resulting colorization in a directory where the ```image_file``` was, along with the user input ab values.
- <b>Quit</b>: Click on the quit button.

### (3) Global Hints Network
<img src='https://richzhang.github.io/ideepcolor/index_files/lab_all_figures45k_small.jpg' width=800>

We include an example usage of our Global Hints Network, applied to global histogram transfer. We show its usage in an iPython notebook.

- Add `./caffe_files` to your `PYTHONPATH`

- Run `ipython notebook`. Click on [`./DemoGlobalHistogramTransfer.ipynb`](./DemoGlobalHistogramTransfer.ipynb)`

## (A) Requirements
- Caffe (See Caffe installation [document](http://caffe.berkeleyvision.org/installation.html))
- OpenCV
```
sudo apt-get install python-opencv
```
- Qt4
```
sudo apt-get install python-qt4
```
- QDarkStyle
```
sudo pip install qdarkstyle
```

## Cat Paper Collection
If you love cats, and love reading cool graphics, vision, and learning papers, please check out the Cat Paper Collection:  
[[Github]](https://github.com/junyanz/CatPapers) [[Webpage]](http://people.eecs.berkeley.edu/~junyanz/cat/cat_papers.html)
