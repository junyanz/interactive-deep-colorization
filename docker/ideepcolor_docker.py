from __future__ import print_function
import sys
from PyQt5.QtWidgets import *
import argparse
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from ui_PyQt5 import gui_design
from data import colorize_image as CI

sys.path.append('./caffe_files')


def parse_args():
    parser = argparse.ArgumentParser(description='iDeepColor: deep interactive colorization')
    # basic parameters
    parser.add_argument('--win_size', dest='win_size', help='the size of the main window', type=int, default=512)
    parser.add_argument('--image_file', dest='image_file', help='input image', type=str, default='test_imgs/mortar_pestle.jpg')
    parser.add_argument('--gpu', dest='gpu', help='gpu id', type=int, default=0)
    parser.add_argument('--cpu_mode', dest='cpu_mode', help='do not use gpu', action='store_true')

    # Caffe - Main colorization model
    parser.add_argument('--color_prototxt', dest='color_prototxt', help='colorization caffe prototxt', type=str,
                        default='./models/reference_model/deploy_nodist.prototxt')
    parser.add_argument('--color_caffemodel', dest='color_caffemodel', help='colorization caffe prototxt', type=str,
                        default='./models/reference_model/model.caffemodel')

    # Caffe - Distribution prediction model
    parser.add_argument('--dist_prototxt', dest='dist_prototxt', type=str, help='distribution net prototxt',
                        default='./models/reference_model/deploy_nopred.prototxt')
    parser.add_argument('--dist_caffemodel', dest='dist_caffemodel', type=str, help='distribution net caffemodel',
                        default='./models/reference_model/model.caffemodel')

    # PyTorch (same model used for both)
    parser.add_argument('--color_model', dest='color_model', help='colorization model', type=str,
                        default='./models/pytorch/caffemodel.pth')
    parser.add_argument('--dist_model', dest='color_model', help='colorization distribution prediction model', type=str,
                        default='./models/pytorch/caffemodel.pth')

    parser.add_argument('--backend', dest='backend', type=str, help='caffe or pytorch', default='pytorch')
    parser.add_argument('--pytorch_maskcent', dest='pytorch_maskcent', help='need to center mask (activate for siggraph_pretrained but not for converted caffemodel)', action='store_true')

    # ***** DEPRECATED *****
    parser.add_argument('--load_size', dest='load_size', help='image size', type=int, default=256)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    if args.cpu_mode:
        args.gpu = -1

    args.win_size = int(args.win_size / 4.0) * 4  # make sure the width of the image can be divided by 4

    if args.backend == 'caffe':
        # initialize the colorization model
        colorModel = CI.ColorizeImageCaffe(Xd=args.load_size)
        colorModel.prep_net(args.gpu, args.color_prototxt, args.color_caffemodel)

        distModel = CI.ColorizeImageCaffeDist(Xd=args.load_size)
        distModel.prep_net(args.gpu, args.dist_prototxt, args.dist_caffemodel)
    elif args.backend == 'pytorch':
        colorModel = CI.ColorizeImageTorch(Xd=args.load_size,maskcent=args.pytorch_maskcent)
        colorModel.prep_net(path=args.color_model)

        distModel = CI.ColorizeImageTorchDist(Xd=args.load_size,maskcent=args.pytorch_maskcent)
        distModel.prep_net(path=args.color_model, dist=True)
    else:
        print('backend type [%s] not found!' % args.backend)

    # initialize application
    app = QApplication(sys.argv)
    window = gui_design.GUIDesign(color_model=colorModel, dist_model=distModel,
                                  img_file=args.image_file, load_size=args.load_size, win_size=args.win_size)
    #app.setStyleSheet(qdarkstyle.load_stylesheet(pyside=False))  # comment this if you do not like dark stylesheet
    app.setWindowIcon(QIcon('imgs/logo.png'))  # load logo
    window.setWindowTitle('iColor')
    window.setWindowFlags(window.windowFlags() & ~Qt.WindowMaximizeButtonHint)   # fix window siz
    window.show()
    app.exec_()
