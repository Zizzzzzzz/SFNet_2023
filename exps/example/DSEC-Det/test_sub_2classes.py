
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 2
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)

        # Modify the path to your own
        self.data_dir = '/root/data1/dataset/DSEC'

        # name of annotation file for training
        self.train_ann = "sub/train_2classes.json"
        # name of annotation file for evaluation
        self.val_ann = "sub/test_2classes.json"
        # name of annotation file for testing
        self.test_ann = "sub/test_2classes.json"

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (640, 640)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65
