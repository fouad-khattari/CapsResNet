import random
from torchvision.transforms import RandomErasing

class CustomRandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=0.1307):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        eraser = RandomErasing(probability=self.probability, sl=self.sl, sh=self.sh, r1=self.r1, mean=self.mean)
        return eraser(img)
