import fastfilters
import skimage
import numpy as np
import datetime
from ndstructs.array5D import Array5D
from ndstructs.point5D import Shape5D

raw_img = np.asarray(skimage.io.imread("../gpu_img_features/big.png"))
print(raw_img.shape)

source_data = Array5D(arr=raw_img, axiskeys="yxc")

step_shape: Shape5D = source_data.shape.updated(c=1)
for (idx, channel_slice) in enumerate(source_data.split(step_shape)):
    start = datetime.datetime.now()
    features  = fastfilters.gaussianSmoothing(channel_slice.raw("yx"), sigma=10.0, window_size=3.5)
    t = datetime.datetime.now() - start
    print(f"Took {t} to process one slice")

# print(img.shape)




