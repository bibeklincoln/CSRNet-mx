import mxnet as mx
import numpy as np
import cv2
import symbol_csrnet
from collections import namedtuple
from mutableModule import MutableModule

MEAN_COLOR = np.array([110.474, 118.574, 123.955]).reshape((3, 1, 1)) # BGR 

def load_checkpoint(prefix, epoch):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

sym = symbol_csrnet.get_symbol()

ctx = mx.cpu(0)

mod = MutableModule(
        context = ctx,
        symbol = sym,
        data_names = ("data", ),
        label_names = ()
)

mod.bind(data_shapes = [("data", (4, 3, 512, 512))])

mod.load_params("./models/shanghaia.params")

Batch = namedtuple('Batch', ['data', 'provide_data'])

def predict(img):
    rows, cols, ts = img.shape

    num_sub_rows = num_sub_cols = 2

    sub_rows = rows // num_sub_rows 
    sub_cols = cols // num_sub_cols 

    img = (img.transpose((2,0,1)).reshape((3, rows, cols)) - MEAN_COLOR).astype(np.float32)

    sum_p = 0.0
    sub_imgs = []
    for r in range(num_sub_rows):
        for c in range(num_sub_cols):
            sub_img = img[:, sub_rows*r:sub_rows*(r+1), sub_cols*c:sub_cols*(c+1)].reshape((3, sub_rows, sub_cols))
            sub_imgs.append(sub_img)

    bdata = (mx.nd.array(sub_imgs, ctx))
    mod.forward(Batch(data = [bdata], provide_data = [('data', bdata.shape)]), is_train = False)
    outputs = mod.get_outputs()[0]
    p = outputs.sum().asscalar()
    return p

if __name__ == '__main__':
    import os
    def is_image_file(fname):
        ext = os.path.splitext(fname)[-1].lower()
        return ext == '.jpg'
    data_path = './data/ShanghaiTech/part_A_final/test_data/'
    image_names = filter(is_image_file, os.listdir(os.path.join(data_path, 'images')))

    fout = open('predict.txt', 'w')
    num_images = len(image_names)
    for i, img_name in enumerate(image_names):
        fname = os.path.join(data_path, 'images', img_name)
        img = cv2.imread(fname) # BGR
        p = predict(img)
        print ('{}/{}: {} {}'.format(i + 1, num_images, img_name, p))
        fout.write('{} {}\n'.format(img_name, p))
