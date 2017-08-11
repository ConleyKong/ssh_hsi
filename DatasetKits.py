"""
4.0 数据组织时增加交叉验证的支持，放弃开始就对原始数据进行shuffle的操作
-------------------
3.0   重新组织对于频谱二维化的形式，尝试使用10xN的方式将178个波段信息二维化，缺少的部分使用0进行补充
-------------------
2.3.3 放弃标准化操作
2.3.1 使用MinMaxScalar替换StandardScaler，防止出现负数
2.3 增加数据预处理中的标准化操作
-------------------
"""
import collections
import scipy.io as sio
import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
# from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from hsi_inference import IMAGE_WIDTH_SIZE
from hsi_inference import LABEL_SIZE

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet(object):
    def __init__(self, images, labels, fake_data=False, one_hot=False, dtype=dtypes.float32, reshape=False, seed=None,
                 channel_ceiling=65535):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]
            # print("样本总量：",self._num_examples)
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            # if reshape:
            #   # assert images.shape[3] == 1
            #   assert images.shape[-1] != None
            #   images = images.reshape(images.shape[0],
            #                           images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, channel_ceiling] -> [0.0, 1.0].
                images = images.astype(np.float32)
                images = np.multiply(images, 1.0 / channel_ceiling)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._sample_length = images.shape[1]

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def sample_length(self):
        return self._sample_length

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
            ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples  # 是个负数，从后向前数
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


# def read_ksc_sets(train_dir, fake_data=False, one_hot=False, dtype=dtypes.float32, reshape=True, seed=None,
#                   validation_ratio=0.2, test_ratio=0.2, remove_class_zero = True,shuffle=True):
"""
    version4.0之后增加留一验证参数，去除对原始数据的shuffle功能
    @:param cross_index 留一的坐标，0-9之间的整数，整体数据10等分，取某一部分作为测试集
"""
def read_ksc_sets(train_dir, fake_data=False, one_hot=False, dtype=dtypes.float32, reshape=True, seed=None,
                  validation_ratio=0.1, train_ratio=0.9, remove_class_zero=True, cross_index=0):
    if fake_data:
        def fake():
            return DataSet(
                [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

        train = fake()
        validation = fake()
        test = fake()
        return Datasets(train=train, validation=validation, test=test)
    image_mat = sio.loadmat(os.path.join(train_dir, "KSCData.mat"))
    """
    #测试代码
      # print(image_mat.keys())   #dict_keys(['DataSet', '__version__', '__globals__', '__header__'])
      # print(image_mat['DataSet'].shape)  # 包含512x614个像素，每个像素176个频道，共314368条数据
    """
    images = image_mat['DataSet']
    """
      #测试代码
      # print("数据存储类型：",type(images))     #数据存储类型： <class 'np.ndarray'>  
      # print("波段最大值为：",images.max())     #波段最大值为： 65535
    """
    dim_list = list(images.shape)
    raw_flated_images = np.reshape(images, [-1, dim_list[-1]])  # 得到的是 img_length * channel的一个二维矩阵
    """
    #测试reshape之后的结构，符合预期
        print("第二行的图像数据：",images[1][:2][:])
        print("flat之后：",flated_images[dim_list[1]:dim_list[1]+2][:])
    """
    img_length = dim_list[0] * dim_list[1]  # 计算整个高光谱图像的像素点个数
    """
        根据卷积核大小，使用np.append填充二维化的频谱信息
    """
    channel_length = dim_list[-1]
    _2dim_width = IMAGE_WIDTH_SIZE
    _appendix_width = _2dim_width-(channel_length%_2dim_width)
    _appendix_array = np.zeros([img_length,_appendix_width])
    flated_images = np.append(raw_flated_images,_appendix_array,axis=1)

    training_size = int(img_length * train_ratio)
    validation_size = int(training_size * validation_ratio)

    label_mat = sio.loadmat(os.path.join(train_dir, "KSCGt.mat"))
    """
    #测试代码
        print(label_mat.keys())  # dict_keys(['__version__', '__globals__', '__header__', 'ClsID'])
        print(label_mat['ClsID'].shape)  # 包含512x614个像素，每个像素一个分类
    """
    labels = label_mat['ClsID']
    _1d_labels = np.reshape(labels, [img_length, ])  # labels是一维的数据，长度为img_length
    """
      for a in _1d_labels:
        if(a!=0):
          print(a)
    """
    flated_labels = np.ndarray(shape=(img_length, LABEL_SIZE), dtype=float)
    _row_counter = 0
    for row in flated_labels:
        j = _1d_labels[_row_counter]
        # print(j)
        row[j] = 1.0
        flated_labels[_row_counter] = row
        _row_counter += 1
    """
        print(type(flated_labels))
        for x in flated_labels:
            if(x[0]!=1.0):
            print(x)
    """
    """
        print("gt存储类型：",type(labels))  # gt存储类型： <class 'np.ndarray'>
        print("标签中的最大值：",labels.max())    #标签中的最大值： 13
        print("标签中的最小值：", labels.min())   #标签中的最小值： 0
    """

    if(remove_class_zero):
        _nest_ele_num = 0
        _nest_images = []       #借用列表的append方法来追加建立纯净的应用数据
        _nest_labels = []       #借用列表的append方法来追加建立纯净的应用数据
        for (i,l) in zip(flated_images,flated_labels):
            if(l[0] != 1):      #对于有效数据，0下的标记不能为1
                _nest_images.append(i)
                _nest_labels.append(l)
                _nest_ele_num+=1
        """
            for l in _nest_labels:
                print("label:",l)
        """
        flated_size = _nest_ele_num
        training_size = int(flated_size * train_ratio)
        validation_size = int(training_size*validation_ratio)
        flated_images = np.array(_nest_images,dtype=np.float32)
        flated_labels = np.array(_nest_labels,dtype=np.float32)
        # print("data shape: ",np.shape(flated_images),"label shape: ",np.shape(flated_labels))
    """
        for (i,j) in zip(flated_images,flated_labels):
            print("image: ",i,"label: ",j)
    """
    """
    #为增加随机性，增加shuffle功能
    if shuffle:
        perm = np.arange(flated_size)
        np.random.shuffle(perm)
        _images = flated_images[perm]
        _labels = flated_labels[perm]
        flated_images = _images
        flated_labels = _labels
    """


    train_images = flated_images[:training_size][:]
    train_labels = flated_labels[:training_size][:]

    test_images = flated_images[training_size:][:]
    test_labels = flated_labels[training_size:][:]

    if not 0 <= validation_size <= training_size:
      raise ValueError(
          'Validation size should be between 0 and {}. Received: {}.'
          .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    print("训练样本总量：", train_images.shape[0])
    print("测试样本总量：", test_images.shape[0])
    print("验证样本总量：", validation_images.shape[0])

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape, seed=seed)
    validation = DataSet(validation_images,validation_labels,dtype=dtype,reshape=reshape,seed=seed)
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape, seed=seed)

    return Datasets(train=train, validation=validation, test=test)


def read_preped_ksc_sets(data_dir, neighbor=1, train_ratio_size=80, validation_ratio=0.1, dtype=dtypes.float32, reshape=True, seed=None):
    file_name = "KSC_" + str(neighbor) + "_" + str(train_ratio_size) + ".mat"
    image_mat = sio.loadmat(os.path.join(data_dir, file_name))
    """
     # 测试代码
     print(image_mat.keys())  # dict_keys(['CIdTe', 'DataTe', 'PositionsTr', 'DataTr', '__globals__', 'PositionsTe', '__header__', '__version__', 'CIdTr'])
     print("训练数据：",image_mat['DataTr'].shape," 测试数据：",image_mat['DataTe'].shape)  #训练数据： (4174, 176)  测试数据： (1037, 176)
     print("训练标签：",image_mat['CIdTr'].shape," 测试标签：",image_mat['CIdTe'].shape)    #训练标签： (1, 4174)  测试标签： (1, 1037)
    """
    train_images = image_mat['DataTr']
    train_dim = train_images.shape
    training_size = int(train_dim[0])
    channel_length = train_dim[-1]
    _2dim_width = IMAGE_WIDTH_SIZE
    _appendix_width = _2dim_width-(channel_length%_2dim_width)
    _appendix_array = np.zeros([training_size,_appendix_width])
    train_images = np.append(train_images,_appendix_array,axis=1)
    """
        print("extended shape of train images:",train_images.shape)
    """
    raw_train_labels = image_mat['CIdTr']
    raw_train_labels.shape = (1,training_size)
    _1d_train_labels = raw_train_labels.transpose()
    #将标记转化为label向量
    train_labels = np.ndarray(shape=(training_size, LABEL_SIZE), dtype=float)
    _row_counter = 0
    for row in train_labels:
        j = _1d_train_labels[_row_counter]
        # print(j)
        row[j] = 1.0
        train_labels[_row_counter] = row
        _row_counter += 1
    """
        print("训练标签维度：",train_labels.shape)
    """

    test_images = image_mat['DataTe']
    test_dim = test_images.shape
    test_size = int(test_dim[0])
    channel_length = train_dim[-1]
    _2dim_width = IMAGE_WIDTH_SIZE
    _appendix_width = _2dim_width - (channel_length % _2dim_width)
    _appendix_array = np.zeros([test_size, _appendix_width])
    test_images = np.append(test_images, _appendix_array, axis=1)
    raw_test_labels = image_mat['CIdTe']
    raw_test_labels.shape = (1,test_size)
    _1d_test_labels = raw_test_labels.transpose()
    # 将标记转化为label向量
    test_labels = np.ndarray(shape=(test_size, LABEL_SIZE), dtype=float)
    _row_counter = 0
    for row in test_labels:
        j = _1d_test_labels[_row_counter]
        # print(j)
        row[j] = 1.0
        test_labels[_row_counter] = row
        _row_counter += 1

    validation_size = int(training_size * validation_ratio)
    if not 0 <= validation_size <= training_size:
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))
    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]
    print("the shape of train_labels",train_labels.shape)
    """
        print("校正后标签维度：",train_labels.shape)
    """

    print("训练样本总量：", train_images.shape[0])
    print("测试样本总量：", test_images.shape[0])
    print("验证样本总量：", validation_images.shape[0])

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape, seed=seed)
    validation = DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape, seed=seed)
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape, seed=seed)

    return Datasets(train=train, validation=validation, test=test)

def load_ksc(train_dir='../DATA/KSC'):
    return read_ksc_sets(train_dir)

def load_preped_ksc(data_dir='../DATA/KSC',neighbor=1,train_ratio=0.8):
    train_ratio_size = int(train_ratio * 100)
    return read_preped_ksc_sets(data_dir, neighbor, train_ratio_size=train_ratio_size)

if __name__ == '__main__':
    # load_ksc()
    load_preped_ksc()