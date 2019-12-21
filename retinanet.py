

# In[1]:


import tensorflow as tf
import  numpy as np
import cv2
print(tf.__version__)


# get_ipython().run_line_magic('matplotlib', 'inline')

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[5]:


import keras
import numpy as np
import math


# In[6]:


# This is for bias initialization explained in section 3.3 of retinanet paper
# This is for stabalization of tranining
# Here p is prior probability which is set to 0.01 generally



class PriorProbability(keras.initializers.Initializer):
    
    def __init__(self, probability=0.01):
        self.probability = probability
    
    def get_config(self):
        return{
            'probability': self.probability
        }
    
    def __call__(self, shape, dtype=None):
        
        return np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)


# In[7]:


class UpsampleLike(keras.layers.Layer):
    
    def call(self, inputs, **kwargs):
        
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == 'channels_first':
            source = tf.transpose(source, (0, 2, 3, 1))
            output = tf.compat.v1.image.resize_images(source, (target_shape[2], target_shape[3]), method='nearest')
            output = tf.transpose(source, (0, 3, 1, 2))
            return output
        else:
            return tf.compat.v1.image.resize_images(source, (target_shape[2], target_shape[3]), method='nearest')
        
    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)
        


# In[8]:


class AnchorParameters:
    
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales
        
    def num_anchors(self):
        return len(self.ratios) * len(self.scales)
    
AnchorParameters.default = AnchorParameters(
    sizes = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)    


# In[9]:



# In[10]:


####################################################

# NOT ABLE TO UNDERSTAND COMPLETELY

####################################################

def generate_anchors(base_size = 16, ratios=None, scales=None):
    
    if ratios is None:
        ratios = AnchorParameters.default.ratios
    
    if scales is None:
        scales = AnchorParameters.default.scales
        
    num_anchors = len(ratios) * len(scales)
    anchors = np.zeros((num_anchors, 4))
    
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    
    areas = anchors[:, 2] * anchors[:, 3]

    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:,3] * 0.5, (2, 1)).T
    
    return anchors
    
    


# In[11]:
def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.
    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors

def shift_temp(shape, stride, anchors):
    
    #It makes a grid starting from 0.5, jumping by strides everytime
    #Eg: if shape = 10 X 5, stride = 2
    #shift_x = [1, 3, 5, 7, ....., 19]
    #shift_y = [1, 3, 5, 7, 9]
    
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    
    #meshgrid: https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy
    
    #It sort of broadcasts itself, rowwise, columnwise
    
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    
    #vstack: Just stacks the rows vertically
    #ravel: .reshape(-1), converts into 1-D array
    
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel())).transpose()
    
    # Do not understand what and why they are doing
    
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    
    return all_anchors
    


# In[12]:


# Anchors Class

# size of the anchor box, centered at all pixels at stride "strde".
class Anchors(keras.layers.Layer):
    
    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        
        self.size = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        
        if ratios is None:
            self.ratios = AnchorParameters.default.ratios
            
        #isinstance checks if ratios is indeed a list    
        elif isinstance(ratios, list):
            self.ratios = np.array(ratios)
        
        if scales is None:
            self.scales = AnchorParameters.default.scales
        elif isinstance(scales, list):
            self.scales = np.array(scales)
        
        
        self.num_anchors = len(ratios) * len(scales)
        self.anchors = keras.backend.variable(generate_anchors(
            base_size=size,
            ratios=ratios,
            scales=scales,))
        
        super(Anchors, self).__init__(*args, **kwargs)
        
    def call(self, inputs, **kwargs):
        features = inputs
        features_shape  = keras.backend.shape(features)
        
        if keras.backend.image_data_format() == 'channels_first':
            anchors = shift(features_shape[2:4], self.stride, self.anchors)
        else:
            anchors = shift(features_shape[1:3], self.stride, self.anchors)
        
        anchors = keras.backend.tile(keras.backend.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))
        
        return anchors
    
    def compute_output_shape(self, input_shape):
        
        if None not in input_shape[1:]:
            
            if keras.backend.image_data_format() == 'channels_first':
                total = np.prod(input_shape[1:3]) * self.num_anchors
                
            else:
                total = np.prod(input_shape[1:3]) * self.num_anchors
                
            return (input_shape[0], total, 4)
        
        else:
            return (input_shape[0], None, 4)
   
    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size' : self.size,
            'stride': self.stride,
            'ratios': self.ratios.tolist(),
            'scales': self.scales.tolist(),
        })
        
        return config


# In[13]:


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability = 0.01,
    classification_feature_size=256,
    name='classification_model'):
    
    options = {
        'kernel_size' : 3, #filter size is 3X3
        'strides' : 1,
        'padding' : 'same', #to keep the dimensions same
    }
    
    if keras.backend.image_data_format() == 'channels_first':
        inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
        
    outputs = inputs
    
    
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
            )(outputs)
        
    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors, # K X A: classes X num_anchors
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=PriorProbability(probability=prior_probability),
        name = 'pyramid_classification',
        **options
    )(outputs)
    
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
        
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)
    
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    
    


# In[14]:


# Testing the default classification model

#output: batch X height X width X kernel_size


test_classification_model = default_classification_model(10, 3)
test_classification_model.summary()


# In[15]:


# implement regression model


"""
For each anchor it regresses upon num_values. This means that in our dataset we can regress it with let say 6 values also
Also the values returned are the relative offsets between anchors and ground truths. 
"""

def default_regression_model(num_values, num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    options = {
        'kernel_size' : 3,
        'strides' : 1,
        'padding' : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer' : 'zeros'
    }
    
    if keras.backend.image_data_format() == 'channels_first':
        inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    
    
    outputs = inputs
    
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
        activation='relu',
        name='pyramid_regression_{}'.format(i),
        **options
        )(outputs)
        
    outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
    
    if keras.backend.image_data_format() == 'channels_first':
        outputs  = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)
    
    
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


# In[16]:


test_regression_model = default_regression_model(4, 10)
test_regression_model.summary()


# In[17]:


"""

On top of the normal convolutional model, a new pyramidal model is made
The upper layers are high on semantic meaning but low on resolution
The lower layers are high in resolution, but low on semantic meaning
This is called bottom up model (C1, C2, C3, C4, C5 ......)

A new top-down model is made by taking the topmost layer in bottom-up 
model as input and doing upsampling using KNN. Skip connections are used
between similar resolution images. This produces P1, P2, P3......
"""


"""
RetinaNet uses feature pyramid levelsP3toP7, where P3 to P5 arecomputed
from the output of the corresponding ResNet residual stage (C3throughC5)
using top-down and lateral connections just as in [20],P6isobtained via a
3×3 stride-2 conv onC5, andP7is computed by apply-ing ReLU followed by a
3×3 stride-2 conv onP6.  This differs slightlyfrom [20]: (1) we don’t use
the high-resolution pyramid levelP2for com-putational reasons, (2)P6is
computed by strided convolution instead ofdownsampling, and (3) we include
P7to improve large object detection.These minor modifications improve
speed while maintaining accuracy
"""

def _create_pyramid_features(C3, C4, C5, feature_size=256):
    
    #Upsample C5 to get P5
    
    # On C5 we apply 1X1 layer to adjust number of layers
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    
    #Upsample it to make it same as C4
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)
    
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])
    P4 = keras.layers.Conv2D(feature_size, kernel_size =3, strides=1, padding='same', name='P4')(P4)
    
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)
    
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)
    
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)
    
    return [P3, P4, P5, P6, P7]


# In[18]:


def default_submodels(num_classes, num_anchors):
    
    return [('regression', default_regression_model(4, num_anchors)),
           ('classification', default_classification_model(num_classes, num_anchors))]


# In[19]:


default_submodels(4, 10)


# In[20]:


"""
Takes a model as input (like classification) and applies to all the 
layers of the feature pyramid network P2, P3, ......
It returns the concatenation of all the layers thus obtained
"""

def build_model_pyramid(name, model, features):
    
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])
    


# In[21]:


"""
n is basically name of the submodel: regression or classification
m is basically the submodel
features are pyramid layers
"""

def build_pyramid(models, features):
    
    return [build_model_pyramid(n, m, features) for n, m in models]


# In[22]:


def build_anchors(anchor_parameters, features):
    
    anchors = [
        Anchors(
        size=anchor_parameters.sizes[i],
        stride=anchor_parameters.strides[i],
        ratios=anchor_parameters.ratios,
        scales=anchor_parameters.scales,
        name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]
    
    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


# In[23]:


def retinanet(
    inputs, 
    backbone_layers,
    num_classes,
    num_anchors = None,
    create_pyramid_features = _create_pyramid_features,
    submodels = None,
    name = 'retinanet'):
    
    
    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()
    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors)
        
    C3, C4, C5 = backbone_layers
    
    features = create_pyramid_features(C3, C4, C5)
    
    pyramids = build_pyramid(submodels, features)
    
    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


# In[24]:


def assert_training_model(model):

  assert(all(output in model.output_names for output in ['regression', 'classification'])), f"Input isn not training model: {model.output_names}"


# In[25]:


#Since the regression results are deltas and not the bounding box themselves,
#apply the deltas again to the anchors to get the actual bounding boxes.

def bbox_transform_inv(boxes, deltas, mean=None, std=None):

  if mean is None:
    mean = [0, 0, 0, 0]

  if std is None:
    std = [0.2, 0.2, 0.2, 0.2]

  width  = boxes[:, :, 2] - boxes[:, :, 0]
  height = boxes[:, :, 3] - boxes[:, :, 1]

  x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
  y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
  x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
  y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

  pred_boxes = keras.backend.stack([x1, y1, x2, y2], axis=2)

  return pred_boxes


# In[26]:


#regression boxes
class RegressBoxes(keras.layers.Layer):

  def __init__(self, mean=None, std=None, *args, **kwargs):

    if mean is None:
      mean = np.array([0, 0, 0, 0])

    if std is None:
      std = np.array([0.2, 0.2, 0.2, 0.2])

    #if mean is provided as a list or tuple
    if isinstance(mean, (list, tuple)):
      mean = np.array(mean)

    elif not isinstance(mean, np.ndarray):
      raise ValueError(f"Mean should be np.ndarray, list or tuple. Received {type(mean)}") 

    if isinstance(std, (list, tuple)):
      std = np.array(std)

    elif not isinstance(std, np.ndarray):
      raise ValueError(f"Standard deviation should be np.ndarray, list or tuple. Received {type(std)}")

    self.mean = mean
    self.std = std
    super(RegressBoxes, self).__init__(*args, **kwargs)

  #################To Do: backend.bbox_transform_inv###################################
  
  def call(self, inputs, **kwargs):
    anchors, regression = inputs
    return bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

  def compute_output_shape(self, input_shape):
    return input_shape[0]

  def get_config(self):
    config = super(RegressBoxes, self).get_config()
    config.update({
        'mean': self.mean.tolist(),
        'std' : self.std.tolist(),
    })  

    return config


# In[27]:


class ClipBoxes(keras.layers.Layer):

  def call(self, inputs, **kwargs):
    image, boxes = inputs
    shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())


   
    if keras.backend.image_data_format() == 'channels_first':
      _, _, height, width = tf.unstack(shape, axis = 0)

    else:
      _, height, width, _ = tf.unstack(shape, axis = 0)

    x1, y1, x2, y2  = tf.unstack(boxes, axis=-1)
    x1 = tf.clip_by_value(x1, 0, width - 1)
    y1 = tf.clip_by_value(y1, 0, height - 1)
    x2 = tf.clip_by_value(x2, 0, width - 1)
    y2 = tf.clip_by_value(y2, 0, height - 1)
    
    return keras.backend.stack([x1, y1, x2, y2], axis=2)

  def compute_output_shape(self, input_shape):
    return input_shape[1]


# In[28]:


def filter_detections(
    boxes,
    classification,
    other = [],
    class_specific_filter = True,
    nms = True,
    score_threshold = 0.05,
    max_detections = 300,
    nms_threshold = 0.5
):

  def _filter_detections(scores, labels):


    indices = tf.where(keras.backend.greater(scores, score_threshold))

    if nms:

      filtered_boxes = tf.gather_nd(boxes, indices)
      filtered_scores = keras.backend.gather(scores, indices)[:, 0]


      nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)

      indices = keras.backend.gather(indices, nms_indices)
      labels = tf.gather_nd(labels, indices)
      indices = keras.backend.stack([indices[:, 0], labels], axis=1)
      return indices
  

  if class_specific_filter:
    all_indices = []

    for c in range(int(classification.shape[1])):
      scores = classification[:, c]
      labels = c * tf.ones((keras.backend.shape(scores)[0], ), dtype='int64')
      all_indices.append(_filter_detections(scores, labels))
  
    indices = keras.backend.concatenate(all_indices, axis=0)

  else:
    scores = keras.backend.max(classification, axis = 1)
    labels = keras.backend.argmax(classification, axis = 1)
    indices = _filter_detections(scores, labels)

  scores = tf.gather_nd(classification, indices)
  labels = indices[:, 1]
  scores, top_indices = tf.nn.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

  indices = keras.backend.gather(indices[:, 0], top_indices)
  boxes = keras.backend.gather(boxes, indices)
  labels = keras.backend.gather(labels, top_indices)
  other_ =  [keras.backend.gather(o, indices) for o in other]


  pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
  boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
  scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
  labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
  labels = tf.cast(labels, 'int32')
  other_ = [tf.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]


  boxes.set_shape([max_detections, 4])
  scores.set_shape([max_detections])
  labels.set_shape([max_detections])

  for o, s in zip(other_, [list(keras.backend.int_shape(o)) for o in other]):
    o.set_shape([max_detections] + s[1:])

  return [boxes, scores, labels] + other_




# In[29]:


class FilterDetections(keras.layers.Layer):
   

  def __init__(
      self,
      nms                   = True,
      class_specific_filter = True,
      nms_threshold         = 0.5,
      score_threshold       = 0.05,
      max_detections        = 300,
      parallel_iterations   = 32,
      **kwargs
  ):
      
    self.nms                   = nms
    self.class_specific_filter = class_specific_filter
    self.nms_threshold         = nms_threshold
    self.score_threshold       = score_threshold
    self.max_detections        = max_detections
    self.parallel_iterations   = parallel_iterations
    super(FilterDetections, self).__init__(**kwargs)

  def call(self, inputs, **kwargs):
    
    boxes          = inputs[0]
    classification = inputs[1]
    other          = inputs[2:]

      # wrap nms with our parameters
    def _filter_detections(args):
      boxes          = args[0]
      classification = args[1]
      other          = args[2]

      return filter_detections(
          boxes,
          classification,
          other,
          nms                   = self.nms,
          class_specific_filter = self.class_specific_filter,
          score_threshold       = self.score_threshold,
          max_detections        = self.max_detections,
          nms_threshold         = self.nms_threshold,
      )

    # call filter_detections on each batch
    outputs = tf.map_fn(
        _filter_detections,
        elems=[boxes, classification, other],
        dtype=[keras.backend.floatx(), keras.backend.floatx(), 'int32'] + [o.dtype for o in other],
        parallel_iterations=self.parallel_iterations
    )

    return outputs

  def compute_output_shape(self, input_shape):
      
    return [
        (input_shape[0][0], self.max_detections, 4),
        (input_shape[1][0], self.max_detections),
        (input_shape[1][0], self.max_detections),
    ] + [
        tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(2, len(input_shape))
    ]

  def compute_mask(self, inputs, mask=None):
      
    return (len(inputs) + 1) * [None]

  def get_config(self):
      
    config = super(FilterDetections, self).get_config()
    config.update({
        'nms'                   : self.nms,
        'class_specific_filter' : self.class_specific_filter,
        'nms_threshold'         : self.nms_threshold,
        'score_threshold'       : self.score_threshold,
        'max_detections'        : self.max_detections,
        'parallel_iterations'   : self.parallel_iterations,
    })

    return config


# In[30]:


def retinanet_bbox(
    model = None,
    nms = True,
    class_specific_filter = True,
    name = 'retinanet-bbox',
    anchor_params = None,
    **kwargs
):

  if anchor_params is None:
    anchor_params = AnchorParameters.default

  if model is None:
    model = retinanet(num_anchors=anchor_params.num_anchors(), **kwargs)
  else:
    assert_training_model(model)

  features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
  anchors = build_anchors(anchor_params, features)

  regression = model.outputs[0]
  classification = model.outputs[1]

  other = model.outputs[2:]

  boxes  = RegressBoxes(name='boxes')([anchors, regression])
  boxes = ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

  detections = FilterDetections(
      nms = nms,
      class_specific_filter = class_specific_filter,
      name = 'filtered_detections'
  )([boxes, classification] + other)

  return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)


