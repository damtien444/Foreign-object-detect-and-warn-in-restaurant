# import os
#
# import cv2
# import numpy as np
# from tensorflow import keras
# import tensorflow as tf
# import time
#
#
# def preprocess_image(img):
#     if img.shape[0] != 224 or img.shape[1] != 224:
#         img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
#     img = (img / 127.5)
#     img = img - 1
#     img = np.expand_dims(img, axis=0)
#     return img
#
#
# class Classify:
#     def __init__(self):
#         # file_path = os.path.abspath(os.path.dirname(__file__))
#         # path = os.path.join(file_path, '\models\graph\self_trained\\trained_2\\')
#         self.inf_model = keras.models.load_model("best.hdf5")
#
#         self.classes = ['sitting', 'standing']
#
#     def classify(self, img):
#         pred = self.inf_model.predict(preprocess_image(img))
#         result = self.classes[np.argmax(pred)]
#         print(result)
#         return result
#
# def load_graph(frozen_graph_filename):
#     with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
#         graph_def = tf.compat.v1.GraphDef()
#         graph_def.ParseFromString(f.read())
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def)
#     return graph
#
# def analyze_inputs_outputs(graph):
#     ops = graph.get_operations()
#     outputs_set = set(ops)
#     inputs = []
#     for op in ops:
#         if len(op.inputs) == 0 and op.type != 'Const':
#             inputs.append(op)
#         else:
#             for input_tensor in op.inputs:
#                 if input_tensor.op in outputs_set:
#                     outputs_set.remove(input_tensor.op)
#     outputs = list(outputs_set)
#     return (inputs, outputs)
#
#
# import glob
#
# files = glob.glob("data/withoutBackGround/sitting/*.jpeg")
# img = cv2.imread(files[0])
# checkpoint = time.time()
# cla = Classify()
# next = time.time()
#
# print("Time to create session: "+str(next-checkpoint))
#
# checkpoint = time.time()
#
# cla.classify(img)
# next = time.time()
# print("Time to predict: "+str(next-checkpoint))

# all_ops = tf.get_default_graph().get_operations()
# print(all_ops)

# load_graph("saved_model.pb")
# all_placeholders = [placeholder for op in tf.get_default_graph().get_operations() if op.type=='Placeholder' for placeholder in op.values()]



# use this line to detect whether input and output node :<
# print(analyze_inputs_outputs(load_graph("saved_model.pb")))
#
# import tensorflow as tf
# from tensorflow import keras
# from keras import backend as K
# # This line must be executed before loading Keras models.
# K.set_learning_phase(0)
#
# models = keras.models.load_model('best_1.hdf5')
#
# def freeze_session(session, keep_var_names=None, output_names=None,clear_devices=True):
#     """
#     Freezes the state of a session into a pruned computation graph.
#
#     Creates a new computation graph where variable nodes are replaced by
#     constants taking their current value in the session. The new graph will be
#     pruned so subgraphs that are not necessary to compute the requested
#     outputs are removed.
#     @param session The TensorFlow session to be frozen.
#     @param keep_var_names A list of variable names that should not be frozen,
#                           or None to freeze all the variables in the graph.
#     @param output_names Names of the relevant graph outputs.
#     @param clear_devices Remove the device directives from the graph for better portability.
#     @return The frozen graph definition.
#     """
#     from tensorflow.python.framework.graph_util import convert_variables_to_constants
#     graph = session.graph
#     with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#         output_names = output_names or []
#         output_names += [v.op.name for v in tf.global_variables()]
#         # Graph -> GraphDef ProtoBuf
#         input_graph_def = graph.as_graph_def()
#         if clear_devices:
#             for node in input_graph_def.node:
#                 node.device = ""
#         frozen_graph = convert_variables_to_constants(session, input_graph_def,
#                                                       output_names, freeze_var_names)
#         return frozen_graph
#
# frozen_graph = freeze_session(K.get_session(),
#                               output_names=[out.op.name for out in models.outputs])
#
# tf.train.write_graph(frozen_graph, "", "withBackGround_model.pb", as_text=False)
import glob
import os

import cv2
import numpy as np
from keras_preprocessing.image import ImageDataGenerator


def image_generator():
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        # brightness_range=0.1,
        # vertical_flip=True,
        horizontal_flip=True)

    image_paths = glob.glob(os.path.join('./data/withBackGround/standing/', '*.jpeg'))
    print((len((image_paths))))
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (224, 224))
        images.append(image)
    datagen.fit(np.stack(images))
    count = 1
    for i in np.arange(len(images)):
        no_img = 0
        for x in datagen.flow(np.expand_dims(images[i], axis=0), batch_size=1):
            cv2.imwrite('./data/withBackGround/generated/standing/withBackGround%d.jpeg' % count, (x[0]).astype(np.int32))
            print((x[0]).astype(np.int32).shape)
            count += 1
            no_img += 1
            if no_img == 3:
                break

image_generator()