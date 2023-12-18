""" 
Following code has been ran within the Docker Container provided by DeepCRISPR. https://github.com/bm2-lab/DeepCRISPR
The code below runs pre-trained model and calculate gradient to compute saliency map. 

python == 3.6
tensorflow == 1.3.0
sonnet == 1.9

"""

import numpy as np
import tensorflow as tf
import deepcrispr as dc
from tensorflow.contrib import slim


file_path = 'examples/eg_reg_on_target.repisgt'
input_data = dc.Episgt(file_path, num_epi_features=4, with_y=True)
x, y = input_data.get_dataset()
x = np.expand_dims(x, axis=2)  # shape(x) = [100, 8, 1, 23]
sess = tf.InteractiveSession()
on_target_model_dir = 'trained_models/ontar_pt_cnn_reg'
dcmodel = dc.DCModelOntar(sess, on_target_model_dir, is_reg=True, seq_feature_only=False)
predicted_on_target = dcmodel.ontar_predict(x)


np.save("y_input.npy", y)
np.save("predicted_on_target.npy", predicted_on_target)

### PART TWO 

file_path = 'examples/eg_reg_on_target.repisgt'
input_data = dc.Episgt(file_path, num_epi_features=4, with_y=True)
x, y = input_data.get_dataset()
x = np.expand_dims(x, axis=2)  # shape(x) = [100, 8, 1, 23]



max_min_x = np.load('merged_array.npy')
max_min_y = [[0.88665013] , [0.002470402]]
x = np.concatenate((x, max_min_x))
y = np.append(y, max_min_y)

# Define the Saliency Map Calculation
def compute_saliency_map(model_output, input_tensor):
  saliency_map = tf.gradients(model_output, input_tensor)[0]
  saliency_map /= (tf.reduce_max(tf.abs(saliency_map), axis = (1,2), keepdims = True) + 1e-8)
  return saliency_map

# # Session and Saver
sess = tf.InteractiveSession()
on_target_model_dir = 'trained_models/ontar_pt_cnn_reg'
dcmodel = dc.DCModelOntar(sess, on_target_model_dir, is_reg=True, seq_feature_only=False)

model_output = dcmodel.pred_ontar

input_tensor = dcmodel.inputs_sg

# Compute the Saliency Map
saliency_map_tensor = compute_saliency_map(model_output, input_tensor)
x_transposed = np.transpose(x, (0, 2, 3, 1))

saliency_values = sess.run(saliency_map_tensor, feed_dict={input_tensor: x_transposed})

np.save("saliency_values_reg_norm.npy", saliency_values)
