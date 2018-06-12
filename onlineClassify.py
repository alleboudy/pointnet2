'''
    Receives classification requests and reply with top class
'''
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import modelnet_dataset
import modelnet_h5_dataset
from flask import Flask, jsonify, render_template, request





# import pointnet_colored as MODEL_onlycolored
# import onevsall as MODEL_onlyPoints
# import pointnet_colored as MODEL_onlynormals
# import pointnet_coloredNormals as MODEL_normalsandcolors





parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name. [default: pointnet2_cls_ssg]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 16]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
parser.add_argument('--pipeline_code', type=int, default=2, help='which pipeline to use, 0 = colored(+points) 1 = colored+normals(+points) 2 = only points 3 = only normals(+points) [default: 2]')
FLAGS = parser.parse_args()

pipelineCode = FLAGS.pipeline_code
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(ROOT_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

HOSTNAME = socket.gethostname()

# Shapenet official train/test split
if FLAGS.normal:
    assert(NUM_POINT<=10000)
    DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
else:
    assert(NUM_POINT<=2048)
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# def evaluate(num_votes):
#     is_training = False
     
#     with tf.device('/gpu:'+str(GPU_INDEX)):
#         pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
#         is_training_pl = tf.placeholder(tf.bool, shape=())

#         # simple model
#         pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
#         #MODEL.get_loss(pred, labels_pl, end_points)
#         #losses = tf.get_collection('losses')
#         #total_loss = tf.add_n(losses, name='total_loss')
        
#         # Add ops to save and restore all the variables.
#         saver = tf.train.Saver()
        
#     # Create a session
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     config.allow_soft_placement = True
#     config.log_device_placement = False
#     sess = tf.Session(config=config)

#     # Restore variables from disk.
#     saver.restore(sess, MODEL_PATH)
#     log_string("Model restored.")

#     ops = {'pointclouds_pl': pointclouds_pl,
#            'labels_pl': labels_pl,
#            'is_training_pl': is_training_pl,
#            'pred': pred
#            #,'loss': total_loss
#            }

#     eval_one_epoch(sess, ops, num_votes)
reverseDict={}
c=0
with open('shape_names.txt') as f:
    for line in f:
        reverseDict[c]=line.replace('\n','')
        c+=1;




testFile=''

is_training = False
pointclouds_pl=None
pointclouds_rgb_pl=None
labels_pl=None
is_training_pl = tf.placeholder(tf.bool, shape=())

if pipelineCode==2:
    pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
    pred, end_points = MODEL.get_model(pointclouds_pl,is_training_pl)
else:# pipelineCode ==0:
    pointclouds_pl,pointclouds_rgb_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
    pred, end_points = MODEL.get_model(pointclouds_pl, pointclouds_rgb_pl,is_training_pl)



#for when I add colors and normals...
# MODEL=None

# if pipelineCode ==0:
#     MODEL = MODEL_onlycolored
#     MODEL_PATH='logcolored/model.ckpt'
# elif pipelineCode ==1:
#     MODEL = MODEL_normalsandcolors
#     MODEL_PATH='log/model.ckpt'
# elif pipelineCode==2:
#     MODEL = MODEL_onlyPoints
#     MODEL_PATH = 'log/model.ckpt'
# elif pipelineCode==3:
#     MODEL = MODEL_onlycolored
#     MODEL_PATH='logdatanormalsnocolor/model.ckpt'







#loss = MODEL.get_loss(pred, labels_pl, end_points)
pred = tf.nn.softmax(pred)
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
    
# Create a session


sess = tf.Session()

# Restore variables from disk.
saver.restore(sess, MODEL_PATH)
#log_string("Model restored.")
ops={'pointclouds_pl': pointclouds_pl,
       'is_training_pl': is_training_pl,
       'pred': pred,
       }
if pipelineCode!=2:
    ops['pointclouds_rgb_pl']=pointclouds_rgb_pl



path2colorsavgSigma='colorsnormalstrain.h5trainAverageStdColor.txt'


from io import StringIO

class StringBuilder:
     _file_str = None

     def __init__(self):
         self._file_str = StringIO()

     def Append(self, str):
         self._file_str.write(str)

     def __str__(self):
         return self._file_str.getvalue()





def eval_one_epoch(testFile=testFile):
    is_training = False
    current_data=[]

    current_data,current_colors,current_normals = provider.load_ply_data(testFile,NUM_POINT,path2colorsavgSigma)
    #current_label = np.squeeze(current_label)
    if pipelineCode==1:
        current_colors = np.concatenate((current_colors,current_normals),axis=1)
    if pipelineCode==3:
        current_colors = current_normals
    
    current_data = np.asarray([current_data,np.zeros_like(current_data)])
    current_colors = np.asarray([current_colors,np.zeros_like(current_colors)])
    current_normals = np.asarray([current_normals,np.zeros_like(current_normals)])
    #print(current_data.shape)
            
    #file_size = current_data.shape[0]
    num_batches = 1
    #print(file_size)
      
    
    batch_pred_sum = np.zeros((current_data.shape[0], NUM_CLASSES)) # score for classes
    batch_pred_classes = np.zeros((current_data.shape[0], NUM_CLASSES)) # 0/1 for classes
    feed_dict = {ops['pointclouds_pl']: current_data,
                 
                 ops['is_training_pl']: is_training}
    if pipelineCode!=2:# not only points 
        feed_dict[ops['pointclouds_rgb_pl']]=current_colors
    pred_val = sess.run( ops['pred'],feed_dict=feed_dict)
    sb = StringBuilder()
    #if(len(onlyPlyfiles)==0):
    #    onlyPlyfiles.append(testFile)
    #for i in range(len(onlyPlyfiles)):
    sb.Append(unicode(str(np.max(pred_val[0]))))
    sb.Append(unicode(","))
    sb.Append(unicode(reverseDict[np.argmax(pred_val[0])]))
    #sb.Append('\n')
            #print(str(np.max(pred_val[i]))+","+reverseDict[np.argmax(pred_val[i])])

    return sb.__str__();




app = Flask(__name__)



@app.route('/', methods=['POST'])
def main():
    #input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    #testFile = request.args.get('testFile')
    #testFile = request.data
    with open('testFile.ply','w') as of:
        of.write(request.data)
    output = eval_one_epoch(testFile='testFile.ply')
    print(output)
    return output


if __name__=='__main__':
    with tf.Graph().as_default():
            app.run(host = '0.0.0.0',port=5070)
