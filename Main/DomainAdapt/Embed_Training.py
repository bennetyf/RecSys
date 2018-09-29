# Training user embeddings using AutoRec

import sys, os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import tensorflow as tf

import GenUtils as gtl
from AutoRec_Model import AutoRec

# Set model saving and restoration paths
_dataset = 'ML1M'
_date = '20180921'
_filename = 'autorec.ckpt'
_model_chkpath = ''
_embedding_file_mat = 'Data/ML1M-ebd.mat'
_embedding_file_excel = 'Data/ML1M-ebd.xlsx'
_embedding_dim = 120

def trainAutoRec(nepoch, ebd_gen = False):
    global _model_chkpath
    global _dataset
    global _date
    global _filename

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        intra_op_parallelism_threads=16,
                                        inter_op_parallelism_threads=16,
                                        gpu_options=gpu_options)) as sess:

        model = AutoRec(sess,
                        num_factors=_embedding_dim,
                        reg=1.0,
                        model='user',
                        lr=0.001,
                        dropout=0.15,
                        epochs=nepoch,
                        batch_size=256,
                        T=2000,
                        verbose=True)

        model.prepare_data(original_matrix=original_matrix, train_matrix=train_matrix, test_matrix=test_matrix)
        model.build_model()

        # Trained embedding saving path
        _model_chkpath = "Embed/%s/AutoRec/embed_%d/reg_%s/%s/" % (_dataset, model.num_factors, str(model.reg), _date)

        if not os.path.exists(_model_chkpath):
            os.makedirs(_model_chkpath)

        model.train(restore=False, save=True, datafile=_model_chkpath + _filename)

        if ebd_gen:
            model.gen_embedding(datafile=_embedding_file_mat)

def genEmbed(restore_path):
    print(restore_path)
    if not os.path.exists(restore_path):
        print("[genEmbed]: Restore File Does Not Exist!")
        return

    restore_file = restore_path + _filename

    gpu_options = tf.GPUOptions(allow_growth = True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement = True,
                                          intra_op_parallelism_threads = 16,
                                          inter_op_parallelism_threads = 16,
                                          gpu_options = gpu_options)) as sess:

        model = AutoRec(sess, num_factors = _embedding_dim)
        model.prepare_data(original_matrix = original_matrix, train_matrix = train_matrix, test_matrix = test_matrix)
        model.build_model()

        model.restore_model(restore_file, verbose = True)
        model.eval_one_epoch(-1)
        embedding_arr = model.session.run(model.embedding_out,
                                          feed_dict = { model.ratings:      train_matrix.toarray(),
                                                        model.istraining:   False,
                                                        model.dropout_rate: 0.0})

        # gtl.array_to_excel(_embedding_file_excel, embedding = embedding_arr)
        gtl.array_to_mat(_embedding_file_mat, embedding = embedding_arr)

########################################################################################################################

# original_matrix \
#     = mtl.load_original_matrix(datafile='/share/scratch/fengyuan/Projects/RecSys/Data/ML/ml-1m/ratings.dat', header=['uid', 'iid', 'ratings', 'time'], sep='::')

# train_matrix, test_matrix = mtl.matrix_split(original_matrix, opt='prediction', mode='user', test_size=0.1, random_state=10)

# gtl.matrix_to_mat('Data/UAutoRec_ML1M_90_Data.mat', opt='coo', original=original_matrix,train=train_matrix,test=test_matrix)

data = gtl.load_mat_as_matrix('Data/UAutoRec_ML1M_90_Data.mat',opt='coo')
original_matrix, train_matrix, test_matrix = data['original'], data['train'], data['test']

########################################################################################################################

trainAutoRec(nepoch=200)
genEmbed(restore_path = _model_chkpath)