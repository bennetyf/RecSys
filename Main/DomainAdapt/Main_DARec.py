import sys, os
from DARec import DARec
import tensorflow as tf

# import RecEval as evl
# import MatUtils as mtl
import GenUtils as gtl
# import ModUtils as mod

sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Loading Data from Hard Drive
path_dict = {'amazon-1':
             ('Data/amazon-1-sc.mat', 'Data/amazon-1-tg.mat', 'Data/amazon-1-sc-ebd.mat', 'Data/amazon-1-tg-ebd.mat')}

dataset = 'amazon-1'
path_sc, path_tg, path_sc_ebd, path_tg_ebd = path_dict[dataset]

print('Loading Data From Source {0} and Target {0}'.format(path_sc, path_tg))
data_sc, data_tg = gtl.load_mat_as_matrix(path_sc, opt='coo'), gtl.load_mat_as_matrix(path_tg, opt='coo')
original_matrix_sc, train_matrix_sc, test_matrix_sc = data_sc['original'], data_sc['train'], data_sc['test']
original_matrix_tg, train_matrix_tg, test_matrix_tg = data_tg['original'], data_tg['train'], data_tg['test']
sc_ebd, tg_ebd = gtl.load_mat_as_array(path_sc_ebd), gtl.load_mat_as_array(path_tg_ebd)
embedding_arr_sc, embedding_arr_tg = sc_ebd['embedding'], tg_ebd['embedding']

# The Main Program
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                    intra_op_parallelism_threads=24,
                                    inter_op_parallelism_threads=24,
                                    gpu_options=gpu_options)) as sess:
    model = DARec(
        sess,
        top_K=[5, 10],
        input_dim=200,

        pred_dim=100,
        pred_lambda=1.0,
        pred_reg=0.0,

        cls_layers=[100, 2],
        cls_reg=0.0,

        pred_cls_lambda=1.0,
        grl_lambda=1.0,
        drop_out_rate=0.1,

        lr=0.001,
        epochs=1000,
        batch_size=256,
        T=10 ** 3,
        verbose=False
    )

    model.prepare_data(original_matrix_sc, train_matrix_sc, test_matrix_sc,
                        original_matrix_tg, train_matrix_tg, test_matrix_tg,
                        embedding_arr_sc, embedding_arr_tg)

    model.build_model()
    model.train()
