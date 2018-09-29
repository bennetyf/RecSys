import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf


import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl
import Utils.ModUtils as mod

from DeepNN.Adv_MF import AMF

date = '20180801'
filename = 'checkpoint.ckpt'
# resname = 'org_res.mat'
# resname = 'adv_res.mat'

path_dict = {'ml1m-full':   'Data/ml1m-hr-full.mat',
             'ml1m-345':    'Data/ml1m-hr-345.mat',
             'ml1m-45':     'Data/ml1m-hr-45.mat',
             'ml1m-5':      'Data/ml1m-hr-5.mat',
             'ciao-full':   'Data/ciao-hr-full.mat',
             'ciao-345':    'Data/ciao-hr-345.mat',
             'ciao-45':     'Data/ciao-hr-45.mat',
             'ciao-5':      'Data/ciao-hr-5.mat',
             'douban-full': 'Data/douban-hr-full.mat',
             'douban-345':  'Data/douban-hr-345.mat',
             'douban-45':   'Data/douban-hr-45.mat',
             'douban-5':    'Data/douban-hr-5.mat',
             'filmtrust-full':  'Data/filmtrust-hr-full.mat',
             'filmtrust-34':   'Data/filmtrust-hr-34.mat',
             'filmtrust-4':     'Data/filmtrust-hr-4.mat',
             'flixster-full':   'Data/flixster-hr-full.mat',
             'flixster-345':    'Data/flixster-hr-345.mat',
             'flixster-45':     'Data/flixster-hr-45.mat',
             'flixster-5':      'Data/flixster-hr-5.mat',
             'ymov-full':       'Data/ymov-hr-full.mat',
             'ymov-345':        'Data/ymov-hr-345.mat',
             'ymov-45':         'Data/ymov-hr-45.mat',
             'ymov-5':          'Data/ymov-hr-5.mat',
             'ymus-full':       'Data/ymus-hr-full.mat',
             'ymus-345':        'Data/ymus-hr-345.mat',
             'ymus-45':         'Data/ymus-hr-45.mat',
             'ymus-5':          'Data/ymus-hr-5.mat',
}

dataset = 'filmtrust-34'
path = path_dict[dataset]

print('Loading Data From {0}'.format(path))
data = gtl.load_mat_as_matrix(path, opt='coo')
original_matrix, train_matrix, test_matrix = data['original'], data['train'], data['test']
print('Users:{0}, Items:{1}, Ratings:{2}'.format(original_matrix.shape[0], original_matrix.shape[1], original_matrix.nnz))

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=24,
                                          inter_op_parallelism_threads=24,
                                          gpu_options=gpu_options)) as sess:
    model = AMF(sess,
                top_K=[5,10],
                num_factors=40,
                reg=0.015,

                reg_adv=100,
                noise_type='grad',
                is_adv=True,
                eps=0.1,
                lr=0.1,

                save_T=10,

                is_prec=False,
                epochs=500,
                batch_size=128,
                T=1000,
                verbose=True)

    model.prepare_data(original_matrix=original_matrix, train_matrix=train_matrix, test_matrix=test_matrix)
    # resname='testforfun.mat'

    if model.is_adv:
        ckpt_save_path = "Pretrain/%s/APR/embed_%d/reg_%s/%s/" % (dataset, model.num_factors, str(model.reg), date)
        ckpt_restore_path = "Pretrain/%s/BPR/embed_%d/reg_%s/%s/" % (dataset, model.num_factors, str(model.reg),date)
        res_save_path = "Result/%s/BPR/embed_%d/reg_%s/%s/" % (dataset, model.num_factors, str(model.reg), date)
    else:
        ckpt_save_path = "Pretrain/%s/BPR/embed_%d/reg_%s/%s/" % (dataset, model.num_factors, str(model.reg), date)
        ckpt_restore_path = ''
        res_save_path = "Result/%s/BPR/embed_%d/reg_%s/%s/" % (dataset, model.num_factors, str(model.reg), date)

    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path)
    if ckpt_restore_path and not os.path.exists(ckpt_restore_path):
        os.makedirs(ckpt_restore_path)
    if not os.path.exists(res_save_path):
        os.makedirs(res_save_path)

    # resname = 'robust_adv_res.mat'
    # for i in range(16):
    #     model.eps = i
    model.build_model()
        # model.evaluate(ckpt_restore_path + filename)

    model.train(restore=True, save=False,
                save_datafile= ckpt_save_path + filename,
                restore_datafile=ckpt_restore_path + filename)

    # gtl.list_to_mat(res_save_path + resname, HR5=model.metric1, NDCG5=model.metric2)