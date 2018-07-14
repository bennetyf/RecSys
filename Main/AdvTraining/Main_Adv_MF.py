import sys,os
sys.path.append('/share/scratch/fengyuan/Projects/RecSys/')

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf


import Utils.RecEval as evl
import Utils.MatUtils as mtl
import Utils.GenUtils as gtl
import Utils.ModUtils as mod

from DeepNN.Adv_MF import AMF

date = '20180713'

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
             'flimtrust-full':  'Data/filmtrust-hr-full.mat',
             'flimtrust-345':   'Data/filmtrust-hr-345.mat',
             'flimtrust-45':    'Data/filmtrust-hr-45.mat',
             'flimtrust-5':     'Data/filmtrust-hr-5.mat',
             'ymov-full':       'Data/ymov-hr-full.mat',
             'ymov-345':        'Data/ymov-hr-345.mat',
             'ymov-45':         'Data/ymov-hr-45.mat',
             'ymov-5':          'Data/ymov-hr-5.mat',
             'ymus-full':       'Data/ymus-hr-full.mat',
             'ymus-345':        'Data/ymus-hr-345.mat',
             'ymus-45':         'Data/ymus-hr-45.mat',
             'ymus-5':          'Data/ymus-hr-5.mat',
}

dataset = 'ml1m-345'
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
                num_factors=50,
                reg=0.001,

                reg_adv=10,
                noise_type='grad',
                is_adv=True,
                eps=1,
                lr=0.05,

                is_prec=False,
                epochs=2000,
                batch_size=1024,
                T=500,
                verbose=True)

    model.prepare_data(original_matrix=original_matrix, train_matrix=train_matrix, test_matrix=test_matrix)

    model.build_model()

    # for i in range(16):
    #     model.eps = i
    # model.evaluate('SavedModel/AMF/ML-1M/BPR_HR_CE.ckpt')
    # model.evaluate('SavedModel/GAN_AE/Ciao/Weight_Noise/Pure_AE_Ciao_SQ_MIX_ALL.ckpt')
    # model.evaluate('SavedModel/GAN_AE/Douban/Weight_Noise/Pure_AE_Douban_SQ_MIX_ALL.ckpt')
    # model.evaluate('SavedModel/GAN_AE/Epinions/Weight_Noise/Pure_AE_Epinions_SQ_MIX_ALL.ckpt')
    # model.evaluate('SavedModel/GAN_AE/YahooMovie/Weight_Noise/Pure_AE_YahooMovie_SQ_MIX_ALL.ckpt')
    # model.evaluate('SavedModel/GAN_AE/YahooMusic/Weight_Noise/Pure_AE_YahooMusic_SQ_MIX_ALL.ckpt')
    # model.evaluate('SavedModel/GAN_AE/FilmTrust/Weight_Noise/Pure_AE_FilmTrust_SQ_MIX_ALL.ckpt')
    # model.evaluate('SavedModel/GAN_AE/ML-10M/Weight_Noise/Pure_AE_ML10M_SQ_MIX_ALL_0_01.ckpt')
    # model.evaluate('SavedModel/GAN_AE/BookCrossing/Weight_Noise/Pure_AE_BX_SQ_MIX_ALL_1_0.ckpt')

    if model.is_adv:
        ckpt_save_path = "Pretrain/%s/APR/embed_%d/%s/" % (dataset, model.num_factors, date) + 'checkpoint.ckpt'
        ckpt_restore_path = "Pretrain/%s/BPR/embed_%d/%s/" % (dataset, model.num_factors, date) + 'checkpoint.ckpt'
    else:
        ckpt_save_path = "Pretrain/%s/BPR/embed_%d/%s/" % (dataset, model.num_factors, date) + 'checkpoint.ckpt'
        ckpt_restore_path = None

    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path)
    if ckpt_restore_path and not os.path.exists(ckpt_restore_path):
        os.makedirs(ckpt_restore_path)

    model.train(restore=True, save=True,
                save_datafile= ckpt_save_path, restore_datafile=ckpt_restore_path)

                    # save_datafile = 'SavedModel/AMF/ML-1M/BPR_HR_CE.ckpt')
                    # save_datafile = 'SavedModel/AMF/Ciao/BPR_HR_CE.ckpt')

        # model.train(restore=False, save=True,
        #             save_datafile = 'SavedModel/AMF/ML-1M/BPR_HR_CE.ckpt')

                    # save_datafile='SavedModel/AMF/Ciao/AMF_ORG.ckpt')
                    #  save_datafile= 'SavedModel/AMF/Ciao/AMF_Precision.ckpt')

                    # save_datafile = 'SavedModel/AMF/Ciao/BPR_Precision_ORG.ckpt')
                    # save_datafile = 'SavedModel/AMF/FilmTrust/BPR_HR_CE_ORG.ckpt')