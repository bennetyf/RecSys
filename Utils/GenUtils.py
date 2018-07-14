import os
from random import shuffle
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import inspect

########################################### System Operations ##########################################################

# Convert bytes into other units
def convert_from_bytes(num, unit='MB'):
    divider = {
        "B":    1.0,
        "KB":   1024.0,
        "MB":   1024.0**2,
        "GB":   1024.0**3,
        "TB":   1024.0**4
    }.get(unit, 1024.0**2)
    return num/divider

# Return the size of a specific file on the disk
def file_size(file_path):
    if os.path.isfile(file_path):
        return convert_from_bytes(os.stat(file_path).st_size)

# Merge a list of files
def merge_files(target_file, filelist):
    # Always write into a new file
    if os.path.exists(target_file):
        os.remove(target_file)
    fout = open(target_file, "a")
    # now the rest:
    for i in range(len(filelist)):
        f = open(filelist[i])
        for line in f:
            fout.write(line)
        f.close()  # not really needed
    fout.close()

# Delete a list of files
def delete_files(filelist):
    for i in range(len(filelist)):
        os.remove(filelist[i])

########################################################################################################################

def shuffle_list(*lists):
    '''
    Shuffle a list of lists randomly
    :param lists:
    :return:
    '''
    l = list(zip(*lists))
    shuffle(l)
    return map(list, zip(*l)) # Input lists and returns lists

def mphelper(args):
    '''
    Helper function for multiprocess programming
    :param args:
    :return:
    '''
    arg = [*args]
    return arg[0](*arg[1:]) #arg[0] is the function to be applied onto the following arguments

########################################################################################################################

# Save one matrix into one csv file
def matrix_to_csv(datafile, sp_matrix, opt='coo'):
    if opt == 'all':
        np.savetxt(datafile,sp_matrix.todense(),delimiter=',')
    elif opt == 'coo':
        coo_mat = sp_matrix.tocoo()
        df = pd.DataFrame(np.column_stack((coo_mat.row, coo_mat.col, coo_mat.data)))
        df.to_csv(datafile, header=False, index=False)

# Save multiple matrices into one mat file
def matrix_to_mat(datafile, opt='all', **kwargs):
    res_dict = {}
    if opt == 'all':
        for name in kwargs:
            res_dict[name] = kwargs[name].todense()
    elif opt == 'coo':
        for name in kwargs:
            coo_mat = kwargs[name].tocoo()
            res_dict[name] = np.column_stack((coo_mat.row, coo_mat.col, coo_mat.data))
            # print(name)
    sio.savemat(datafile,res_dict)

# Save multiple matrices into one excel spreed sheet
def matrix_to_excel(datafile, opt='coo', **kwargs):
    with pd.ExcelWriter(datafile) as writer:
        if opt == 'all':
            for name in kwargs:
                df = pd.DataFrame(kwargs[name].todense())
                df.to_excel(writer, sheet_name=name, header=False, index=False)
        elif opt == 'coo':
            for name in kwargs:
                coo_mat = kwargs[name].tocoo()
                df = pd.DataFrame(np.column_stack((coo_mat.row, coo_mat.col, coo_mat.data)))
                df.to_excel(writer, sheet_name=name, header=False, index=False)

# Load matlab format data into sparse matrix format
def load_mat_as_matrix(datafile, opt='all'):
    raw = sio.loadmat(datafile)

    if opt == 'all':
        res_dict = {}
        for key in raw:
            if isinstance(raw[key],np.ndarray):
                mat = sp.csr_matrix(raw[key])
                mat.eliminate_zeros()
                res_dict[key] = mat
        return res_dict

    elif opt == 'coo':
        return {key: sp.csr_matrix((raw[key][:,2],(raw[key][:,0],raw[key][:,1]))).tolil()
                for key in raw if isinstance(raw[key],np.ndarray)}

########################################################################################################################

# Save a numpy array into a csv file
def array_to_csv(datafile, array):
    np.savetxt(datafile, array, delimiter=',')

# Save multiple arrays into one matlab mat file
def array_to_mat(datafile, **kwargs):
    sio.savemat(datafile, kwargs)

# Save multiple arrays into one excel file
def array_to_excel(datafile, **kwargs):
    with pd.ExcelWriter(datafile) as writer:
        for name in kwargs:
            df = pd.DataFrame(kwargs[name])
            df.to_excel(writer, sheet_name=name, header=False, index=False)

########################################################################################################################

# Store a list of lists with the same lengths into a csv file
def list_to_csv(datafile, *lists):
    l = list(map(list, zip(*lists)))
    df = pd.DataFrame(l)
    df.to_csv(datafile, header=False, index=False)

# Store a list of lists with the same lengths into an excel file
def list_to_excel(datafile, **kwargs):
    with pd.ExcelWriter(datafile) as writer:
        for name in kwargs:
            df = pd.DataFrame(np.column_stack(tuple(kwargs[name])))
            df.to_excel(writer, sheet_name=name, header=False, index=False)

# Store a set of list of lists with the same lengths into a single mat format file
def list_to_mat(datafile, **kwargs):
    res_dict = {}
    for name in kwargs:
        res_dict[name] = np.column_stack(tuple(kwargs[name]))
    sio.savemat(datafile, res_dict)

########################################################################################################################
# Set the numpy random seed
def set_random_seed(seed = None):
    np.random.seed(seed=seed)

# Print the parameters of current function frame
def print_paras(frame):
    args, _, _, values = inspect.getargvalues(frame)
    print('=' * 55)
    print('function name "%s"' % inspect.getframeinfo(frame)[2])
    for i in args:
        print("    %s = %s" % (i, values[i]))
    print('=' * 55)

########################################################################################################################
# if __name__=='__main__':
#     l1,l2,l3 = list(range(10)), list(range(10,20)),list(range(20,30))
#     res1,res2,res3 = shuffle_list(l1,l2,l3)
#     print(res1,res2,res3)
