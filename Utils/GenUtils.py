from random import shuffle

def shuffle_list(*lists):
    l = list(zip(*lists))
    shuffle(l)
    return map(list, zip(*l)) # Input lists and returns lists

def mphelper(args):
    arg = [*args]
    return arg[0](*arg[1:]) #arg[0] is the function to be applied onto the following arguments

########################################################################################################################
if __name__=='__main__':
    l1,l2,l3 = list(range(10)), list(range(10,20)),list(range(20,30))
    res1,res2,res3 = shuffle_list(l1,l2,l3)
    print(res1,res2,res3)
