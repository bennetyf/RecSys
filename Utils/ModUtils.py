import tensorflow as tf

########################################################################################################################
# General Utility Functions for a Typical Model

def train(model, restore=False, save=False, restore_datafile=None, save_datafile=None):
    if restore: # Restore the model from checkpoint
        restore_model(model, restore_datafile, verbose=True)
    else:
        model.session.run(tf.global_variables_initializer())

    if not save: # Do not save the model
        model.eval_one_epoch(-1)
        for i in range(model.epochs):
            model.train_one_epoch(i)
            model.eval_one_epoch(i)

    else: # Save the model while training
        previous_metric1,_ = model.eval_one_epoch(-1)
        # previous_metric1 = 0
        for i in range(model.epochs):
            model.train_one_epoch(i)
            metric1,_ = model.eval_one_epoch(i)
            if metric1 > previous_metric1:
                previous_metric1 = metric1
                model.save_model(save_datafile, verbose=False)

    # Save the model
def save_model(model, datafile, verbose=False):
    saver = tf.train.Saver()
    path = saver.save(model.session, datafile)
    if verbose:
        print("Model Saved in Path: {0}".format(path))

# Restore the model
def restore_model(model, datafile, verbose=False):
    saver = tf.train.Saver()
    saver.restore(model.session, datafile)
    if verbose:
        print("Model Restored from Path: {0}".format(datafile))

# Evaluate the model
def evaluate(model, datafile):
    model.restore_model(datafile,True)
    model.eval_one_epoch(-1)
