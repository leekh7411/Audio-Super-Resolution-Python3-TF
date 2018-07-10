import os
import time
import numpy as np
import tensorflow as tf
import librosa
from dataset import DataSet
default_opt   = { 'alg': 'adam', 'lr': 1e-4, 'b1': 0.99, 'b2': 0.999,
                  'layers': 4, 'batch_size': 128 }

class Model(object):
    """Generic tensorflow model training code"""

    def __init__(self, from_ckpt=False, n_dim=None, r=2,opt_params=default_opt, log_prefix='./run'):

        # make session
        self.sess = tf.Session()
        
        # save params
        self.opt_params = opt_params
        self.layers     = opt_params['layers']

        if from_ckpt: 
            pass # we will instead load the graph from a checkpoint
        else:
            # create input vars
            X = tf.placeholder(tf.float32, shape=(None,  8192, 1), name='X')
            Y = tf.placeholder(tf.float32, shape=(None, 16384, 1), name='Y')
            alpha = tf.placeholder(tf.float32, shape=(), name='alpha') # weight multiplier

            # save inputs
            self.inputs = (X, Y, alpha)
            tf.add_to_collection('inputs', X)
            tf.add_to_collection('inputs', Y)
            tf.add_to_collection('inputs', alpha)

            # create model outputs
            self.predictions = self.create_model(n_dim, r)
            tf.add_to_collection('preds', self.predictions)

            # init the model
            init = tf.global_variables_initializer()
            self.sess.run(init)

            # create training updates
            self.train_op = self.create_train_op(X, Y, alpha)
            tf.add_to_collection('train_op', self.train_op)

        # logging
        lr_str = '.' + 'lr%f' % opt_params['lr']
        g_str  = '.g%d' % self.layers
        b_str  = '.b%d' % int(opt_params['batch_size'])

        self.logdir = log_prefix + lr_str + '.%d' % r + g_str + b_str
        self.checkpoint_root = os.path.join(self.logdir, 'model.ckpt')
    
    
    def get_params(self):
        return [ v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                 if 'soundnet' not in v.name ]
       
        
    def create_model(self, n_dim, r):
        raise NotImplementedError() # The inherited must be defined
        
        
    def create_optimzier(self, opt_params):
        if opt_params['alg'] == 'adam':
            lr, b1, b2 = opt_params['lr'], opt_params['b1'], opt_params['b2']
            optimizer = tf.train.AdamOptimizer(lr, b1, b2)
        else:
            raise ValueError('Invalid optimizer: ' + opt_params['alg'])

        return optimizer
        
        
    def create_gradients(self, loss, params):
        '''
        
        compute_gradients(
            loss,
            var_list=None,
            gate_gradients=GATE_OP,
            aggregation_method=None,
            colocate_gradients_with_ops=False,
            grad_loss=None
        )
        Compute gradients of loss for the variables in var_list.

        This is the first part of minimize(). 
        It returns a list of (gradient, variable) pairs where "gradient" is the gradient for "variable".
        Note that "gradient" can be a Tensor, an IndexedSlices, 
        or None if there is no gradient for the given variable.
        
        '''
        gv = self.optimizer.compute_gradients(loss, params) # return 'gradient' and 'variable'
        g, v = zip(*gv)
        return g
    
    
    def create_objective(self, X, Y, opt_params):
        # load model output and true output
        P = self.predictions

        # compute l2 loss
        sqrt_l2_loss = tf.sqrt(tf.reduce_mean((P-Y)**2 + 1e-6, axis=[1,2]))
        sqrn_l2_norm = tf.sqrt(tf.reduce_mean(Y**2, axis=[1,2]))
        snr = 20 * tf.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / tf.log(10.)

        avg_sqrt_l2_loss = tf.reduce_mean(sqrt_l2_loss, axis=0)
        avg_snr = tf.reduce_mean(snr, axis=0)

        # track losses
        tf.summary.scalar('l2_loss', avg_sqrt_l2_loss)
        tf.summary.scalar('snr', avg_snr)

        # save losses into collection
        tf.add_to_collection('losses', avg_sqrt_l2_loss)
        tf.add_to_collection('losses', avg_snr)

        return avg_sqrt_l2_loss
    
    
    def create_train_op(self, X, Y, alpha):
        # load params
        opt_params = self.opt_params
        print('creating train_op with params:', opt_params)

        # create loss
        self.loss = self.create_objective(X, Y, opt_params)

        # create params - get trainable variables
        params = self.get_params()

        # create optimizer
        self.optimizer = self.create_optimzier(opt_params)

        # create gradients 
        grads = self.create_gradients(self.loss, params)

        # create training op
        with tf.name_scope('optimizer'):
            # ref - https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
            train_op = self.create_updates(params, grads, alpha, opt_params)

        # initialize the optimizer variabLes
        optimizer_vars = []
        for v in tf.global_variables():
            if 'optimizer/' in v.name:
                optimizer_vars.append(v)
               
        init = tf.variables_initializer(optimizer_vars)
        self.sess.run(init)

        return train_op
    
    
    def create_updates(self, params, grads, alpha, opt_params):
        # create a variable to track the global step.
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # update grads
        grads = [alpha*g for g in grads]

        # use the optimizer to apply the gradients that minimize the loss
        gv = zip(grads, params)
        '''
        apply_gradients(
            grads_and_vars,
            global_step=None,
            name=None
        )
        Apply gradients to variables.

        This is the second part of minimize(). It returns an Operation that applies gradients.

        Args:
        grads_and_vars: List of (gradient, variable) pairs as returned by compute_gradients().
        global_step: Optional Variable to increment by one after the variables have been updated.
        name: Optional name for the returned operation. Default to the name passed to the Optimizer constructor.
        Returns:
        An Operation that applies the specified gradients. If global_step was not None, that operation also increments global_step.

        Raises:
        TypeError: If grads_and_vars is malformed.
        ValueError: If none of the variables have gradients.
        RuntimeError: If you should use _distributed_apply() instead.
        '''
        train_op = self.optimizer.apply_gradients(gv, global_step=self.global_step)

        return train_op
    
#########################################################################################################################################################
#########################################################################################################################################################

    def load(self, ckpt):
        # get checkpoint name
        # ref - https://www.tensorflow.org/api_docs/python/tf/train/latest_checkpoint
        if os.path.isdir(ckpt): checkpoint = tf.train.latest_checkpoint(ckpt)
        else: checkpoint = ckpt
        meta = checkpoint + '.meta'
        print(checkpoint)

        # load graph
        self.saver = tf.train.import_meta_graph(meta)
        g = tf.get_default_graph()

        # load weights
        self.saver.restore(self.sess, checkpoint)

        # get graph tensors
        X, Y, alpha = tf.get_collection('inputs')

        # save tensors as instance variables
        self.inputs = X, Y, alpha
        self.predictions = tf.get_collection('preds')[0]

        # load existing loss, or erase it, if creating new one
        g.clear_collection('losses')

        # create a new training op
        self.train_op = self.create_train_op(X, Y, alpha)
        g.clear_collection('train_op')
        tf.add_to_collection('train_op', self.train_op)
    
    
    def load_batch(self, batch, alpha=1, train=True):
        X_in, Y_in, alpha_in = self.inputs
        X, Y = batch

        if Y is not None:
            feed_dict = {X_in : X, Y_in : Y, alpha_in : alpha}
        else:
            feed_dict = {X_in : X, alpha_in : alpha}

        '''# this is ugly, but only way I found to get this var after model reload
        g = tf.get_default_graph()
        
        k_tensors = []
        for n in g.as_graph_def().node:
            if 'keras_learning_phase' in n.name and 'input' not in n.name:
                print('tf.default_graph.node:',n.name)
                k_tensors.append(n)
                
        #k_tensors = [n for n in g.as_graph_def().node if 'keras_learning_phase' in n.name]
        
        # ?????????????????????????/
        #assert len(k_tensors) <= 1
        assert len(k_tensors) <= 1
        
        if k_tensors: 
            k_learning_phase = g.get_tensor_by_name(k_tensors[0].name + ':0')
            feed_dict[k_learning_phase] = train'''

        return feed_dict
    
    def train(self, feed_dict):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss
    
    def fit(self, X_train, Y_train, X_val, Y_val, n_epoch=10):
        # initialize log directory                  
        if tf.gfile.Exists(self.logdir): tf.gfile.DeleteRecursively(self.logdir)
        tf.gfile.MakeDirs(self.logdir)

        # load some training params
        n_batch = self.opt_params['batch_size']

        # create saver
        self.saver = tf.train.Saver()

        # summarization
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

        # load data into DataSet
        train_data = DataSet(X_train, Y_train)
        val_data   = DataSet(X_val, Y_val)

        # train the model
        start_time = time.time()
        step, epoch = 0, train_data.epochs_completed
        
        print('start training epoch (n:%d)'%n_epoch)
        print('num-of-batch:',n_batch)
        
        while train_data.epochs_completed < n_epoch:

            step += 1
            # load the batch
            # alpha = min((n_epoch - train_data.epochs_completed) / 200, 1.)
            # alpha = 1.0 if epoch < 100 else 0.1
            alpha = 1.0
            
            print('get next batch from train data...')
            batch = train_data.next_batch(n_batch)
            print('...done')
            
            print('load batch and get feed-dict...')
            feed_dict = self.load_batch(batch, alpha)
            print('..done')
            
            # take training step
            print('train sequence start...')
            tr_objective = self.train(feed_dict)
            print('...done')
            
            tr_obj_snr = 20 * np.log10(1. / np.sqrt(tr_objective) + 1e-8)
            
            if step % 50 == 0:
                print(step, tr_objective, tr_obj_snr)

            # log results at the end of each epoch
            if train_data.epochs_completed > epoch:
                print('epoch-complete!')
                epoch = train_data.epochs_completed
                end_time = time.time()
                
                print('eval-err start...')
                tr_l2_loss, tr_l2_snr = self.eval_err(X_train, Y_train, n_batch=n_batch)
                va_l2_loss, va_l2_snr = self.eval_err(X_val, Y_val, n_batch=n_batch)
                print('...done!')                
                
                print("Epoch {} of {} took {:.3f}s ({} minibatches)".format(
                  epoch, n_epoch, end_time - start_time, len(X_train) // n_batch))
                print("  training l2_loss/segsnr:\t\t{:.6f}\t{:.6f}".format(
                  tr_l2_loss, tr_l2_snr))
                print("  validation l2_loss/segsnr:\t\t{:.6f}\t{:.6f}".format(
                  va_l2_loss, va_l2_snr))

                # compute summaries for overall loss
                objectives_summary = tf.Summary()
                objectives_summary.value.add(tag='tr_l2_loss', simple_value=tr_l2_loss)
                objectives_summary.value.add(tag='tr_l2_snr' , simple_value=tr_l2_snr)
                objectives_summary.value.add(tag='va_l2_snr' , simple_value=va_l2_loss)

                # compute summaries for all other metrics
                summary_str = self.sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.add_summary(objectives_summary, step)

                # write summaries and checkpoints
                summary_writer.flush()
                self.saver.save(self.sess, self.checkpoint_root, global_step=step)

                # restart clock
                start_time = time.time()

                
    def eval_err(self, X, Y, n_batch=128):
        batch_iterator = iterate_minibatches(X, Y, n_batch, shuffle=True)
        l2_loss_op, l2_snr_op = tf.get_collection('losses')

        l2_loss, snr = 0, 0
        tot_l2_loss, tot_snr = 0, 0
        for bn, batch in enumerate(batch_iterator):
            feed_dict = self.load_batch(batch, train=False)
            l2_loss, l2_snr = self.sess.run([l2_loss_op, l2_snr_op], feed_dict=feed_dict)
            tot_l2_loss += l2_loss
            tot_snr += l2_snr
            
        return tot_l2_loss / (bn+1), tot_snr / (bn+1)

    def predict(self, X):
        raise NotImplementedError()

# ----------------------------------------------------------------------------
# helpers

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    