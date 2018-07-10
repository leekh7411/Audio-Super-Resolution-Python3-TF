import numpy as np
import tensorflow as tf
from scipy import interpolate
from subpixel import SubPixel1D, SubPixel1D_v2
from dataset import DataSet
import os
import time
import librosa

default_opt   = { 'alg': 'adam', 'lr': 1e-4, 'b1': 0.99, 'b2': 0.999,
                  'layers': 4, 'batch_size': 128 }
class ASRNet(): 
# ----------------------------------------------------------------------------
# initialize sequence 

    # based AudioUNet
    def __init__(self, from_ckpt=False, n_dim=None, r=2,
               opt_params=default_opt, log_prefix='./run'):
        self.r = r
        # make session
        self.sess = tf.Session()
        
        # save params
        self.opt_params = opt_params
        self.layers     = opt_params['layers']

        if from_ckpt: 
            pass # we will instead load the graph from a checkpoint
        else:
            # create input vars
            X = tf.placeholder(tf.float32, shape=(None, None, 1), name='X')
            Y = tf.placeholder(tf.float32, shape=(None, None, 1), name='Y')
            #alpha = tf.placeholder(tf.float32, shape=(), name='alpha') # weight multiplier

            # save inputs
            self.inputs = (X, Y)
            tf.add_to_collection('inputs', X)
            tf.add_to_collection('inputs', Y)
            #tf.add_to_collection('inputs', alpha)

            # create model outputs
            self.predictions = self.create_model(n_dim, r)
            tf.add_to_collection('preds', self.predictions)

            # init the model
            # init = tf.global_variables_initializer()
            # init = tf.initialize_all_variables()
            # self.sess.run(init)

            # create training updates
            self.train_op = self.create_train_op(X, Y)
            tf.add_to_collection('train_op', self.train_op)

        # logging
        lr_str = '.' + 'lr%f' % opt_params['lr']
        g_str  = '.g%d' % self.layers
        b_str  = '.b%d' % int(opt_params['batch_size'])

        self.logdir = log_prefix + lr_str + '.%d' % r + g_str + b_str
        self.checkpoint_root = os.path.join(self.logdir, 'model.ckpt') 
           
    def create_model(self, n_dim, r):
        # load inputs
        X, _ = self.inputs
        L = self.layers
        
        #n_filters = [128, 256, 512,512]
        #n_filtersizes = [65, 33, 17,  9]
        
        n_filters = [32,  48,  64, 64]
        n_filtersizes = [16, 10, 5,  5]
        
        #n_filters = [10, 20, 40, 40, 40, 40, 40, 40]
        #n_filtersizes = [30, 20, 15, 10, 10, 10, 10, 10]
        
        downsampled_l = []
        with tf.name_scope('generator'):
            # save origin-X 
            oX = X
            print('>> Generator Model init...')
            # downsampling layers
            for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
                X = downsample_layer(X, nf, fs)
                downsampled_l.append(X)
                print('D-Block >> ' ,X)
            
            # Bottle-neck layer
            X = downsample_layer(X, n_filters[-1], n_filtersizes[-1], B=True)
            print('B-Block >> ', X)
            
            # Upsample layer
            L = reversed(range(L))
            n_filters = reversed(n_filters)
            n_filtersizes = reversed(n_filtersizes)
            downsampled_l = reversed(downsampled_l)

            for l, nf, fs, l_in in zip( L, (n_filters), (n_filtersizes), (downsampled_l)):
                #X = reshape1Dto2D(X)
                X = upsample_layer(X, nf*2, fs)
                #X = reshape2Dto1D(X)
                X = tf.concat([X,l_in],axis=-1)
                print('U-Block >> ',X)
                
            # Final layer and add input layer
            X = upsample_layer(X,nf=2,ks=9)
            G = tf.add(X,oX)
            print('Fin-Layer >> ',G)
            print('>> ...finish')
            print()
          
        return G
    
    def create_train_op(self, X, Y):
        # load params
        opt_params = self.opt_params
        print('creating train_op with params:', opt_params)

        # create loss
        self.loss = self.create_objective(X, Y, opt_params)

        # create params - get trainable variables
        # params = self.get_params()

        # create optimizer
        self.optimizer = self.create_optimzier(opt_params)

        # create gradients 
        # grads = self.create_gradients(self.loss, params)

        # create training op
        #with tf.name_scope('optimizer'):
        # ref - https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
        #train_op = self.create_updates(params, grads, alpha, opt_params)
        train_op = self.optimizer.minimize(self.loss)

        # initialize the optimizer variabLes
        #optimizer_vars = []
        #for v in tf.global_variables():
        #    if 'optimizer/' in v.name:
        #        optimizer_vars.append(v)
               
        #init = tf.variables_initializer(optimizer_vars)
        #self.sess.run(init)

        return train_op
    
    
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
    
    
    def create_optimzier(self, opt_params):
        if opt_params['alg'] == 'adam':
            lr, b1, b2 = opt_params['lr'], opt_params['b1'], opt_params['b2']
            optimizer = tf.train.AdamOptimizer(lr, b1, b2)
        else:
            raise ValueError('Invalid optimizer: ' + opt_params['alg'])

        return optimizer

    
# ----------------------------------------------------------------------------
# in training sequence   
    
    def fit(self, X_train, Y_train, X_val, Y_val, n_epoch=100):
        
        # init the model
        init = tf.global_variables_initializer()
        # init = tf.initialize_all_variables()
        self.sess.run(init)

        
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
        
        
        
        for epoch in range(n_epoch):
            is_batch_fin = False
            step = 1
            start_time = time.time()
            # loop train data on batch size
            count = 0
            while not is_batch_fin:
                count += 1
                
                # load next batch data
                d,l,is_batch_fin = train_data.next_batch(n_batch)
                batch = (d,l)
                
                # get batch feed-dict
                feed_dict = self.load_batch(batch)
                
                # training batch-size
                tr_objective = self.train_batch(feed_dict)
                tr_obj_snr = 20 * np.log10(1. / np.sqrt(tr_objective) + 1e-8)
                
                # print batch log
                print('count %d / obj: %f / snr: %f'%(count, tr_objective, tr_obj_snr))
                    
                # last case
                if is_batch_fin:
                    end_time = time.time()
                    
                    # evaluation model each epoch
                    tr_l2_loss, tr_l2_snr = self.eval_err(X_train, Y_train, n_batch=n_batch)
                    va_l2_loss, va_l2_snr = self.eval_err(X_val, Y_val, n_batch=n_batch)
                    
                    # print epoch log
                    print()
                    print("Epoch {} of {} took {:.3f}s ({} minibatches)".format(
                        epoch+1, n_epoch, end_time - start_time, len(X_train) // n_batch))
                    print("  training l2_loss/segsnr:\t\t{:.6f}\t{:.6f}".format(
                        tr_l2_loss, tr_l2_snr))
                    print("  validation l2_loss/segsnr:\t\t{:.6f}\t{:.6f}".format(
                        va_l2_loss, va_l2_snr))
                    print("-----------------------------------------------------------------------")
                    
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
                    self.saver.save(self.sess, self.checkpoint_root)

                    # restart clock
                    start_time = time.time()
            
            # in for loop    
            step += 1
    
    def load_batch(self, batch, train=True):
        X_in, Y_in = self.inputs
        X, Y = batch

        if Y is not None:
            feed_dict = {X_in : X, Y_in : Y}
        else:
            feed_dict = {X_in : X}

        return feed_dict
    
    
    def train_batch(self, feed_dict):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss
    
    
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
    
    
    def predict(self, X, Y):
        assert len(X) == 1
        x_sp = spline_up(X, self.r)
        x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(self.layers+1)))]
        X = x_sp.reshape((1,len(x_sp),1))
        feed_dict = self.load_batch((X,Y), train=False)
        return self.sess.run(self.predictions, feed_dict=feed_dict)
    
    
    def load(self, ckpt):
        # get checkpoint name
        # ref - https://www.tensorflow.org/api_docs/python/tf/train/latest_checkpoint
        if os.path.isdir(ckpt): checkpoint = tf.train.latest_checkpoint(ckpt)
        else: checkpoint = ckpt
        meta = checkpoint + '.meta'
        print('checkpoint:',checkpoint)
        print('ckpt:',ckpt)
        
        # load
        self.saver = tf.train.Saver()
        #self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, checkpoint)
        
        '''
        # load graph
        self.saver = tf.train.import_meta_graph(meta)
        g = tf.get_default_graph()
        # load weights
        self.saver.restore(self.sess, checkpoint)
        
        # get graph tensors
        X, Y = tf.get_collection('inputs')

        # save tensors as instance variables
        self.inputs = X, Y
        self.predictions = tf.get_collection('preds')[0]

        # load existing loss, or erase it, if creating new one
        g.clear_collection('losses')

        # create a new training op
        self.train_op = self.create_train_op(X, Y)
        g.clear_collection('train_op')
        tf.add_to_collection('train_op', self.train_op)
        
        '''
        
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
        
        
def downsample_layer(x , nf, ks, B=False):
    x = tf.layers.conv1d(
        x,
        filters = nf,
        kernel_size = ks,
        strides=1,
        padding='same',
        data_format='channels_last',
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None
    )
    x = tf.layers.max_pooling1d(
        x,
        pool_size = 2,
        strides = 2,
        padding='same',
        data_format='channels_last',
        name=None
    )
    
    if B : x = tf.layers.dropout(x, rate=0.5)
        
    x = tf.nn.relu(x)
    return x

def upsample_layer(x, nf, ks):
    '''x = tf.layers.conv2d_transpose(
        x,
        filters = nf,
        kernel_size = [1,ks],
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None
    )'''
    x = tf.layers.conv1d(
        x,
        filters = nf,
        kernel_size = ks,
        strides=1,
        padding='same',
        data_format='channels_last',
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None
    )
    x = tf.layers.dropout(x, rate=0.5)
    x = tf.nn.relu(x)
    x = SubPixel1D(x,r=2)
    return x 

def spline_up(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)

    x_sp = interpolate.splev(i_hr, f)

    return x_sp 
    