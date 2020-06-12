import sys
import math
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import numpy as np

## default settings
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
MAX_INITS = 2            # number of restarts
ABORT_EARLY = True       # if we can't improve anymore, abort early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess
POSITIVE_MEAN = 0        # 0: free mean; 1: biased penalty; 2: softplus; 3: clipping
VARIANCE = .1
NUM_ROWS = 1
NUM_COLUMNS = 1

class ManRGB:

    def __init__(self, sess, model, batch_size=1, confidence=CONFIDENCE,
            targeted=TARGETED, learning_rate=LEARNING_RATE,
            max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
            initial_const=INITIAL_CONST, max_inits = MAX_INITS,
            positive_mean=POSITIVE_MEAN, variance=VARIANCE, num_rows=NUM_ROWS,
            num_columns=NUM_COLUMNS, channel=None):

        self.sess = sess

        self.ABORT_EARLY    = abort_early
        self.MAX_ITERATIONS = max_iterations
        self.MAX_INITS      = max_inits
        self.TARGETED       = targeted

        image_size   = model.image_size
        num_channels = model.num_channels 
        num_labels   = model.num_labels

        assert image_size % num_rows == 0
        assert image_size % num_columns == 0

        cell_width  = int(image_size / num_columns)
        cell_height = int(image_size / num_rows)

        # to make the following code shorter...
        sample_shape   = (batch_size, image_size, image_size, num_channels)
        sample_shape   = np.asarray(sample_shape)
        variable_shape = (num_rows, num_columns, num_channels)
        cell_shape     = (batch_size, cell_width, cell_height)


        # this way is more efficient in sending data to tf every iteration
        self.assign_tlab = tf.placeholder(tf.float32, shape=[num_labels])
        self.assign_timg = tf.placeholder(tf.float32, shape=sample_shape[1:])
        self.tlab = tf.Variable(np.zeros(num_labels), dtype=tf.float32)
        self.timg = tf.Variable(np.zeros(sample_shape[1:]), dtype=tf.float32)

        init_noi_mean = tf.random.uniform(dtype=tf.float32,
                            shape=variable_shape, minval=0., maxval=1.)
        noi_mean = tf.Variable(init_noi_mean, validate_shape=False)
        self.noi_mean_nonneg = tf.math.softplus(noi_mean)

        def channel_model(channel, x, delta):
        ## channel emulation

            if channel:
                color_matrix      = channel['color_matrix']
                const_dig         = channel['const_dig']
                const_ana         = channel['const_ana']
                ana_intensity     = channel['ana_intensity']
                digital_intensity = channel['dig_intensity']
                const_ill         = channel['const_ill']
                const_rho         = channel['const_rho']
                env_ill           = channel['env_ill']
            else: # y = (x_f/|x_f| + x_o)
                color_matrix = np.identity(3).astype(np.float32)
                const_dig, const_ana, ana_intensity = 0., 0., 0.
                const_ill, env_ill, const_rho = np.inf, np.inf, 1.
                digital_intensity = 1.

            # we will look at only the ratio among three channels
            noi_mean_max = tf.reduce_max(delta, axis=[2])[:, :, None]
            noi_mean_normalized = delta / noi_mean_max * digital_intensity

            # color mapping: tensor -> matrix -> tensor
            shape_2d = [num_channels, num_rows * num_columns]
            shape_3d = [num_channels, num_rows,  num_columns]
            noi_mean_mapped = tf.transpose(noi_mean_normalized, perm=[2, 0, 1])
            noi_mean_mapped = tf.reshape(noi_mean_mapped, shape=shape_2d)
            noi_mean_mapped = tf.matmul(color_matrix, noi_mean_mapped)
            noi_mean_mapped = tf.reshape(noi_mean_mapped, shape=shape_3d)
            noi_mean_mapped = tf.transpose(noi_mean_mapped, perm=[1, 2, 0])

            dig_intensity = tf.reduce_mean(noi_mean_normalized)
            ill = tf.sigmoid(const_dig * dig_intensity
                    + const_ana * ana_intensity + const_ill)
            noi_mean_boosted = noi_mean_mapped * ill * const_rho

            # setup noise distributions
            tfd = tfp.distributions
            dis = tfd.Normal(loc=noi_mean_boosted, scale=variance)
            noi = dis.sample(cell_shape)
            noi = tf.transpose(noi, perm=[0, 5, 3, 2, 4, 1])
            noi = tf.reshape(noi, shape=sample_shape[[0, 3, 1, 2]])
            modifier = tf.transpose(noi, perm=[0, 2, 3, 1])

            gamma = env_ill / (ill + env_ill)
            y = gamma * (modifier + x)

            return y

        self.newimg = channel_model(channel, self.timg, self.noi_mean_nonneg)

        self.l2dist = tf.norm(self.newimg - self.timg) # l2 distance
        self.output = model.predict(self.newimg) # logits
        
        # compute the probability of the label class versus the maximum other
        self.real = tf.reduce_sum((self.tlab)*self.output, 1) # 2nd largest logits
        self.other = tf.reduce_max((1-self.tlab)*self.output -
                (self.tlab*10000), 1) # targeted class logits

        if self.TARGETED: # optimize for making the other class most likely
            self.logit_loss = tf.maximum(0.0, self.other-self.real+confidence)
        else: # optimize for making this class least likely.
            self.logit_loss = tf.maximum(0.0, self.real-self.other+confidence)

        ## losses
        self.loss_mean = tf.norm(noi_mean)
        self.loss_logits = tf.reduce_sum(self.logit_loss) / sample_shape[0]
        self.loss = self.loss_mean + initial_const * self.loss_logits


        # Setup the adam optimizer
        var_list   = [noi_mean]
        optimizer  = tf.train.AdamOptimizer(learning_rate)
        self.train = optimizer.minimize(self.loss, var_list=var_list)

        ## Initialization
        init_tlab = self.tlab.assign(self.assign_tlab)
        init_timg = self.timg.assign(self.assign_timg)
        init_noi  = tf.variables_initializer(var_list)
        init_opt  = tf.variables_initializer(optimizer.variables())
        self.initializer = tf.group([init_tlab, init_timg, init_noi, init_opt])

    def attack(self, img, target):

        max_succ_ratio = 0.
        prev_logit_loss = np.inf
        # begin searching from randomly chosen initial points for
        # self.MAX_INITS times
        for init_index in range(self.MAX_INITS):
            print('No.{} init'.format(init_index))

            self.sess.run(self.initializer,
                          feed_dict={self.assign_timg: img,
                                     self.assign_tlab: target})

            prev_loss = math.inf
            for iteration in range(self.MAX_ITERATIONS):

                ## one iteration of the optimization
                self.sess.run(self.train)

                rslts = self.sess.run([
                    self.loss, self.l2dist,
                    self.output, self.newimg,
                    self.noi_mean_nonneg,
                    self.loss_mean, self.loss_logits])

                ## attack evaluation for each iteration
                logit_loss = rslts[-1]
                if logit_loss <= prev_logit_loss:
                    prev_logit_loss = logit_loss
                    best_img = rslts[3][0]
                    best_mean = rslts[4]

                    # count the hits
                    preds = np.argmax(rslts[2], axis=1)
                    origs = np.argmax(target)
                    if self.TARGETED:
                        succ_ratio = np.sum(preds == origs) / len(preds)
                    else:
                        succ_ratio = np.sum(preds != origs) / len(preds)
                    max_succ_ratio = succ_ratio

                    ## DONE!
                    if np.isclose(logit_loss, 0):
                        print(iteration, 'logits_loss = 0. Done!')
                        #print('avg logits', np.mean(rslts[2], axis=0))
                        return max_succ_ratio, best_img, best_mean, logit_loss


                ## verbose
                if iteration % (self.MAX_ITERATIONS // 10) == 0:
                    curr_loss, lm, ll = rslts[0], rslts[-2], logit_loss
                    print(iteration, succ_ratio, curr_loss, lm, ll)

                    # abort early if we're going nowhere
                    if self.ABORT_EARLY and curr_loss > prev_loss:
                        print('going nowhere... abort')
                        break

                    prev_loss = curr_loss

        ## return something even it is not 100% successful
        return max_succ_ratio, best_img, best_mean, logit_loss
