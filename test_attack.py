## Author: Yanmao Man <yman@email.arizona.edu>.

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
from PIL import Image

from setup_cifar import CIFARModel
from setup_lisa import LISAModel
from setup_inception import InceptionModel

from rgb_attack import ManRGB

def single_image(
        target_label,           # targeted class
        img_path=None,          # path to the image
        batch_size=100,         # T: number of trials to approximate the mean
        max_iterations=1000,    # max num of iteration of Adam until we give up
        initial_const=1,        # c in Eq. 14
        targeted=True,          # targeted attack
        variance=.005,          # the variance of the attack noise
        positive_mean=2,        # using softplus to guarantee non-negativity
        max_inits=10,           # max numof restarts
        num_rows=2,             # N_row in Sec. 5.4.3
        num_columns=2,          # N_col in Sec. 5.4.3
        confidence=0,           # kappa in Eq. 9
        objective='creation',   # attack objective: alteration or creation?
        dataset='cifar',        # dataset / classifier to be attacked
        rho=30,                 # rho in Eq. 8
        ana=.2,                 # P_a in Eq. 6
        dig=.6,                 # T_d in Eq. 6
        env=.02):               # I_env in Eq. 7

    ## channel parameters
    H = np.array([[0.6,   0, 0],# color calibration matrix: H_c
                  [0,   0.5, 0],
                  [0,     0, 1]]).astype(np.float32)
    const_dig, const_ana, const_ill = 8.9, 6.7, -7.8 # a, b, c_t in Eq. 6
    const_rho, ana_intensity, env_ill, digital_intensity = rho, ana, env, dig
    channel = {'color_matrix': H, 'const_dig': const_dig,
            'const_ana': const_ana, 'ana_intensity': ana_intensity,
            'digital_intensity': digital_intensity,
            'const_ill': const_ill, 'const_rho':const_rho,
            'env_ill': env_ill}

    with tf.Session() as sess:

        ## load the classifier
        if dataset == 'cifar':
            model = CIFARModel("./models/cifar", sess)
        elif dataset == 'lisa':
            model = LISAModel('./models/lisa_color_balanced', sess)
        elif dataset == 'imagenet':
            model = InceptionModel(
                    './models/inception_v3_2016_08_28_frozen.pb', sess)

        num_labels, image_size = model.num_labels, model.image_size
        num_channels = model.num_channels

        if objective == 'alteration':
            assert img_path != None # must provide an image

            ben_img = Image.open(img_path).resize([image_size, image_size])
            ben_img = np.array(ben_img).astype(np.float32) / 255.

            # verify the benign class
            x_shape = [None, image_size, image_size, num_channels]
            x = tf.placeholder(tf.float32, x_shape)
            y = model.predict(x)
            pred = sess.run(y, {x: ben_img[None, ...]})
            pred = np.argmax(pred, axis=1)
            print('Benign label:', pred)

        else: # default: creation attacks
            ben_img = np.zeros([image_size, image_size, num_channels])

        tar_lab = np.zeros(num_labels)
        tar_lab[target_label] = 1

        ## get ready for the attack
        attack = ManRGB(sess, model, batch_size=batch_size,
                max_iterations=max_iterations, confidence=confidence,
                initial_const=initial_const*batch_size, targeted=targeted,
                num_rows=num_rows, num_columns=num_columns, variance=variance,
                positive_mean=positive_mean, max_inits=max_inits, channel=channel)

        ## attack: it may take some time
        rslts = attack.attack(ben_img, tar_lab)

        ## attack results
        hit = rslts[0]
        adv = rslts[1]
        best_mean = rslts[2]

        ## save the adversarial example
        adv = (adv * 255).astype(np.uint8)
        adv = Image.fromarray(adv)
        adv.save('a.png')

if __name__ == "__main__":

    single_image(img_path='./benign_images/ILSVRC2012_val_00019992.JPEG',
            target_label=555, objective='alteration', dataset='imagenet',
            dig=.7, ana=.1, num_rows=20, num_columns=20)

    single_image(img_path='./benign_images/ship1.png', target_label=6,
            objective='alteration', dataset='cifar', dig=.7, ana=.1,
            num_rows=2, num_columns=2)

    single_image(img_path='./benign_images/5.jpg', target_label=1,
            objective='alteration', dataset='lisa', dig=.7, ana=.1,
            num_rows=2, num_columns=2)

    single_image(target_label=5, objective='creation', dataset='lisa',
            dig=.7, ana=.1, num_rows=2, num_columns=2)
