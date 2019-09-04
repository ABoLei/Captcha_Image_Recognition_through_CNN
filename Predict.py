import tensorflow as tf
import Definitions
import numpy as np
from Train_VerificationCode_CNN import GenerateNextBatch
from Train_VerificationCode_CNN import Vector2Text
from Train_VerificationCode_CNN import GetWrappedCaptchaTextAndImage
from Train_VerificationCode_CNN import GenerateBatch
from Train_VerificationCode_CNN import crack_captcha_cnn
from Train_VerificationCode_CNN import Convert2gray
from Train_VerificationCode_CNN import FlattenImage2Array
from Train_VerificationCode_CNN import Crack_Image

# Define CNN
def restore_cnn(w_c1, b_c1, w_c2, b_c2, w_c3, b_c3, w_d, b_d, w_out, b_out):
    # 将占位符 转换为 按照图片给的新样式
    x = tf.reshape(X, shape=[-1, Definitions.IMAGE_HEIGHT, Definitions.IMAGE_WIDTH, 1])

    #w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    #w_c2_alpha = np.sqrt(2.0/(3*3*32))
    #w_c3_alpha = np.sqrt(2.0/(3*3*64))
    #w_d1_alpha = np.sqrt(2.0/(8*32*64))
    #out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    # w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]), name='w_c1') # 从正太分布输出随机值
    # tf.summary.histogram(name='Layer0/Weights', values=w_c1)#for tensorboard visualization
    # b_c1 = tf.Variable(b_alpha*tf.random_normal([32]), name='b_c1')
    # tf.summary.histogram(name='Layer0/Biases', values=b_c1)#for tensorboard visualization
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]), name='w_c2')
    # tf.summary.histogram(name='Layer1/Weights', values=w_c2)#for tensorboard visualization
    # b_c2 = tf.Variable(b_alpha*tf.random_normal([64]), name='b_c2')
    # tf.summary.histogram(name='Layer1/Biases', values=b_c2)#for tensorboard visualization
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]), name='w_c3')
    # tf.summary.histogram(name='Layer2/Weights', values=w_c3)  # for tensorboard visualization
    # b_c3 = tf.Variable(b_alpha*tf.random_normal([64]), name='b_c3')
    # tf.summary.histogram(name='Layer2/Biases', values=b_c3)  # for tensorboard visualization
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    # w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]), name='w_d')
    # tf.summary.histogram(name='Layer3/Weights', values=w_d)  # for tensorboard visualization
    # b_d = tf.Variable(b_alpha*tf.random_normal([1024]), name='b_d')
    # tf.summary.histogram(name='Layer3/Biases', values=b_d)  # for tensorboard visualization
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    # w_out = tf.Variable(w_alpha*tf.random_normal([1024, Definitions.MAX_CAPTCHA * Definitions.CHAR_SET_LEN]), name='w_out')
    # tf.summary.histogram(name='Layer4/Weights', values=w_out)  # for tensorboard visualization
    # b_out = tf.Variable(b_alpha*tf.random_normal([Definitions.MAX_CAPTCHA * Definitions.CHAR_SET_LEN]), name='b_out')
    # tf.summary.histogram(name='Layer4/Biases', values=b_out)  # for tensorboard visualization
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # tf.summary.histogram(name='Output/', values=out)  # for tensorboard visualization
    #out = tf.nn.softmax(out)
    return out

def crack_sol1():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("./Models/crack_capcha.model-112400.meta")
        # saver.restore(sess, "./Models")
        model = tf.train.get_checkpoint_state("./Models/")
        saver.restore(sess, model.model_checkpoint_path)
        # saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")

        graph = tf.get_default_graph()
        w_c1 = graph.get_tensor_by_name("w_c1:0")
        b_c1 = graph.get_tensor_by_name("b_c1:0")
        w_c2 = graph.get_tensor_by_name("w_c2:0")
        b_c2 = graph.get_tensor_by_name("b_c2:0")
        w_c3 = graph.get_tensor_by_name("w_c3:0")
        b_c3 = graph.get_tensor_by_name("b_c3:0")
        w_d = graph.get_tensor_by_name("w_d:0")
        b_d = graph.get_tensor_by_name("b_d:0")
        w_out = graph.get_tensor_by_name("w_out:0")
        b_out = graph.get_tensor_by_name("b_out:0")
        # feed_dict = {w1: 13.0, w2: 17.0}

        output = restore_cnn(w_c1, b_c1, w_c2, b_c2, w_c3, b_c3, w_d, b_d, w_out, b_out)

        print("Variables restored.")

        predict = tf.reshape(output, [-1, Definitions.MAX_CAPTCHA, Definitions.CHAR_SET_LEN])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(Y, [-1, Definitions.MAX_CAPTCHA, Definitions.CHAR_SET_LEN]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        text, imagenp, image = GetWrappedCaptchaTextAndImage()

        from matplotlib.pyplot import imshow
        imshow(imagenp)

        batch_x_test, batch_y_test = GenerateBatch(text, imagenp, 1)

        # feed_dict = {X: batch_x_test, keep_prob: 1.}
        # predictions = output.eval(feed_dict=feed_dict)

        # batch_x_test, batch_y_test = GenerateNextBatch(1)
        # feed_dict = {X: batch_x_test, Y: batch_y_test, keep_prob: 1.}
        feed_dict = {X: batch_x_test, keep_prob: 1.}

        output = tf.nn.softmax(output)
        predictions = sess.run(output, feed_dict=feed_dict)
        index = np.argmax(predictions)
        char = Definitions.charSet[index]

        acc = sess.run(accuracy, feed_dict=feed_dict)
        print('accuracy', acc)

        predictions, = sess.run(predict, feed_dict=feed_dict)
        # highest_probability_index = np.argmax(predictions)
        vectorIndices = []
        charSets = []
        for i in predictions:
            index = np.argmax(i)
            vectorIndices.append(index)
            charSets.append(Definitions.charSet[index])

        # out = tf.nn.softmax(output)
        # result = out.eval(feed_dict=feed_dict)
        predictions, = sess.run(output, feed_dict=feed_dict)
        highest_probability_index = np.argmax(predictions)
        predictions = Vector2Text(highest_probability_index)
        print('result', predictions)

        a = 0

        # print("w_c1 : %s" % w_c1.eval())
        # print("b_c1 : %s" % b_c1.eval())
        # output = crack_captcha_cnn()
        # predict = tf.reshape(output, [-1, Definitions.MAX_CAPTCHA, Definitions.CHAR_SET_LEN])
        # max_idx_p = tf.argmax(predict, 2)
        # max_idx_l = tf.argmax(tf.reshape(Y, [-1, Definitions.MAX_CAPTCHA, Definitions.CHAR_SET_LEN]), 2)

text, imagenp, image = GetWrappedCaptchaTextAndImage(4)
imagenp = Convert2gray(imagenp)
imagenp = FlattenImage2Array(imagenp)
predictText = Crack_Image(imagenp)
print('Actual:', text, ', Predict:', predictText)