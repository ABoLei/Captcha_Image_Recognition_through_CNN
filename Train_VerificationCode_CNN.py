import numpy as np
import tensorflow as tf
import Definitions
from CaptchaImage import GetCaptchaTextAndImage
# from PIL import Image
# from matplotlib.pyplot import imshow


def GetImageWH(image):
    # import numpy as np

    if isinstance(image, str):
        from PIL import Image
        image = Image.open(image)

    img_np = np.array(image)

    return img_np.shape[1], img_np.shape[0]


# 灰階化
def Convert2gray(image):
    if len(image.shape) > 2:
        gray = np.mean(image, -1)
        return gray
    else:
        return image


# Standardlize Text To Vector
def Text2Vector(text):
    #     import ipynb.fs.full.Definitions

    text_len = len(text)
    if text_len > Definitions.MAX_CAPTCHA:
        raise ValueError('MAX_CAPTCHA out of length')

    vector = np.zeros(Definitions.MAX_CAPTCHA * Definitions.CHAR_SET_LEN)

    def GetCharIndex(char):
        return Definitions.charSet.index(char)

    for index, char in enumerate(text):
        idx = index * Definitions.CHAR_SET_LEN + GetCharIndex(char)
        vector[idx] = 1

    return vector


# vec = Text2Vector('hC2a')
# print(vec.nonzero()[0])

def Vector2Text(vector):
    text = []
    charPositions = vector.nonzero()[0]

    for index, charPos in enumerate(charPositions):
        charIndex = charPos - index * Definitions.CHAR_SET_LEN
        text.append(Definitions.charSet[charIndex])

    return text

# Make sure image is 160(W) * 60(H) * 3(RGB) format
def GetWrappedCaptchaTextAndImage(charCount=1):
    text, imagenp, image = GetCaptchaTextAndImage(charCount)
    return text, imagenp, image
    # while True:
    #     text, imagenp, image = GetCaptchaTextAndImage(charCount)

        # if imagenp.shape == (60, 160, 3):  # 此部分应该与开头部分图片宽高吻合
        #     return text, imagenp

def FlattenImage2Array(image):
    return image.flatten() / 255  # 將圖像一維化

def GenerateBatch(text,image,batchSize=1):
    batch_x = np.zeros([batchSize, Definitions.IMAGE_HEIGHT * Definitions.IMAGE_WIDTH])
    batch_y = np.zeros([batchSize, Definitions.MAX_CAPTCHA * Definitions.CHAR_SET_LEN])

    for i in range(batchSize):
        # text, image = GetWrappedCaptchaTextAndImage()
        image = Convert2gray(image)

        # 将图片数组一维化 同时将文本也对应在两个二维组的同一行
        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = Text2Vector(text)

    # 返回该训练批次
    return batch_x, batch_y

def GenerateNextBatch(batchSize=128):
    import numpy as np

    batch_x = np.zeros([batchSize, Definitions.IMAGE_HEIGHT * Definitions.IMAGE_WIDTH])
    batch_y = np.zeros([batchSize, Definitions.MAX_CAPTCHA * Definitions.CHAR_SET_LEN])

    for i in range(batchSize):
        text, imagenp, image = GetWrappedCaptchaTextAndImage()
        image = Convert2gray(image)

        # 将图片数组一维化 同时将文本也对应在两个二维组的同一行
        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = Text2Vector(text)

    # 返回该训练批次
    return batch_x, batch_y


X = tf.placeholder(tf.float32, [None, Definitions.IMAGE_HEIGHT * Definitions.IMAGE_WIDTH], name='X')
Y = tf.placeholder(tf.float32, [None, Definitions.MAX_CAPTCHA * Definitions.CHAR_SET_LEN], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout


# Define CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    # 将占位符 转换为 按照图片给的新样式
    x = tf.reshape(X, shape=[-1, Definitions.IMAGE_HEIGHT, Definitions.IMAGE_WIDTH, 1])

    #w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    #w_c2_alpha = np.sqrt(2.0/(3*3*32))
    #w_c3_alpha = np.sqrt(2.0/(3*3*64))
    #w_d1_alpha = np.sqrt(2.0/(8*32*64))
    #out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]), name='w_c1') # 从正太分布输出随机值
    tf.summary.histogram(name='Layer0/Weights', values=w_c1)#for tensorboard visualization
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]), name='b_c1')
    tf.summary.histogram(name='Layer0/Biases', values=b_c1)#for tensorboard visualization
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]), name='w_c2')
    tf.summary.histogram(name='Layer1/Weights', values=w_c2)#for tensorboard visualization
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]), name='b_c2')
    tf.summary.histogram(name='Layer1/Biases', values=b_c2)#for tensorboard visualization
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]), name='w_c3')
    tf.summary.histogram(name='Layer2/Weights', values=w_c3)  # for tensorboard visualization
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]), name='b_c3')
    tf.summary.histogram(name='Layer2/Biases', values=b_c3)  # for tensorboard visualization
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]), name='w_d')
    tf.summary.histogram(name='Layer3/Weights', values=w_d)  # for tensorboard visualization
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]), name='b_d')
    tf.summary.histogram(name='Layer3/Biases', values=b_d)  # for tensorboard visualization
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([1024, Definitions.MAX_CAPTCHA * Definitions.CHAR_SET_LEN]), name='w_out')
    tf.summary.histogram(name='Layer4/Weights', values=w_out)  # for tensorboard visualization
    b_out = tf.Variable(b_alpha*tf.random_normal([Definitions.MAX_CAPTCHA * Definitions.CHAR_SET_LEN]), name='b_out')
    tf.summary.histogram(name='Layer4/Biases', values=b_out)  # for tensorboard visualization
    out = tf.add(tf.matmul(dense, w_out), b_out)
    tf.summary.histogram(name='Output/', values=out)  # for tensorboard visualization
    #out = tf.nn.softmax(out)
    return out


def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    tf.summary.scalar('loss', loss)  # for tensorboard visualization
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, Definitions.MAX_CAPTCHA, Definitions.CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, Definitions.MAX_CAPTCHA, Definitions.CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)  # for tensorboard visualization

    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 將視覺化輸出
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('Tensorboard/', sess.graph)

        step = 0
        while True:
            batch_x, batch_y = GenerateNextBatch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            batch_x_test, batch_y_test = GenerateNextBatch(100)
            acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
            print('step:', step, 'loss:', loss_, 'accuracy:', acc)

            # if step % 20 == 0:
            #     result = sess.run(merged, feed_dict={X: batch_x, Y: batch_y})
            #     writer.add_summary(result, step)
            result = sess.run(merged, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            writer.add_summary(result, step)

            # 每100 step计算一次准确率
            if step % 100 == 0:
                target_acc = 0.9
                batch_x_test, batch_y_test = GenerateNextBatch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print('step:', step, 'accuracy:', acc)
                # 如果准确率大于50%,保存模型,完成训练
                if acc > target_acc:
                    target_acc += 0.1
                    saver.save(sess, "./Models/crack_capcha.model", global_step=step)
                    # writer.close()
                    # break
            step += 1

def Crack_Image(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./Models'))

        # predict = tf.reshape(output, [-1, Definitions.MAX_CAPTCHA, Definitions.CHAR_SET_LEN])
        predict = tf.argmax(tf.reshape(output, [-1, Definitions.MAX_CAPTCHA, Definitions.CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(Definitions.MAX_CAPTCHA * Definitions.CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * Definitions.CHAR_SET_LEN + n] = 1
            i += 1
        return Vector2Text(vector)

# train_crack_captcha_cnn()