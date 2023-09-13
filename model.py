import tensorflow as tf

def fully_connected(input, size):
    weights = tf.compat.v1.get_variable('weights', 
                              shape=[input.get_shape()[1],size], #!why one
                              initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
                              )
    biases = tf.compat.v1.get_variable('biases',
                             shape=[size],
                             initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
                             )
    return tf.matmul(input,weights)+biases

def fully_connected_relu(input, size):
    return tf.nn.relu(fully_connected(input,size))

def conv_relu(input, kernel_size, depth):
    weights= tf.compat.v1.get_variable('weights',
                             shape=[kernel_size,kernel_size,input.get_shape()[3],depth],
                             initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
                            )
    biases = tf.compat.v1.get_variable('biases',
                             shape=[depth],
                            initializer = tf.compat.v1.keras.initializers.VarianceScaling( scale=1.0, mode="fan_avg", distribution="uniform")
                            )
    conv = tf.nn.conv2d(input, filters=weights, strides=[1,1,1,1],padding='SAME')
    return tf.nn.relu(conv+biases)

def pool(input,size):
    return tf.nn.max_pool2d(
        input=input,
        ksize=[1,size,size,1],
        strides=[1,size,size,1],
        padding='SAME'
    )

def model_pass(input, params, is_training):
    with tf.compat.v1.variable_scope('conv1'):
        conv1=conv_relu(input, kernel_size=params.conv1_k, depth=params.conv1_d)
        pool1=pool(conv1,size=2)
        pool1=tf.cond(is_training, lambda: tf.nn.dropout(pool1, rate = 1 - (params.conv1_p)), lambda: pool1)

    with tf.compat.v1.variable_scope('conv2'):
        conv2=conv_relu(input, kernel_size=params.conv2_k,depth = params.conv2_d)
        pool2=pool(conv2,size=2)
        pool2=tf.cond(is_training, lambda: tf.nn.dropout(pool2, rate = 1 - (params.conv2_p)), lambda: pool2)

    with tf.compat.v1.variable_scope('conv3'):
        conv3=conv_relu(input, kernel_size=params.conv3_k,depth = params.conv3_d)
        pool3=pool(conv3,size=2)
        pool3=tf.cond(is_training, lambda: tf.nn.dropout(pool3, rate = 1 - (params.conv3_p)), lambda: pool3)

    # 1st stage output
    pool1 = pool(pool1, size = 4)
    shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])
    
    # 2nd stage output
    pool2 = pool(pool2, size = 2)
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]]) 
    
    # 3rd stage output
    shape = pool3.get_shape().as_list()
    pool3 = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])
    
    flattened = tf.concat([pool1, pool2,pool3],axis=1)  

    with tf.compat.v1.variable_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size = params.fc4_size)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, rate = 1 - (params.fc4_p)), lambda: fc4)
    with tf.compat.v1.variable_scope('out'):
        logits = fully_connected(fc4, size = params.num_classes)
    return logits

