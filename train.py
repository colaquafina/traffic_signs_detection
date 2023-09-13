import Path
import logger
import tensorflow as tf
import time
import model
import numpy as np
from matplotlib import pyplot
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))     


class EarlyStopping(object):

    def __init__(self, saver, session,patience = 100, minimize = True):
        self.minimize = minimize
        self.patience = patience
        self.saver = saver
        self.session = session
        self.best_monitored_value = np.inf if minimize else 0.
        self.best_monitored_epoch = 0
        self.restore_path = None

    def __call__(self, value, epoch):
        if (self.minimize and value < self.best_monitored_value) or (not self.minimize and value > self.best_monitored_value):
            self.best_monitored_value = value
            self.best_monitored_epoch = epoch
            self.restore_path = self.saver.save(self.session,  "traffic_sign_classification/EarlyStopping/early_stopping_checkpoint")
        elif self.best_monitored_epoch + self.patience < epoch:
            if self.restore_path != None:
                self.saver.restore(self.session, self.restore_path)
            else:
                print("ERROR: Failed to restore session")
            return True
        
        return False

def train_model(params, X_train, y_train, X_valid, y_valid, X_test, y_test):
    # Initialisation routines: generate variable scope, create logger, note start time.
    paths = Path.Paths(params)
    start = time.time()
    model_variable_scope = paths.var_scope

    
    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a variable that will be fed at run time with a training minibatch.
        tf_x_batch = tf.compat.v1.placeholder(tf.float32, shape = (None, params.image_size[0], params.image_size[1], 1))
        tf_y_batch = tf.compat.v1.placeholder(tf.float32, shape = (None, params.num_classes))
        is_training = tf.compat.v1.placeholder(tf.bool)
        current_epoch = tf.Variable(0, trainable=False)  # count the number of epochs

        # Model parameters.
        if params.learning_rate_decay:
            learning_rate = tf.keras.optimizers.schedules(params.learning_rate, current_epoch, decay_steps = params.max_epochs, decay_rate = 0.01)
        else:
            learning_rate = params.learning_rate

        with tf.compat.v1.variable_scope(model_variable_scope):
            logits = model.model_pass(tf_x_batch, params, is_training)
            if params.l2_reg_enabled:
                with tf.compat.v1.variable_scope('fc4', reuse = True):
                    l2_loss = tf.nn.l2_loss(tf.compat.v1.get_variable('weights'))
            else:
                l2_loss = 0

        predictions = tf.nn.softmax(logits)
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = tf_y_batch, logits = logits)
        loss = tf.reduce_mean(softmax_cross_entropy) + params.l2_lambda * l2_loss 
        
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate = learning_rate
        ).minimize(loss)
    
    with tf.compat.v1.Session(graph=graph) as session:
        session.run(tf.compat.v1.global_variables_initializer())
        def get_accuracy_and_loss_in_batches(X,y):
            p=[]
            sce=[]
            num_samples = X.shape[0]
            batch_size = params.batch_size
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                x_batch = X[start:end]
                y_batch = y[start:end]

                p_batch, sce_batch = session.run([predictions, softmax_cross_entropy], feed_dict={
                    tf_x_batch: x_batch,
                    tf_y_batch: y_batch,
                    is_training: False
                })
                p.extend(p_batch)
                sce.extend(sce_batch)
            p = np.array(p)
            sce = np.array(sce)
            accuracy = 100.0 * np.sum(np.argmax(p, 1) == np.argmax(y, 1)) / p.shape[0]
            loss = np.mean(sce) 
            return (accuracy, loss)
        if params.resume_training: 
            try:
                tf.compat.v1.train.Saver().restore(session, paths.model_path)
            except Exception as e:
                print("Failed restoring previously trained model: file does not exist.")
                pass
        saver = tf.compat.v1.train.Saver()
        early_stopping = EarlyStopping(tf.compat.v1.train.Saver(), session, patience = params.early_stopping_patience, minimize = True)
        train_loss_history = np.empty([0], dtype = np.float32)
        train_accuracy_history = np.empty([0], dtype = np.float32)
        valid_loss_history = np.empty([0], dtype = np.float32)
        valid_accuracy_history = np.empty([0], dtype = np.float32)
        if params.max_epochs > 0:
            print("================= TRAINING ==================")
        else:
            print("================== TESTING ==================")       
        print(" Timestamp: " + logger.get_time_hhmmss())

        for epoch in range(params.max_epochs):
            current_epoch = epoch
            batch_size = params.batch_size
            num_samples = len(X_train)
            shuffle_indices = np.arange(num_samples)
            np.random.shuffle(shuffle_indices)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = shuffle_indices[start:end]
                x_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                # Run the optimizer to update model weights
                session.run([optimizer], feed_dict={
                    tf_x_batch: x_batch,
                    tf_y_batch: y_batch,
                    is_training: True
                })
                if(epoch %params.log_epoch==0):
                    valid_accuracy, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
                    train_accuracy, train_loss = get_accuracy_and_loss_in_batches(X_train, y_train)

                    if(epoch % params.print_epoch ==0):
                        print("-------------- EPOCH %4d/%d --------------" % (epoch, params.max_epochs))
                        print("     Train loss: %.8f, accuracy: %.2f%%" % (train_loss, train_accuracy))
                        print("Validation loss: %.8f, accuracy: %.2f%%" % (valid_loss, valid_accuracy))
                        print("      Best loss: %.8f at epoch %d" % (early_stopping.best_monitored_value, early_stopping.best_monitored_epoch))
                        print("   Elapsed time: " + logger.get_time_hhmmss(start))
                        print("      Timestamp: " + logger.get_time_hhmmss())
                else:
                    valid_loss = 0.
                    valid_accuracy = 0.
                    train_loss = 0.
                    train_accuracy = 0.
                
                valid_loss_history = np.append(valid_loss_history, [valid_loss])
                valid_accuracy_history = np.append(valid_accuracy_history, [valid_accuracy])
                train_loss_history = np.append(train_loss_history, [train_loss])
                train_accuracy_history = np.append(train_accuracy_history,[train_accuracy])

                if params.early_stopping_enabled:
                    if valid_loss == 0:
                        _, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
                    if early_stopping(valid_loss, epoch): 
                        print("Early stopping.\nBest monitored loss was {:.8f} at epoch {}.".format(
                            early_stopping.best_monitored_value, early_stopping.best_monitored_epoch
                        ))
                        break

        test_accuracy, test_loss = get_accuracy_and_loss_in_batches(X_test, y_test)
        valid_accuracy, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
        print("=============================================")
        print(" Valid loss: %.8f, accuracy = %.2f%%)" % (valid_loss, valid_accuracy)) 
        print(" Test loss: %.8f, accuracy = %.2f%%)" % (test_loss, test_accuracy)) 
        print(" Total time: " + logger.get_time_hhmmss(start))
        print("  Timestamp: " + logger.get_time_hhmmss())

        saved_model_path = saver.save(session, paths.model_path)
        print("Model file: " + saved_model_path)
        np.savez(paths.train_history_path, train_loss_history = train_loss_history, train_accuracy_history = train_accuracy_history, 
                 valid_loss_history = valid_loss_history, valid_accuracy_history = valid_accuracy_history)
        print("Train history file: " + paths.train_history_path)
        
        logger.plot_learning_curves(params)
        
        pyplot.show()
