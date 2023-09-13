import os
class Paths(object):

    def __init__(self,params):
        self.model_name = self.get_model_name(params)
        self.var_scope = self.get_variables_scope(params)
        self.root_path = 'traffic_sign_classification/' + "/models/" + self.model_name + "/"
        self.model_path = self.get_model_path()
        self.train_history_path = self.get_model_path()
        self.learning_curves_path = self.get_learning_curves_path()
        os.makedirs(self.root_path, exist_ok = True)
    def get_variables_scope(self, params):
        # We will encode model settings in its name: architecture, optimisations applied, etc.
        var_scope = "k{}d{}_k{}d{}_k{}d{}_fc{}_fc0".format(
            params.conv1_k, params.conv1_d,
            params.conv2_k, params.conv2_d,
            params.conv3_k, params.conv3_d, 
            params.fc4_size
        )
        return var_scope
    def get_model_name(self, params):
        # We will encode model settings in its name: architecture, optimisations applied, etc.
        model_name = "k{}d{}p{}_k{}d{}p{}_k{}d{}p{}_fc{}p{}".format(
            params.conv1_k, params.conv1_d, params.conv1_p, 
            params.conv2_k, params.conv2_d, params.conv2_p, 
            params.conv3_k, params.conv3_d, params.conv3_p, 
            params.fc4_size, params.fc4_p
        )
        model_name += "_lrdec" if params.learning_rate_decay else "_no-lrdec"
        model_name += "_l2" if params.l2_reg_enabled else "_no-l2"
        return model_name
    def get_model_path(self):
        return self.root_path+'model.ckpt'
    def get_train_history(self):
        return self.root_path+'train_history'
    def get_learning_curves_path(self):
        return self.root_path+'learning_curves.png'