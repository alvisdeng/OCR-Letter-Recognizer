import argparse
import numpy as np
import os

class NerualNetwork():
    def __init__(self):
        self.X = None
        self.Y = None

        self.alpha = None
        self.beta = None
        self.beta_asterisk = None
        self.alpha_asterisk = None
    
    def read_file(self,input_file):
        data = np.loadtxt(input_file,delimiter=',')
        X = []
        Y = [] 
        for row in data:
            y = row[0]
            x = row[1:]
            Y.append(self.convert_to_onehot(10,int(y)))
            X.append(np.insert(x.reshape(len(x),1),0,[1],axis=0))
        return X,Y

    def fit(self,input_file,hidden_units,init_flag):
        self.X,self.Y = self.read_file(input_file)

        feature_num = len(self.X[0])
        if int(init_flag) == 1:
            self.alpha_asterisk = np.random.uniform(low=-0.1,high=0.1,size=(int(hidden_units),feature_num-1))
            self.beta_asterisk = np.random.uniform(low=-0.1,high=0.1,size=(10,int(hidden_units)))

            self.alpha = np.insert(self.alpha_asterisk,0,[0],axis=1)
            self.beta = np.insert(self.beta_asterisk,0,[0],axis=1)
        elif int(init_flag) == 2:
            self.alpha = np.zeros([int(hidden_units),feature_num])
            self.beta = np.zeros([10,int(hidden_units)+1])
            self.alpha_asterisk = self.alpha[:,1:]
            self.beta_asterisk = self.beta[:,1:]
    
    def train(self,num_epoch,learning_rate,valid_file,metric_file):
        valid_X,valid_Y = self.read_file(valid_file)
        out_cross_entropy = []

        for epoch in range(1,int(num_epoch)+1):
            for i in range(len(self.Y)):
                x = self.X[i]
                y = self.Y[i]

                obj = self.forward_function(x,self.alpha,self.beta)
                diff_params = self.backward_function(obj,x,y)
                self.update_params(diff_params,float(learning_rate))

                if i <= 2:
                    print("i: " + str(i))
                    print('alpha:')
                    print(self.alpha)
                    print('beta:')
                    print(self.beta)
            
            train_Y_hat = self.get_Y_hat(self.X)
            valid_Y_hat = self.get_Y_hat(valid_X)

            train_cross_entropy = self.cross_entropy_function(self.Y,train_Y_hat)
            valid_cross_entropy = self.cross_entropy_function(valid_Y,valid_Y_hat)
            out_cross_entropy.append((epoch,train_cross_entropy,valid_cross_entropy))

        with open(metric_file,mode='a') as f:
            for pair in out_cross_entropy:
                epoch = pair[0]
                train_cross_entropy = pair[1]
                valid_cross_entropy = pair[2]
                f.write('epoch='+str(epoch)+' crossentropy(train): '+str(train_cross_entropy)+'\n')
                f.write('epoch='+str(epoch)+' crossentropy(validation): '+str(valid_cross_entropy)+'\n')
    
    def predict(self,input_file,output_file):
        input_X,input_Y = self.read_file(input_file)
        input_Y_hat = self.get_Y_hat(input_X,self.alpha,self.beta)

        with open(output_file,mode='w') as f:
            for y_hat in input_Y_hat:
                f.write(str(self.classify(y_hat))+'\n')
    
    def evaluate(self,true_file,predict_file):
        Y = []
        Y_hat = []
        with open(true_file,mode='r') as f:
            for line in f.readlines():
                line = line.strip().split(',')
                Y.append(int(line[0]))
        with open(predict_file,mode='r') as f:
            for line in f.readlines():
                Y_hat.append(int(line.strip()))
        
        wrong = 0
        for i in range(len(Y)):
            if Y[i] != Y_hat[i]:
                wrong += 1
        return wrong/len(Y)

    def get_Y_hat(self,X,alpha=None,beta=None):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta

        Y_hat = []
        for x in X:
            obj = self.forward_function(x,alpha,beta)
            y_hat = obj[-1]
            Y_hat.append(y_hat)
        return Y_hat

    def classify(self,y_hat):
        return np.argmax(y_hat)

    def cross_entropy_function(self,Y,Y_hat):
        cross_entropy = 0
        for i in range(len(Y)):
            y = Y[i]
            y_hat = Y_hat[i]
            cross_entropy += -np.dot(y.T,np.log(y_hat))[0][0]
        mean_cross_entropy = cross_entropy/len(Y)
        return mean_cross_entropy

    def forward_function(self,x,alpha=None,beta=None):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        
        a = self.linear_prod(v=x,M=alpha)
        z = self.sigmoid(a)
        z_dummy = np.insert(z,0,[1],axis=0)
        b = self.linear_prod(v=z_dummy,M=beta)
        y_hat = self.softmax(b)

        obj = [z,z_dummy,y_hat]
        return obj
    
    def backward_function(self,obj,x,y):
        diff_b = self.get_diff_b(y,obj[2])
        diff_beta = self.get_diff_beta(diff_b,obj[1])
        diff_z = self.get_diff_z(self.beta_asterisk,diff_b)
        diff_a = self.get_diff_a(diff_z,obj[0])
        diff_alpha = self.get_diff_alpha(diff_a,x)
        diff_params = [diff_alpha,diff_beta]
        return diff_params
    
    def update_params(self,diff_params,learning_rate):
        diff_alpha = diff_params[0]
        diff_beta = diff_params[1]

        self.alpha = self.alpha - learning_rate*diff_alpha
        self.beta = self.beta - learning_rate*diff_beta
        self.beta_asterisk = self.beta[:,1:]

    def get_diff_b(self,y,y_hat):
        return -y+y_hat
    
    def get_diff_beta(self,diff_b,z_dummy):
        return np.dot(diff_b,z_dummy.T)
    
    def get_diff_z(self,beta_asterisk,diff_b):
        return np.dot(beta_asterisk.T,diff_b)
    
    def get_diff_a(self,diff_z,z):
        return diff_z*z*(1-z)
    
    def get_diff_alpha(self,diff_a,x):
        return np.dot(diff_a,x.T)

    def linear_prod(self,v,M):
        return np.dot(M,v)
    
    def sigmoid(self,v):
        return 1/(1+np.exp(-v))
    
    def softmax(self,v):
        denominator = np.sum(np.exp(v))
        return np.exp(v)/denominator
    
    def convert_to_onehot(self,num_class,num):
        v = np.zeros([num_class,1])
        v[num] = 1
        return v
    
    def clear(self,*args):
        for file_name in args:
            if os.path.exists(file_name):
                os.remove(file_name)

def parse_args():
    """
    Parse input positional arguments from command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser('Handwritten Letter Recognizer - Neural Network')
    parser.add_argument('train_input',help='path to the training input .csv ﬁle')
    parser.add_argument('validation_input',help='path to the validation input .csv ﬁle')
    parser.add_argument('train_out',help='path to output .labels ﬁle to which the prediction on the training data should be written')
    parser.add_argument('validation_out',help='path to output .labels ﬁle to which the prediction on the validation data should be written')
    parser.add_argument('metrics_out',help='path of the output .txt ﬁle to which metrics such as train and validation error should be written')
    parser.add_argument('num_epoch',help='integer specifying the number of times backpropogation loops through all of the training data')
    parser.add_argument('hidden_units',help='positive integer specifying the number of hidden units')
    parser.add_argument('init_flag',help='integer taking value 1 or 2 that speciﬁes whether to use RANDOM or ZERO initialization')
    parser.add_argument('learning_rate',help='ﬂoat value specifying the learning rate for SGD')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    nn_classifier = NerualNetwork()
    nn_classifier.clear(args.train_out,args.validation_out,args.metrics_out)

    nn_classifier.fit(args.train_input,args.hidden_units,args.init_flag)
    nn_classifier.train(args.num_epoch,args.learning_rate,args.validation_input,args.metrics_out)

    nn_classifier.predict(args.train_input,args.train_out)
    nn_classifier.predict(args.validation_input,args.validation_out)

    train_err = nn_classifier.evaluate(args.train_input,args.train_out)
    validation_err = nn_classifier.evaluate(args.validation_input,args.validation_out)

    with open(args.metrics_out,mode='a') as f:
        f.write('error(train): '+str(train_err)+'\n')
        f.write('error(validation): '+str(validation_err))
        