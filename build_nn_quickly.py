# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


# pyTorch model.
class signNet(torch.nn.Module):
    
    def __init__(self, d_in, d_out, set_relu=False):
        self.flag = set_relu
        super(signNet, self).__init__()
        self.linear = torch.nn.Linear(d_in, d_out)
        if set_relu!=False:
            self.relu = torch.nn.ReLU()
        self.linear.weight.data.fill_(1)
        self.linear.bias.data.fill_(0.1)


    def forward(self, input_features): 
        linearOut = self.linear(input_features)
        if self.flag != False:
            actvOut = self.relu(linearOut)
            return actvOut
        return linearOut

# initialize the input vector
x = torch.randn(1,3)
print ("\nINPUT: ", x, "\n")

# create first neural network(Using ReLU)
net1 = signNet(3, 4, True)
result = net1(x)
print ("--NETWORK--\n",net1,"\nOUTPUT:", result, "\n")

# create second neural network(w/o ReLU)
net2 = signNet(3, 4)
result = net2(x)
print ("--NETWORK--\n",net2,"\nOUTPUT:", result)