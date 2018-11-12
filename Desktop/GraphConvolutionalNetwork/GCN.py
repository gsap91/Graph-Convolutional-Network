#https://github.com/tkipf/pygcn
import torch.nn as nn
import torch.nn.functional as F
from GraphConvolution import GraphConvolution
import torch._C
import mnist_on_plane
import convolutions
import graph_utils

class GCN(nn.Module):
    def __init__(self, shape, conv_channels, linear_channels):
        super(GCN, self).__init__()

        self.conv_channels = conv_channels
        self.linear_channels = linear_channels

        self.A = graph_utils.add_fictional_node_to_A(mnist_on_plane.grid_adj_matrix(shape[0],shape[1]))
        self.norm_A = torch.FloatTensor(convolutions.kipf_welling_norm(convolutions.degree_matrix(self.A), self.A))

        self.gc1 = GraphConvolution(conv_channels[0], conv_channels[1])
        self.gc1_bn = nn.BatchNorm1d(conv_channels[1])
        #self.gc1_drop = nn.Dropout(p=0.2)

        self.gc2 = GraphConvolution(conv_channels[1], conv_channels[2])
        self.gc2_bn = nn.BatchNorm1d(conv_channels[2])
        #self.gc2_drop = nn.Dropout(p=0.2)

        self.gc3 = GraphConvolution(conv_channels[2], conv_channels[3])
        self.gc3_bn = nn.BatchNorm1d(conv_channels[3])
        #self.gc3_drop = nn.Dropout(p=0.2)


        self.gc4 = GraphConvolution(conv_channels[3], conv_channels[4])
        self.gc4_bn = nn.BatchNorm1d(conv_channels[4])
        #self.gc4_drop = nn.Dropout(p=0.2)
        '''
        self.gc5 = GraphConvolution(conv_channels[4], conv_channels[5])
        self.gc5_bn = nn.BatchNorm1d(conv_channels[5])
        #self.gc5_drop = nn.Dropout(p=0.2)

        self.gc6 = GraphConvolution(conv_channels[5], conv_channels[6])
        self.gc6_bn = nn.BatchNorm1d(conv_channels[6])
        #self.gc6_drop = nn.Dropout(p=0.2)
        
        self.gc7 = GraphConvolution(conv_channels[6], conv_channels[7])
        self.gc7_bn = nn.BatchNorm1d(conv_channels[7])
        #self.gc7_drop = nn.Dropout(p=0.2)

        self.gc8 = GraphConvolution(conv_channels[7], conv_channels[8])
        self.gc8_bn = nn.BatchNorm1d(conv_channels[8])
        #self.gc8_drop = nn.Dropout(p=0.2)


        self.gc9 = GraphConvolution(conv_channels[8], conv_channels[9])
        self.gc9_bn = nn.BatchNorm1d(conv_channels[9])

        
        self.gc10 = GraphConvolution(conv_channels[9], conv_channels[10])
        self.gc10_bn = nn.BatchNorm1d(conv_channels[10])

        self.gc11 = GraphConvolution(conv_channels[10], conv_channels[11])
        self.gc11_bn = nn.BatchNorm1d(conv_channels[11])

        self.gc12 = GraphConvolution(conv_channels[11], conv_channels[12])
        self.gc12_bn = nn.BatchNorm1d(conv_channels[12])

        self.gc13 = GraphConvolution(conv_channels[12], conv_channels[13])
        self.gc13_bn = nn.BatchNorm1d(conv_channels[13])

        self.gc14 = GraphConvolution(conv_channels[13], conv_channels[14])
        self.gc14_bn = nn.BatchNorm1d(conv_channels[14])

        self.gc15 = GraphConvolution(conv_channels[14], conv_channels[15])
        self.gc15_bn = nn.BatchNorm1d(conv_channels[15])

        self.gc16 = GraphConvolution(conv_channels[15], conv_channels[16])
        self.gc16_bn = nn.BatchNorm1d(conv_channels[16])

        self.gc17 = GraphConvolution(conv_channels[16], conv_channels[17])
        self.gc17_bn = nn.BatchNorm1d(conv_channels[17])

        self.gc18 = GraphConvolution(conv_channels[17], conv_channels[18])
        self.gc18_bn = nn.BatchNorm1d(conv_channels[18])

        self.gc19 = GraphConvolution(conv_channels[18], conv_channels[19])
        self.gc19_bn = nn.BatchNorm1d(conv_channels[19])

        self.gc20 = GraphConvolution(conv_channels[19], conv_channels[20])
        self.gc20_bn = nn.BatchNorm1d(conv_channels[20])

        
        self.gc21 = GraphConvolution(conv_channels[20], conv_channels[21])
        self.gc21_bn = nn.BatchNorm1d(conv_channels[21])

        self.gc22 = GraphConvolution(conv_channels[21], conv_channels[22])
        self.gc22_bn = nn.BatchNorm1d(conv_channels[22])

        self.gc23 = GraphConvolution(conv_channels[22], conv_channels[23])
        self.gc23_bn = nn.BatchNorm1d(conv_channels[23])

        
        self.gc24 = GraphConvolution(conv_channels[23], conv_channels[24])
        self.gc24_bn = nn.BatchNorm1d(conv_channels[24])
        '''


        self.fc1 = nn.Linear(conv_channels[4], linear_channels[0])
        self.fc1_bn = nn.BatchNorm1d(linear_channels[0])
        #self.fc1_drop = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(linear_channels[0], linear_channels[1])
        self.fc2_bn = nn.BatchNorm1d(linear_channels[1])
        #self.fc2_drop = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(linear_channels[1], linear_channels[2])
        self.fc3_bn = nn.BatchNorm1d(linear_channels[2])

        self.fc4 = nn.Linear(linear_channels[2], linear_channels[3])
        self.fc4_bn = nn.BatchNorm1d(linear_channels[3])
        
        self.fc5 = nn.Linear(linear_channels[3], linear_channels[4])
        self.fc5_bn = nn.BatchNorm1d(linear_channels[4])

        self.fc6 = nn.Linear(linear_channels[4], linear_channels[5])
        self.fc6_bn = nn.BatchNorm1d(linear_channels[5])

        self.fc7 = nn.Linear(linear_channels[5], linear_channels[6])
        self.fc7_bn = nn.BatchNorm1d(linear_channels[6])

        self.fc8 = nn.Linear(linear_channels[6], linear_channels[7])
        self.fc8_bn = nn.BatchNorm1d(linear_channels[7])



    def forward(self, x):

        x = self.gc1(x, self.norm_A)
        x = self.gc1_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)
        #x = F.dropout(x, training=self.training)

        x = self.gc2(x, self.norm_A)
        x = self.gc2_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)
        #x = F.dropout(x, training=self.training)

        x = self.gc3(x, self.norm_A)
        x = self.gc3_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)
        #x = F.dropout(x, training=self.training)

        x = self.gc4(x, self.norm_A)
        x = self.gc4_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)
        #x = F.dropout(x, training=self.training)
        '''
        x = self.gc5(x, self.norm_A)
        x = self.gc5_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)
        #x = F.dropout(x, training=self.training)

        x = self.gc6(x, self.norm_A)
        x = self.gc6_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)
        #x = F.dropout(x, training=self.training)
        
        x = self.gc7(x, self.norm_A)
        x = self.gc7_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)
        #x = F.dropout(x, training=self.training)

        x = self.gc8(x, self.norm_A)
        x = self.gc8_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)
        #x = F.dropout(x, training=self.training)


        x = self.gc9(x, self.norm_A)
        x = self.gc9_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)
        
        x = self.gc10(x, self.norm_A)
        x = self.gc10_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)


        x = self.gc11(x, self.norm_A)
        x = self.gc11_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)

        x = self.gc12(x, self.norm_A)
        x = self.gc12_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)

        x = self.gc13(x, self.norm_A)
        x = self.gc13_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)

        x = self.gc14(x, self.norm_A)
        x = self.gc14_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)

        x = self.gc15(x, self.norm_A)
        x = self.gc15_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)

        x = self.gc16(x, self.norm_A)
        x = self.gc16_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)
        
        x = self.gc17(x, self.norm_A)
        x = self.gc17_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)

        x = self.gc18(x, self.norm_A)
        x = self.gc18_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)

        x = self.gc19(x, self.norm_A)
        x = self.gc19_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)

        x = self.gc20(x, self.norm_A)
        x = self.gc20_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)
        
        x = self.gc21(x, self.norm_A)
        x = self.gc21_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)

        x = self.gc22(x, self.norm_A)
        x = self.gc22_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)

        x = self.gc23(x, self.norm_A)
        x = self.gc23_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)
        
        x = self.gc24(x, self.norm_A)
        x = self.gc24_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x)
        '''


        x = x[:,-1,:]

        x = self.fc1(x)

        x = self.fc1_bn(x)
        x = F.leaky_relu(x)
        #x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        x = self.fc2_bn(x)
        x = F.leaky_relu(x)
        #x = F.dropout(x, training=self.training)
    
        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = F.leaky_relu(x)

        x = self.fc4(x)

        x = self.fc4_bn(x)
        x = F.leaky_relu(x)
        
        x = self.fc5(x)
        x = self.fc5_bn(x)
        x = F.leaky_relu(x)

        x = self.fc6(x)
        x = self.fc6_bn(x)
        x = F.leaky_relu(x)

        x = self.fc7(x)
        x = self.fc7_bn(x)
        x = F.leaky_relu(x)

        x = self.fc8(x)
        x = self.fc8_bn(x)
        x = F.leaky_relu(x)

        return x