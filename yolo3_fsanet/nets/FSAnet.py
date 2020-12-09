import torch
import torch.nn as nn
import torch.nn.functional as F


#1d CapsuleLayer similar to nn.Linear (which outputs scalar neurons),
#here, we output vectored neurons
# input shape: bs, [256, 512, 1024], [(52x52), (26x26), (13x13)]
# output shape: bs, 3, 16, [(52x52), (26x26), (13x13)]
class CapsuleLayer1d(nn.Module):
    # 7*3, 64, 3, 16
    def __init__(self,num_in_capsule,in_capsule_dim,num_out_capsule,out_capsule_dim,routings=3):
        super(CapsuleLayer1d, self).__init__()
        self.routings = routings
        #Affine Transformation Weight Matrix which maps spatial relationship
        #between input capsules and output capsules
        ##initialize affine weight
        # shape (3*(7x3)*16*64)
        weight_tensor = torch.empty(
            num_out_capsule,
            num_in_capsule,
            out_capsule_dim,
            in_capsule_dim)

        init_weight = torch.nn.init.xavier_uniform_(weight_tensor)
        self.affine_w = nn.Parameter(init_weight)

    def squash(self, s, dim=-1):
        norm = torch.sum(s**2, dim=dim, keepdim=True)
        return norm / (1 + norm) * s / (torch.sqrt(norm) + 1e-8)

    def forward(self,x):
        #input shape: [batch,num_in_capsule,in_capsule_dim],
        #We will exapnd its dims so that we can do batch matmul properly
        #expanded input shape: [batch,1,num_in_capsule,1,in_capsule_dim]
        x = x.unsqueeze(1)
        x = x.unsqueeze(3)
        #input shape: [batch,1,num_in_capsule,1,in_capsule_dim],
        #weight shape: [num_out_capsule,num_in_capsule,out_capsule_dim,in_capsule_dim]
        #last two dims will be used for matrix multiply, rest is our batch.
        #result = input*w.T
        #result shape: [batch,num_out_capsule,num_in_capsule,1,out_capsule_dim]
        u_hat = torch.matmul(x,torch.transpose(self.affine_w,2,3))
        #reduced result shape: [batch,num_out_capsule,num_in_capsule,out_capsule_dim]
        u_hat = u_hat.squeeze(3)

        [num_out_capsule,num_in_capsule,out_capsule_dim,in_capsule_dim] = \
        self.affine_w.shape

        #initialize coupling coefficient as zeros
        b = torch.zeros(u_hat.shape[0],num_out_capsule,num_in_capsule).to(u_hat.device)

        for i in range(self.routings):
            #c is used to scale/weigh our input capsules based on their
            #similarity with our output capsules
            #summing up c for all output capsule equals to 1 due to softmax
            #this ensures probability distrubtion to our weights
            c = F.softmax(b,dim=1)
            #expand c
            c = c.unsqueeze(2)

            #u_hat shape: [batch,num_out_capsule,num_in_capsule,out_capsule_dim],
            #c shape: [batch,num_out_capsule,1,num_in_capsule]
            #result = c*u_hat
            #result shape: [batch,num_out_capsule,1,out_capsule_dim]
            outputs = torch.matmul(c,u_hat)
            #Apply non linear activation function
            outputs = self.squash(outputs)

            if i < self.routings - 1:
                #update coupling coefficient
                #u_hat shape: [batch,num_out_capsule,num_in_capsule,out_capsule_dim],
                #outputs shape: [batch,num_out_capsule,1,out_capsule_dim]
                #result = u_hat*outputs.T
                #result shape: [batch,num_out_capsule,num_in_capsule,1]
                b = b + torch.matmul(u_hat,torch.transpose(outputs,2,3)).squeeze(3)
                #reduced result shape: [batch,num_out_capsule,num_in_capsule]
                b = b

        #reduced result shape: [batch,num_out_capsule,out_capsule_dim]
        outputs = outputs.squeeze(2)
        # print(f"[INFO] capsule outputs: {outputs.shape}")
        return outputs

class ExtractAggregatedFeatures(nn.Module):
    def __init__(self, num_capsule):
        super(ExtractAggregatedFeatures, self).__init__()
        self.num_capsule = num_capsule

    def forward(self,x):
        batch_size = x.shape[0]
        bin_size = self.num_capsule//3

        feat_s1 = x[:,:bin_size,:]
        feat_s1 = feat_s1.view(batch_size,-1) #reshape to 1d

        feat_s2 = x[:,bin_size:2*bin_size,:]
        feat_s2 = feat_s2.view(batch_size,-1)

        feat_s3 = x[:,2*bin_size:self.num_capsule,:]
        feat_s3 = feat_s3.view(batch_size,-1)
        # print(f"[INFO] feat_s1: {feat_s1.shape}")
        # print(f"[INFO] feat_s2: {feat_s2.shape}")
        # print(f"[INFO] feat_s3: {feat_s3.shape}")
        return [feat_s1,feat_s2,feat_s3]

class ExtractSSRParams(nn.Module):
    def __init__(self,bins,classes):
        #our classes are: pitch, roll, yaw
        #our bins per stage are: 3
        super(ExtractSSRParams, self).__init__()
        self.bins = bins
        self.classes = classes

        self.shift_fc = nn.Linear(2,classes) #used to shift bins

        self.scale_fc = nn.Linear(2,classes) #used to scale bins

        #every class will have its own probability distrubtion of bins
        #hence total predictions = bins*classes
        self.pred_fc = nn.Linear(4,bins*classes) #classes probability distrubtion of bins

    #x is batches of feature vector of shape: [batches,16]
    def forward(self,x):
        shift_param = torch.tanh(self.shift_fc(x[:,:2]))
        scale_param = torch.tanh(self.scale_fc(x[:,2:4]))
        pred_param = F.relu(self.pred_fc(x[:,4:]))
        pred_param = pred_param.view(pred_param.size(0),
                                    self.classes,
                                    self.bins)
        # print(f"[INFO] shift_param: {shift_param.shape}")
        # print(f"[INFO] scale_param: {scale_param.shape}")
        # print(f"[INFO] pred_param: {pred_param.shape}")
        return [pred_param,shift_param,scale_param]

class SSRLayer(nn.Module):
    def __init__(self, bins):
        #this ssr layer implements MD 3-stage SSR
        super(SSRLayer, self).__init__()
        self.bins_per_stage = bins

    #x is list of ssr params for each stage
    def forward(self,x):
        s1_params,s2_params,s3_params = x

        a = b = c = 0

        bins = self.bins_per_stage

        doffset = bins//2

        V = 99 #max bin width

        #Stage 1 loop over all bins
        for i in range(bins):
            a = a + (i - doffset + s1_params[1]) * s1_params[0][:,:,i]
        #this is unfolded multiplication loop of SSR equation in paper
        #here, k = 1
        a = a / (bins * (1 + s1_params[2]))

        #Stage 2 loop over all bins
        for i in range(bins):
            b = b + (i - doffset + s2_params[1]) * s2_params[0][:,:,i]
        #this is unfolded multiplication loop of SSR equation in paper
        #here, k = 2
        b = b / (bins * (1 + s1_params[2])) / (bins * (1 + s2_params[2]))

        #Stage 3 loop over all bins
        for i in range(bins):
            c = c + (i - doffset + s3_params[1]) * s3_params[0][:,:,i]
        #this is unfolded multiplication loop of SSR equation in paper
        #here, k = 3
        c = c / (bins * (1 + s1_params[2])) / (bins * (1 + s2_params[2])) / (bins * (1 + s3_params[2]))

        pred = (a + b + c) * V
        # print(f"[INFO] pred: {pred.shape}")
        return pred

class FSANet(nn.Module):
    def __init__(self, num_primcaps, primcaps_dim, num_out_capsule, out_capsule_dim, routings, var=False):
        super(FSANet, self).__init__()
        # num_primcaps = 5*3
        # primcaps_dim = 8
        # num_out_capsule = 3
        # out_capsule_dim = 8
        # routings = 2

        self.caps_layer = CapsuleLayer1d(num_primcaps,primcaps_dim,num_out_capsule,out_capsule_dim,routings)
        self.eaf = ExtractAggregatedFeatures(num_out_capsule)
        self.esp_s1 = ExtractSSRParams(3,3)
        self.esp_s2 = ExtractSSRParams(3,3)
        self.esp_s3 = ExtractSSRParams(3,3)
        self.ssr = SSRLayer(3)

    def forward(self,x):
        #Output: 3 capsules with shortened dims each representing a stage
        #Output Shape: capsules has shape [batch,3,16]
        x = self.caps_layer(x)
        # print(x)

        #Input: Output of caps_layer module
        #Output: each stage capsule seprated as 1d vector
        #Output Shape: 3 capsules, each has shape [batch,16]
        x = self.eaf(x)

        #Input: Output of eaf module
        #Output: ssr params for each stage
        #Output Shape: ssr_params = [preds,shift,scale]
        #preds shape: [batch,3,3]
        #shift shape: [batch,3]
        #scale shape: [batch,3]
        ##Extract SSR params of each stage
        ssr_s1 = self.esp_s1(x[0])
        ssr_s2 = self.esp_s2(x[1])
        ssr_s3 = self.esp_s3(x[2])

        #Input: Output of esp modules
        #Output: ssr pose prediction
        #Output Shape: ssr_params = [batch,3]
        ##get prediction from SSR layer
        x = self.ssr([ssr_s1,ssr_s2,ssr_s3])
        return x

if __name__ == '__main__':
    num_primcaps = 7*3
    primcaps_dim = 32
    num_out_capsule = 3
    out_capsule_dim = 16
    routings = 2
    torch.random.manual_seed(10)
    model = FSANet(num_primcaps, primcaps_dim, num_out_capsule, out_capsule_dim, routings, var=True).to('cuda')
    # model = CapsuleLayer1d(num_primcaps,primcaps_dim,num_out_capsule,out_capsule_dim,routings).to('cuda')
    print('##############PyTorch################')
    x = torch.randn(4, 3*3*7*32*2, 52, 52).to('cuda')
    x = x.view((4*52*52*3*2,7*3,32))
    print(x.shape)
    y = model(x)
    print(model)

    print("Output:", y.shape)
    # y = y.view(64, 3, 3, 52, 52).contiguous()
    # print(y.shape)

