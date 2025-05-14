import torch
import torch.nn as nn
from torchquad import MonteCarlo
import numpy as np


class ECELoss(nn.Module):
    def __init__(self, num_bins=15):
        super(ECELoss, self).__init__()
        self.num_bins = num_bins
    
    def forward(self, probs, labels):
        bins = torch.linspace(0, 1, self.num_bins + 1, device=probs.device)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]
        
        ece = torch.tensor(0.0, device=probs.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Get the indices of probabilities within the current bin
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            bin_count = torch.sum(in_bin).item()
            
            if bin_count > 0:
                avg_confidence = torch.mean(probs[in_bin])
                avg_accuracy = torch.mean(labels[in_bin].float())
                ece += (bin_count / len(probs)) * torch.abs(avg_accuracy - avg_confidence)
        return ece

class ECLoss(nn.Module):
    '''
    Expectation consistency loss (Ours)
    for Low-dimensional data
    '''

    def __init__(self,train_x,test_x, model,num_bins=15):
        super(ECLoss, self).__init__()
        self.num_bins = num_bins
        self.train_x = train_x
        self.test_x = test_x
        self.model = model

    def forward(self,batch_train_x,batch_train_y,train_pred,train_probs,test_probs):
        batch_train_x = batch_train_x
        index = batch_train_y == train_pred
        self.P_Y_hatY = sum(index)/len(index)
        self.Xset_Y_hatY = batch_train_x[index]
        
        bins = torch.linspace(0, 1, self.num_bins + 1, device=train_probs.device)
        # quantiles = torch.linspace(0, 1, self.num_bins + 1, device=train_probs.device)
        # bins = torch.quantile(test_probs, quantiles)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]
        bounds = [(-5, 5), (-5, 5)]
        ec = torch.tensor(0.0, device=train_probs.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            self.bin_lower = bin_lower
            self.bin_upper = bin_upper
            train_in_bin = (train_probs > bin_lower) & (train_probs <= bin_upper)
            train_bin_count = torch.sum(train_in_bin).item()
            self.train_bin_rate = (train_bin_count / len(train_probs))

            in_bin = (test_probs > bin_lower) & (test_probs <= bin_upper)
            bin_count = torch.sum(in_bin).item()
            self.test_bin_rate = (bin_count / len(test_probs))
            if bin_count>0:
                mc_integrator = MonteCarlo()
                integral_result = mc_integrator.integrate(
                                    fn=self.integrator_fun, 
                                    integration_domain=bounds, 
                                    dim=2,
                                    N=10000  
                                )
                ec = ec + (bin_count / len(test_probs)) * torch.abs(integral_result)
        return ec
    
    def integrator_fun(self,x):

        x = x
        hatS = torch.softmax(self.model(x), dim=1).max(dim=1).values.detach()
        index = (hatS >= self.bin_lower) & (hatS <= self.bin_upper)
        densities_Xt = torch.tensor([0. for i in range(len(hatS))],device=x.device)
        densities_Xs = torch.tensor([0. for i in range(len(hatS))],device=x.device)

        if self.test_bin_rate != 0.:
            densities_Xt[index] =  self.kernel_density_estimation(self.test_x, x[index])/self.test_bin_rate
        
        if self.train_bin_rate != 0.:
            densities_Xs[index] =  self.kernel_density_estimation(self.train_x, x[index])/self.train_bin_rate

        densities_diff = densities_Xt-densities_Xs

        P_Hs = (self.kernel_density_estimation(self.Xset_Y_hatY,x)*self.P_Y_hatY)/(self.kernel_density_estimation(self.train_x,x)+ 1e-8)
        P_Hs = torch.clamp(P_Hs, min=0.0, max=1.0)
        return P_Hs*densities_diff
    
    def kernel_density_estimation(self, data, query_points):
        n_samples,d = data.shape
        # std_dev = torch.std(data, dim=0).mean()
        # bandwidth = (4 / (d + 2))**(1 / (d + 4)) * n_samples**(-1 / (d + 4)) * std_dev
        bandwidth = 0.3
        n_queries = query_points.shape[0]
        densities = torch.zeros(n_queries)
        for i in range(n_queries):
            diff = query_points[i] - data  
            norm_squared = torch.sum(diff ** 2, axis=1) 
            kernel_values = torch.exp(-norm_squared / (2 * bandwidth ** 2)) 
            densities[i] = torch.sum(kernel_values) / (n_samples * (2 * np.pi * bandwidth ** 2)) 

        return densities