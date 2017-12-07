from ntp.modules import MultiLayerPerceptron
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class EnergyModel(nn.Module):
    def __init__(self, energy_modules, search_iters=50, lr=.1, tol=1e-6):

        super(EnergyModel, self).__init__()
        for module in energy_modules:
            module.spen()
        self.energy_modules_ = nn.ModuleList(energy_modules)
#        self.combine_module = MultiLayerPerceptron(
#            len(energy_modules), 1,
#            hidden_sizes=10,
#            output_activation=None)

        self.lr = lr
        self.search_iters = search_iters
        self.tol = tol

#    def parameters(self):
#        for param in self.combine_module.parameters():
 #           yield param
#        for param in self.energy_modules_[0].parameters():
#            yield param

    def forward(self, inputs, targets, mask=None):
        if mask is None:
            mask = inputs.embedding[:,:,0].eq(-1)
        module_energy = torch.cat([module(inputs, targets, mask)
                                   for module in self.energy_modules], 1)
        self.module_energy = module_energy
        return -module_energy.mean(1, keepdim=True)
        #print(module_energy)
        #print(-module_energy.mean(1, keepdim=True))
        #exit()
        #return self.combine_module(module_energy)
        #return self.combine_module(module_energy)

    def search(self, inputs, mask=None, round=True):
        if mask is None:
            mask = inputs.embedding[:,:,0].eq(-1)

        batch_size = inputs.embedding.size(0)
        input_size = inputs.embedding.size(1)

        targets = inputs.embedding.data.new().resize_(batch_size, input_size)
        targets = targets.normal_(.5, .001).masked_fill_(mask.data, 0)
        targets = nn.Parameter(targets)
        opt = torch.optim.Adam([targets], lr=self.lr)

        prev_energy = float("inf")
        for i in range(self.search_iters):
            opt.zero_grad() 
       
            energy = self.forward(inputs, targets, mask)
            avg_energy = energy.mean()
            avg_energy.backward()
            
            opt.step()
            targets.data.clamp_(0,1)
            if math.fabs(prev_energy - avg_energy.data[0]) < self.tol:
                break
            prev_energy = avg_energy.data[0]
        if round:
            return Variable(targets.data.gt(.5).float())
        else:
            return Variable(targets.data)



       

    @property
    def energy_modules(self):
        return self.energy_modules_
