from ntp.criterion.criterion_base import CriterionBase
import torch
import torch.nn.functional as F


# TODO add tests for exception throwing.
class RankLoss(CriterionBase):

    def __init__(self, mask_value=None,
                 name="RankLoss"):
        super(RankLoss, self).__init__(name)
        
        self.mask_value_ = mask_value
        self.tot_correct_ = 0
        self.tot_examples_ = 0

    @property
    def initial_value(self):
        return float("-inf")

    def is_better(self, new_value, old_value):
        if new_value > old_value:
            return True
        else:
            return False

    def criterion_value_from_result_dict(self, result_dict):
        return result_dict[self.name]["criterion"]    

    @property
    def mask_value(self):
        return self.mask_value_

    @mask_value.setter
    def mask_value(self, mask_value):
        self.mask_value_ = mask_value

    def reset_statistics(self):

        self.tot_correct_ = 0
        self.tot_examples_ = 0

    def eval(self, output, targets):

        total_elements = targets.size(0) * targets.size(1)
        
        if self.mask_value is not None:
            mask = targets.eq(self.mask_value)
            #output_weight.masked_fill_(mask, 0)
            total_elements -= mask.data[:,:,0].sum()
            targets = targets.masked_fill(mask, 0)
        else:
            mask = None

    #    print(output)
    #    print(targets[:,:,0])
        margin = -5 + output.gather(1, targets[:,:,0]) \
                - output.gather(1, targets[:,:,1])

        total_correct = (margin.data + 5).gt(0).sum()
        batch_loss = -margin.sum() / total_elements
#        print(batch_loss)
        self.tot_examples_ += total_elements
        self.tot_correct_ += total_correct
        return batch_loss
        
    #    exit()


#        if self.weight is not None:
#            if output_weight is None:
#                output_weight = self.weight.new().resize_(targets.size())
#                output_weight.fill_(0)
#
#            pos_mask = torch.eq(targets.data, 1)
#            neg_mask = torch.eq(targets.data, 0)
#            output_weight.masked_fill_(pos_mask, self.weight[1])
#            output_weight.masked_fill_(neg_mask, self.weight[0])
#
#        if self.mode == "prob":
#            total_loss = F.binary_cross_entropy(
#                output, targets, weight=output_weight, 
#                size_average=False)
#        else:
#
#            if output_weight is not None:
#                output_weight = torch.autograd.Variable(output_weight)
#            total_loss = F.binary_cross_entropy_with_logits(
#                output, targets, weight=output_weight,
#                size_average=False)
#
#        batch_loss = total_loss / total_elements
#        self.tot_examples_ += total_elements
#        self.tot_cross_entropy_ += total_loss.data[0]
#
#        return batch_loss

    @property
    def avg_loss(self):
        if self.tot_examples_ > 0:
            return self.tot_correct_ / self.tot_examples_
        else:
            return float("nan")
