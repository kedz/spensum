import ntp
import colorama
from ntp.criterion import CriterionBase
from collections import OrderedDict
from .prediction_histogram_reporter import PredictionHistogramReporter
from torch.autograd import Variable


class SPENLoss(CriterionBase):

    def __init__(self, weight=None, name="SPENLoss", mask_value=-1):
        super(SPENLoss, self).__init__(name=name)
        self.mask_value = mask_value
        self.tot_margin_ = 0
        self.tot_aux_loss_ = 0
        self.tot_examples_ = 0
        self.aux_crit = OrderedDict()
        self.aux_crit_weights = []

        self.pred_histogram = PredictionHistogramReporter() 
       #if self.aux_objectives is None:
        #    self.aux_objectives = []

#        self.bce = ntp.criterion.BinaryCrossEntropy(
#            mode="logit", weight=weight, mask_value=-1)
#        self.use_margin = True

    def criterion_value_from_result_dict(self):
        pass
    
    def eval(self):
        pass
    def initial_value(self):
        pass
    def is_better(self):
        pass
    
    def reset_statistics(self):
        self.tot_margin_ = 0
        self.tot_examples_ = 0

    def reset(self):
        self.pred_histogram.reset()
        for reporter in self.reporters_:
            if reporter == self:
                continue
            reporter.reset()
        for crit in self.aux_crit.values():
            crit.reset()

    def add_aux_criterion(self, module, crit, weight=1.0):

        self.aux_crit[module.name] = crit
        self.aux_crit_weights.append(weight)

    def has_aux_criterion(self, module):
        return module.name in self.aux_crit

    def get_aux_criterion(self, module):
        return self.aux_crit.get(module.name, None)

    def minimize(self, batch, model, opt):
    
        targets_mask = batch.targets.eq(self.mask_value)
        pred_targets_soft = model.search(
            batch.inputs, targets_mask=targets_mask, round=False) 

        self.pred_histogram.update(pred_targets_soft, batch.targets)
        pred_targets = Variable(pred_targets_soft.data.ge(.5).float())

        for rep in self.reporters_:
            if rep != self:
                rep.update(pred_targets, batch.targets)

        opt.zero_grad()

        features = model.compute_features(batch.inputs)
        
        gold_energy = model.forward(
            batch.inputs, batch.targets, targets_mask=targets_mask, 
            precomputed=features)
        pred_energy = model.forward(
            batch.inputs, pred_targets, targets_mask=targets_mask, 
            precomputed=features)
        margin = 1 + gold_energy - pred_energy
        
        aux_losses = []
        for m, f in zip(model.submodules, features):
            aux_crit = self.get_aux_criterion(m)
            if aux_crit:
                outputs = m.forward_pass(batch.inputs, f)
                aux_losses.append(aux_crit.eval(outputs, batch.targets))
                for rep in aux_crit.reporters_:
                    if rep != aux_crit:
                        rep.update(outputs, batch.targets)
        

        margin_loss = margin.mean()
        aux_loss = sum([w * l for w , l in zip(
            self.aux_crit_weights, aux_losses)])
        batch_loss = margin_loss + aux_loss
        batch_loss.backward()
        opt.step()

        batch_size = batch.inputs.embedding.size(0)
        self.tot_aux_loss_ += aux_loss.data[0] * batch_size
        self.tot_margin_ += margin.data.sum()
        self.tot_examples_ += batch_size
        return batch_loss.data[0] 

    def compute_loss(self, batch, model, opt):
    
        targets_mask = batch.targets.eq(self.mask_value)
        pred_targets_soft = model.search(
            batch.inputs, targets_mask=targets_mask, round=False) 

        self.pred_histogram.update(pred_targets_soft, batch.targets)
        pred_targets = Variable(pred_targets_soft.data.ge(.5).float())

        for rep in self.reporters_:
            if rep != self:
                rep.update(pred_targets, batch.targets)

        features = model.compute_features(batch.inputs)
        
        gold_energy = model.forward(
            batch.inputs, batch.targets, targets_mask=targets_mask, 
            precomputed=features)
        pred_energy = model.forward(
            batch.inputs, pred_targets, targets_mask=targets_mask, 
            precomputed=features)
        margin = 1 + gold_energy - pred_energy
        
        aux_losses = []
        for m, f in zip(model.submodules, features):
            aux_crit = self.get_aux_criterion(m)
            if aux_crit:
                outputs = m.forward_pass(batch.inputs, f)
                aux_losses.append(
                    aux_crit.eval(outputs, batch.targets))
                for rep in aux_crit.reporters_:
                    if rep != aux_crit:
                        rep.update(outputs, batch.targets)
        

        margin_loss = margin.mean()
        aux_loss = sum([w * l for w , l in zip(
            self.aux_crit_weights, aux_losses)])
        batch_loss = margin_loss + aux_loss

        batch_size = batch.inputs.embedding.size(0)
        self.tot_aux_loss_ += aux_loss.data[0] * batch_size
        self.tot_margin_ += margin.data.sum()
        self.tot_examples_ += batch_size
        return batch_loss.data[0] 


    @property
    def margin_loss(self):
        if self.tot_examples_ == 0:
            return float("nan")
        else:
            return self.tot_margin_ / self.tot_examples_

    @property
    def aux_loss(self):
        if self.tot_examples_ == 0:
            return float("nan")
        else:
            return self.tot_aux_loss_ / self.tot_examples_


    def report(self, indent=""):

        lines = [indent + colorama.Style.BRIGHT + \
            "Auxiliary Criterion"+ colorama.Style.NORMAL]
        for name, crit in self.aux_crit.items():
            lines.append(indent + name)
            lines.append(crit.report(indent=indent))
            lines.append("")

        lines.append("")
        lines.append(
            indent + colorama.Style.BRIGHT + \
            "Prediction Histogram"+ colorama.Style.NORMAL)
        for line in self.pred_histogram.report_string()[0]:
            lines.append(indent + line)

        lines.append(super(SPENLoss, self).report(indent=indent))
        lines.append("")

        return "\n".join(lines)

    @property
    def avg_loss(self):
        return self.margin_loss + self.aux_loss
