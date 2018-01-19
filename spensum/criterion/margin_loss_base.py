import ntp
from ntp.criterion import CriterionBase
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .energy_reporter import EnergyReporter
from collections import defaultdict


class MarginLossBase(CriterionBase):

    def __init__(self, weight=None, name="MarginLossBase"):
        super(MarginLossBase, self).__init__(name=name)
        self.tot_margin_ = 0
        self.tot_examples_ = 0
        self.er = EnergyReporter()
        self.bce = ntp.criterion.BinaryCrossEntropy(
            mode="logit", weight=weight, mask_value=-1)
        self.use_margin = True

    def minimize(self, batch, model, opt):
        
        mask = batch.inputs.embedding[:,:,0].eq(-1)
        pred_output = model.search(batch.inputs, mask=mask, round=False) 
        for reporter in self.reporters_:
            if reporter == self:
                continue
            reporter.update(pred_output, batch.targets)


        opt.zero_grad()

        pred_targets = Variable(pred_output.data.gt(.5).float())

        pred_energy = model(batch.inputs, pred_targets, mask=mask)
        self.er.update_pred(model)

        features = model.compute_features(batch.inputs)
        gold_energy = model(batch.inputs, batch.targets, mask=mask,
            precomputed=features)
        self.er.update_gold(model)

        xent_sal = self.bce.eval(features[0], batch.targets)
        xent_cov = self.bce.eval(features[1], batch.targets)
        xent_pos = self.bce.eval(features[2], batch.targets)
        xent_wc = self.bce.eval(features[3], batch.targets)

        margin = gold_energy - pred_energy
        #batch_loss = self.eval(pred_output, batch.targets, model)

#        print(pred_output)
#        print(batch.targets)
#        print(margin)
        self.tot_margin_ += margin.sum().data[0]
        self.tot_examples_ += batch.inputs.embedding.size(0)
        batch_loss = xent_sal + xent_cov + xent_pos + xent_wc
        if self.use_margin:
            batch_loss = batch_loss + margin.mean()
        batch_loss.backward()
        opt.step()
        return batch_loss.data[0]

    def compute_loss(self, batch, model):
        mask = batch.inputs.embedding[:,:,0].eq(-1)
        pred_output = model.search(batch.inputs, mask=mask, round=False)
        #print(pred_output)
        pred_targets = Variable(pred_output.data.gt(.5).float())

        pred_energy = model(batch.inputs, pred_targets, mask=mask)
        self.er.update_pred(model)
        gold_energy = model(batch.inputs, batch.targets, mask=mask)
        self.er.update_gold(model)

        margin = gold_energy - pred_energy
        self.tot_margin_ += margin.sum().data[0]
        self.tot_examples_ += batch.inputs.embedding.size(0)
        batch_loss = margin.mean() 
        for reporter in self.reporters_:
            if reporter == self:
                continue
            reporter.update(pred_output, batch.targets)
        return batch_loss.data[0]

    def reset(self):
        self.er.reset()
        super(MarginLossBase, self).reset()

    def report(self, indent=""):
        blocks = []
        blocks.append({"lines": defaultdict(list), "width": 0, })
        
        current_block = 0
        max_width = 80

        for reporter in self.reporters_ + [self.er]:
            if hasattr(reporter, "report_string"):
                lines, row, cols = reporter.report_string()

                while True:
                    
                    if cols > max_width:
                        current_block = len(blocks)
                        blocks.append({"lines": defaultdict(list), "width": 0})
                        Warning("Overlength reporter: {}!".format(
                            reporter.name))
                        break
                    
                    block_width = blocks[current_block]["width"]
                    if block_width + cols > max_width:
                        current_block += 1
                        if len(blocks) == current_block:
                            blocks.append(
                                {"lines": defaultdict(list), "width": 0})
                    else:
                        break

                blocks[current_block]["width"] += cols + 2
                for i, line in enumerate(lines):
                    blocks[current_block]["lines"][i].append(line)
                    blocks[current_block]["lines"][i].append("  ")

        output_lines = []
        for block in blocks:
            for i in range(len(block["lines"])):
                output_lines.append(indent + "".join(block["lines"][i][:-1]))
        return "\n".join(output_lines)


