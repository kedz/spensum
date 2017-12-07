import numpy as np
from collections import defaultdict


class EnergyReporter(object):
    def __init__(self):
        self.pred_energy = defaultdict(list)
        self.gold_energy = defaultdict(list)
        self.name = "EnergyReporter"
        

    def update_pred(self, model):
        for i in range(model.module_energy.size(1)):
            self.pred_energy[i].extend(model.module_energy[:,i].data.tolist())

    def update_gold(self, model):
        for i in range(model.module_energy.size(1)):
            self.gold_energy[i].extend(model.module_energy[:,i].data.tolist())
       
    def reset(self):
        self.pred_output = defaultdict(list)
        self.gold_output = defaultdict(list)

    def result_dict(self):
        rd = {self.name: 
            {i: {"predicted": np.mean(self.pred_energy[i]),
                 "expected": np.mean(self.gold_energy[i])}
             for i in range(len(self.pred_energy))}}
        return rd

    def report_string(self):
        rd = self.result_dict()

        labels = []
        preds = []
        exps = []
        for label in range(len(rd[self.name])):
            pred = "{:0.3f}".format(rd[self.name][label]["predicted"])
            exp = "{:0.3f}".format(rd[self.name][label]["expected"])
            label = str(label)
            col_width = max([len(pred), len(exp), len(label)])
            tmp = "{:" + str(col_width) + "s}"

            labels.append(tmp.format(label))
            preds.append(tmp.format(pred))
            exps.append(tmp.format(exp))

        line1 = "      " + "  ".join(labels)
        line2 = "pred  " + "  ".join(preds)
        line3 = "exptd " + "  ".join(exps)
        lines = [line1, line2, line3]
        return lines, 3, len(line1)
