import numpy as np

class PredictionHistogramReporter(object):

    def __init__(self):

        self.bins = 10
        self.pred_output = []
        self.gold_output = []
        self.name = "PredictionHistogram"
        

    
    def update(self, output, expected):
        prob = output.data #.cpu()

        pred_labels = prob.cpu()
        for pred, expected in zip(pred_labels.view(-1),
                                  expected.data.cpu().view(-1)):
            if expected == 0 or expected == 1:
                self.pred_output.append(pred)
                self.gold_output.append(expected)

       
    def reset(self):
        self.pred_output = []
        self.gold_output = []


    def result_dict(self):

        pred_bins, bin_labels = np.histogram(
            self.pred_output, bins=self.bins, range=(0,1))
        gold_bins, _ = np.histogram(
            self.gold_output, bins=self.bins, range=(0,1))

        return {self.name: {"predicted": pred_bins,
                            "expected": gold_bins,
                            "bin_labels": bin_labels}}

    def report_string(self):
        rd = self.result_dict()

        labels = []
        preds = []
        exps = []
        for pred, exp, label in zip(rd[self.name]["predicted"],
                                    rd[self.name]["expected"],
                                    rd[self.name]["bin_labels"]):
            pred = str(pred)
            exp = str(exp)
            label = "{:0.3f}".format(label)
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
