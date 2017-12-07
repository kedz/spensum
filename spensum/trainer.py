import torch
import ntp
from .criterion import SquaredLossAugmentedHingeLoss
from .criterion import PredictionHistogramReporter


def pretrain_module(module, train_dataset, valid_dataset, lr, max_epochs,
                    module_save_path):
    non_salient_count = train_dataset.targets.eq(0).sum()
    salient_count = train_dataset.targets.eq(1).sum()
    weight = torch.FloatTensor([1 / non_salient_count, 1 / salient_count])
    print("Training data:")
    print("# salient = {}".format(salient_count))
    print("# non-salient = {}".format(non_salient_count))

    opt = ntp.optimizer.Adam(module.parameters(), lr=lr)
    crit = ntp.criterion.BinaryCrossEntropy(
        mode="prob", weight=weight, mask_value=-1)
    crit.add_reporter(
        ntp.criterion.BinaryFMeasureReporter(mode="prob"))
    crit.set_selection_criterion("BinaryFMeasureReporter")

    ntp.trainer.optimize_criterion(crit, module, opt, train_dataset,
                                   validation_data=valid_dataset,
                                   max_epochs=max_epochs,
                                   save_model=module_save_path)
    
    print("Restoring model to best epoch...")
    best_module = torch.load(module_save_path)
    return best_module

def train_model(model, train_dataset, valid_dataset, lr, max_epochs,
                model_save_path):
    non_salient_count = train_dataset.targets.eq(0).sum()
    salient_count = train_dataset.targets.eq(1).sum()
    weight = torch.FloatTensor([1 / non_salient_count, 1 / salient_count])
    print("Training data:")
    print("# salient = {}".format(salient_count))
    print("# non-salient = {}".format(non_salient_count))

    opt = ntp.optimizer.Adam(model.parameters(), lr=lr)

    crit = SquaredLossAugmentedHingeLoss()
    #crit = ntp.criterion.BinaryCrossEntropy(
    #    mode="prob", weight=weight, mask_value=-1)
    crit.add_reporter(
        ntp.criterion.BinaryFMeasureReporter(mode="prob"))
    crit.add_reporter(
        PredictionHistogramReporter())
    crit.set_selection_criterion("BinaryFMeasureReporter")

    ntp.trainer.optimize_criterion(crit, model, opt, train_dataset,
                                   validation_data=valid_dataset,
                                   max_epochs=max_epochs,
                                   save_model=model_save_path)
    
    print("Restoring model to best epoch...")
    best_module = torch.load(model_save_path)
    return best_module
