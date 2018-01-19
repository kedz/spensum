from .margin_loss_base import MarginLossBase

class SquaredLossAugmentedHingeLoss(MarginLossBase):

    def __init__(self, weight=None, name="SquaredLossAugmentedHingeLoss"):
        super(SquaredLossAugmentedHingeLoss, self).__init__(name=name, weight=weight)

        self.tot_margin_ = 0
        self.tot_examples_ = 0

    @property
    def initial_value(self):
        return float("inf")

    def is_better(self, new_value, old_value):
        if new_value < old_value:
            return True
        else:
            return False

    def criterion_value_from_result_dict(self, result_dict):
        return result_dict[self.name]["margin"]    

    def reset_statistics(self):

        self.tot_margin_ = 0
        self.tot_examples_ = 0

    @property
    def avg_loss(self):
        if self.tot_examples_ > 0:
            return self.tot_margin_ / self.tot_examples_
        else:
            return float("nan")

    def eval(self, output, targets, model):
        raise NotImplementedError("Ahhh")
