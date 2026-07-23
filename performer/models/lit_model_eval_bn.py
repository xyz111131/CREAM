from performer.models.lit_model import LitModel


import torch


class LitModelEvalBN(LitModel):
    """
    Overwrite LitModel to freeze batchnorm updates. Can be useful if not useing pytorch-enformer's HeadAdapterWrapper, which handles this for you
    """

    def __init__(
        self,
        model,
        tissues_to_train,
        save_dir,
        valid_metrics_save_freq,
        train_dataset,
        target_learning_rate,
        loss_func_str,
        alpha,
        genes_for_training,
        genes_for_valid,
        loss_funcs,
        genes_for_test,
    ):
        super().__init__(
            model,
            tissues_to_train,
            save_dir,
            valid_metrics_save_freq,
            train_dataset,
            target_learning_rate,
            loss_func_str,
            alpha,
            genes_for_training,
            genes_for_valid,
            loss_funcs,
            genes_for_test,
        )
        self.freeze_batchnorm(self.model)

    def freeze_batchnorm(self, model):
        for module in model.modules():
            if (
                isinstance(module, torch.nn.BatchNorm1d)
                or isinstance(module, torch.nn.BatchNorm2d)
                or isinstance(module, torch.nn.BatchNorm3d)
            ):
                module.eval()

    def train(self, mode=True):  # overwrite model.train() to keep batchnorm layers in eval mode
        super().train(mode)  # Sets the model to training mode like usual
        self.freeze_batchnorm(self.model)  # Revert BatchNorm layers to eval mode
        return self
