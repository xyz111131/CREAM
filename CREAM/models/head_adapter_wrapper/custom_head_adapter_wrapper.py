from dataclasses import dataclass
from enformer_pytorch.finetune import HeadAdapterWrapper, get_enformer_embeddings, exists, poisson_loss
import torch

@dataclass
class FeatureInfo:
    name: str
    input_shape: torch.Tensor
    output_shape: torch.Tensor


class CustomHeadAdapterWrapper(HeadAdapterWrapper):
    def __init__(
        self,
        **kwargs,

    ):
        super().__init__(**kwargs)

        # HACK: Add hooks
        # Find and register hook on target layer
        for name, module in self.named_modules():

            #module.register_forward_hook(self._debug_features)

            if name in ["enformer.conv_tower.0", "enformer.conv_tower.1", "enformer.conv_tower.2", "enformer.conv_tower.3", "enformer.conv_tower.4", "enformer.conv_tower.5"]: #enformer.stem.2
                module.register_forward_hook(self._save_features)
        
        self.debug_dict = {}
        self.counter = 0

        self.features_dict = {}


    def _debug_features(self, module, input, output):
        if type(output) == torch.Tensor:
            self.debug_dict[f"{module.name}-{self.counter}"] = FeatureInfo(
                name=module.name,
                # input_shape=input.shape,
                input_shape=None,
                output_shape=output.shape,
            )
        else:
            print(module.name, "output not a Tensor")
            # self.debug_dict[f"{module.name}-{self.counter}"] = (
            #     module.name,
            #     input,
            #     output,
            # )

        self.counter += 1

    def _save_features(self, module, input, output):
        self.features_dict[module.name] = output
    
    def forward(
        self,
        seq,
        *,
        target = None,
        freeze_enformer = False,
        finetune_enformer_ln_only = False,
        finetune_last_n_layers_only = None
    ):
        enformer_kwargs = dict()

        if exists(target) and self.auto_set_target_length:
            enformer_kwargs = dict(target_length = target.shape[-2])

        if self.discrete_key_value_bottleneck:
            embeddings = self.enformer(seq, return_only_embeddings = True, **enformer_kwargs)
        else:
            embeddings = get_enformer_embeddings(self.enformer, seq, freeze = freeze_enformer, train_layernorms_only = finetune_enformer_ln_only, train_last_n_layers_only = finetune_last_n_layers_only, enformer_kwargs = enformer_kwargs)

        #preds = self.to_tracks(embeddings)

        if not exists(target):
            self.features_dict["seq_embeddings"] = embeddings
            # return preds
            return self.features_dict

        return poisson_loss(preds, target)
