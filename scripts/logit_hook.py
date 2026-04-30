# src: https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
# src: https://www.digitalocean.com/community/tutorials/pytorch-hooks-gradient-clipping-debugging
import os
import sys
import torch
from datetime import datetime

if 'OOD_WEATHER' not in os.environ or 'OOD_SEVERITY' not in os.environ:
    raise ValueError(
        "\n\n[!] CRITICAL ERROR: Missing OOD Benchmarking Variables.\n"
        "You must pass OOD_WEATHER and OOD_SEVERITY to the evaluation script.\n"
        "Example: OOD_WEATHER=Snow OOD_SEVERITY=hard ./tools/dist_test.sh ...\n"
    )


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import LOGIT_OUTPUT

weather = os.environ['OOD_WEATHER']
severity = os.environ['OOD_SEVERITY']

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

current_dir = os.path.dirname(os.path.abspath(__file__))
dynamic_save_path = os.path.abspath(
    os.path.join(
        current_dir, 
        '../',
        LOGIT_OUTPUT,
        weather, 
        severity, 
        timestamp
    )
)

os.makedirs(dynamic_save_path, exist_ok=True)


class LogitHook:
    def __init__(self, save_dir=dynamic_save_path):
        self.save_dir = save_dir 
        os.makedirs(self.save_dir, exist_ok=True)
        self.call_counts = {}
        self.handles = []

    def get_logit(self, name):
        def hook(model, input, output):
            pid = os.getpid()
            current_batch = self.call_counts.get(name, 0)

            if current_batch  == 0:
                print("\n HOOKS FIRED.\n")
            
            logits_tensor = None
            if isinstance(output, dict):
                logits_tensor = output.get('all_cls_scores', None)
            elif isinstance(output, tuple) or isinstance(output, list):
                logits_tensor = output[0]

            if logits_tensor is not None:
                final_layer_logits = logits_tensor[-1].detach().cpu()
                
                current_batch = self.call_counts.get(name, 0)
                filename = f"head_final_logits_batch_{current_batch}_gpu_{pid}.pt"
                save_path = os.path.join(self.save_dir, filename)
                
                torch.save(final_layer_logits, save_path)
                
                self.call_counts[name] = current_batch + 1
                
        return hook

    def register_hook(self, model):
        target_name = 'module.pts_bbox_head'
        for name, module in model.named_modules():
            if name == target_name:
                handle = module.register_forward_hook(self.get_logit(name))
                self.handles.append(handle)
                print(f"SUCCESSFULLY ATTACHED HOOK TO: {name}")

    def detach_hook(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

        total_saved = sum(self.call_counts.values())
        print(f"EVALUATION COMPLETE: GPU {os.getpid()} detached its hooks and successfully saved {total_saved} batches.")
