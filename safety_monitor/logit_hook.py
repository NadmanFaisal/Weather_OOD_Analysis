"""
Researcher(s): Hasan Zahid, Nadman Abdullah Bin Faisal, Vaibhav Puram

Artifact: 
PyTorch Forward Logit Interceptor (BEVFormer)
Methodology: Design Science Research (Cycle II: Solution Design)

Purpose:
This utility acts as a PyTorch forward hook designed to intercept intermediate 
tensor outputs from the BEVFormer architecture during the evaluation phase. 
It targets the `pts_bbox_head` (the Deformable DETR decoder) to extract the 
final-layer classification logits (`all_cls_scores`) before they undergo activation 
functions.

By pairing these raw logits with their corresponding nuScenes `sample_token` 
and dynamically routing the saved `.pt` payloads into structured directories 
based on terminal environment variables, this script builds the raw data pipeline 
required for downstream Out-of-Distribution (OOD) energy scoring and AUROC benchmarking.

NOTE 1: To prevent catastrophic GPU memory leaks (OOM errors) during large-scale 
dataset evaluation, intercepted tensors are explicitly detached from the 
computation graph and migrated to the CPU (`.detach().cpu()`) prior to serialization.

NOTE 2: The script currently mitigates multi-GPU race conditions by appending 
the OS process ID (`pid`) to the output filenames, ensuring concurrent workers 
do not overwrite each other's intercepted frames.
"""

# src: https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
# src: https://www.digitalocean.com/community/tutorials/pytorch-hooks-gradient-clipping-debugging
import os
import sys
import torch
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import LOGIT_OUTPUT

class LogitHook:
    def __init__(self):

        # Reads the env vars provided in the CMD to create necessary folders to store logits
        if 'OOD_WEATHER' not in os.environ or 'OOD_SEVERITY' not in os.environ:
            raise ValueError(
                "\n\n[!] CRITICAL ERROR: Missing OOD Benchmarking Variables.\n"
                "You must pass OOD_WEATHER and OOD_SEVERITY to the evaluation script.\n"
                "Example: OOD_WEATHER=Snow OOD_SEVERITY=hard ./tools/dist_test.sh ...\n"
            )

        # Reads the variables
        weather = os.environ['OOD_WEATHER']
        severity = os.environ['OOD_SEVERITY']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # If the variables indicate that it is nuscenes dataset (clean)
        if weather == 'Clear' and severity == 'baseline':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            current_dir = os.path.dirname(os.path.abspath(__file__))
            dynamic_save_path = os.path.abspath(
                os.path.join(
                    current_dir, 
                    '../',
                    LOGIT_OUTPUT,
                    'nuscenes/',
                    timestamp
                )
            )

        # If corrupted data are provided (Fog, snow, or smthng else)
        else:
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

        self.save_dir = dynamic_save_path
        os.makedirs(self.save_dir, exist_ok=True)
        self.call_counts = {}
        self.handles = []

    # TODO: Need to fix this to NOT create multiple folders when using multiple GPUs 
    #       for evaluation.
    def get_logit(self, name):
        def hook(model, input, output):
            pid = os.getpid()
            current_batch = self.call_counts.get(name, 0)

            # Prints once when hook is fired
            if current_batch  == 0:
                print("\n HOOKS FIRED.\n")

            # Saves sample token of each frame to trace back where the logits are coming from
            sample_tokens = []
            
            for arg in input:
                if isinstance(arg, list) and len(arg) > 0 and isinstance(arg[0], dict):
                    if 'sample_idx' in arg[0] or 'token' in arg[0]:
                        for meta in arg:
                            token = meta.get('sample_idx', meta.get('token', f'unknown_batch_{current_batch}'))
                            sample_tokens.append(token)
                        break
            
            if not sample_tokens:
                sample_tokens = [f'unknown_batch_{current_batch}']
            
            # Extracts the logits from the output detection head (defined in forward() method in bevformer's head file)
            logits_tensor = None
            if isinstance(output, dict):
                logits_tensor = output.get('all_cls_scores', None)
            elif isinstance(output, tuple) or isinstance(output, list):
                logits_tensor = output[0]
            
            # Loads the logits into the CPU to save GPU from expanding too mucn
            if logits_tensor is not None:

                # Extract logit outputs from the last layer of the detection head
                final_layer_logits = logits_tensor[-1].detach().cpu()

                # Saves the data
                save_payload = {
                    'sample_token': sample_tokens[0],
                    'logits': final_layer_logits
                }
                
                token_str = str(sample_tokens[0]).replace('/', '_')
                filename = f"logits_token_{token_str}_gpu_{pid}.pt"
                save_path = os.path.join(self.save_dir, filename)
                
                torch.save(save_payload, save_path)
                
                self.call_counts[name] = current_batch + 1
                
        return hook

    # TODO: This method gets called multiple times due to multiple GPUs. Need to be fixed.
    def register_hook(self, model):

        # Attach the hooks to the entire detection head
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
