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
from constants import FEATURE_LOGIT_OUTPUT

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
                    FEATURE_LOGIT_OUTPUT,
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
                    FEATURE_LOGIT_OUTPUT,
                    weather, 
                    severity, 
                    timestamp
                )
            )

        self.save_dir = dynamic_save_path
        os.makedirs(self.save_dir, exist_ok=True)
        self.call_counts = {}
        self.handles = []

        self.feature_cache = {}

    # TODO: Need to fix this to NOT create multiple folders when using multiple GPUs 
    #       for evaluation.
    def get_features_and_logits(self, name):
        def hook(model, input, output):
            pid = os.getpid()

            self.feature_cache[pid] = {
                'features': input[0].detach().cpu(),
                'logits': output.detach().cpu()
            }
        return hook

    # TODO: Need to fix this to NOT create multiple folders when using multiple GPUs 
    #       for evaluation.
    def get_metadata_and_save(self, name):
        """OUTER HOOK: Grabs the sample_token, merges with cached features, and saves."""
        def hook(model, input, output):
            pid = os.getpid()
            current_batch = self.call_counts.get(name, 0)

            if current_batch  == 0:
                print("\n MASTER LOGIT & FEATURE HOOKS FIRED.\n")

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
            
            if pid in self.feature_cache:
                cached_data = self.feature_cache.pop(pid)
                
                # Saves BOTH features and logits
                save_payload = {
                    'sample_token': sample_tokens[0],
                    'features': cached_data['features'],
                    'logits': cached_data['logits']
                }
                
                token_str = str(sample_tokens[0]).replace('/', '_')
                filename = f"payload_token_{token_str}_gpu_{pid}.pt"
                save_path = os.path.join(self.save_dir, filename)
                
                torch.save(save_payload, save_path)
                
                self.call_counts[name] = current_batch + 1
            else:
                print(f"[!] WARNING: Metadata hook fired but no features found in cache for GPU {pid}!")
                
        return hook

    # TODO: This method gets called multiple times due to multiple GPUs. Need to be fixed.
    def register_hook(self, model):
        inner_target = 'module.pts_bbox_head.cls_branches.5'
        outer_target = 'module.pts_bbox_head'
        
        for name, module in model.named_modules():
            # Attach INNER Hook
            if name == inner_target:
                handle = module.register_forward_hook(self.get_features_and_logits(name))
                self.handles.append(handle)
                print(f"SUCCESSFULLY ATTACHED INNER FEATURE HOOK TO: {name}")
                
            # Attach OUTER Hook
            elif name == outer_target:
                handle = module.register_forward_hook(self.get_metadata_and_save(name))
                self.handles.append(handle)
                print(f"SUCCESSFULLY ATTACHED OUTER METADATA HOOK TO: {name}")

    def detach_hook(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.feature_cache.clear()

        total_saved = sum(self.call_counts.values())
        print(f"EVALUATION COMPLETE: GPU {os.getpid()} detached its hooks and successfully saved {total_saved} batches.")
