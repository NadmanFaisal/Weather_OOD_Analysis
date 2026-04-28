# src: https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
# src: https://www.digitalocean.com/community/tutorials/pytorch-hooks-gradient-clipping-debugging

class LogitHook:
    def __init__(self):
        self.logits = {}
        self.handles = []
        self._has_printed_layers = False 

    def get_logit(self, name):
        def hook(model, input, output):
            self.logits[name] = output.detach().cpu()
        return hook

    def register_hook(self, model, num_decoder_layers=6):
        if not self._has_printed_layers:
            with open('tools/bevformer_layers.txt', 'w') as f:
                for name, module in model.named_modules():
                    f.write(f"{name}\n")
            print("SUCCESS: Saved all layer names to 'tools/bevformer_layers.txt'")
            self._has_printed_layers = True

        for i in range(num_decoder_layers):
            target_name = f'module.pts_bbox_head.cls_branches.{i}'
            
            for name, module in model.named_modules():
                if name == target_name:
                    handle = module.register_forward_hook(self.get_logit(name))
                    self.handles.append(handle)
                    print(f"Attached hook to {name}")

    def detach_hook(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        print("All hooks removed.")
