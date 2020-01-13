'''
MIT License

Copyright (c) 2020 gentaman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from collections import OrderedDict, Sequence
import numpy as np
import torch

class StackValueWrapper:
    """
    save values.
    """
    def __init__(self, model, hook_name, candidate_layers=None, replace=True):
        '''
            model is type of torch.Module
            hook_name from [forward_out, backward_out, forward_in, backward_in]
            If any candidates are not specified, the hook is registered to all the layers.
        '''
        self.model = model
        self.pool = OrderedDict()
        self.candidate_layers = candidate_layers  # list
        self.replace = replace
        self.hook_name = hook_name
        self.handlers = []

        def forward_hook(key):
            def forward_hook_(module, input, output):
                # Save featuremaps
                if isinstance(output, dict):
                    ls = input[1]
                    out = output[ls[0]].detach()
                else:
                    out = output.detach()
                
                if self.replace or key not in self.pool:
                    self.pool[key] = out
                else:
                    tmp_key = key
                    while(tmp_key in self.pool):
                        tmp_key = tmp_key + '_r'
                    self.pool[tmp_key] = out

            return forward_hook_
        
        def forward_in_hook(key):
            def forward_hook_(module, input, output):
                # Save featuremaps
                out = [i.detach() for i in input]
                
                if self.replace or key not in self.pool:
                    self.pool[key] = out
                else:
                    tmp_key = key
                    while(tmp_key in self.pool):
                        tmp_key = tmp_key + '_r'
                    self.pool[tmp_key] = out

            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                # Save the gradients correspond to the featuremaps
                if self.replace or key not in self.pool:
                    self.pool[key] = grad_out[0].detach()
                else:
                    tmp_key = key
                    while(tmp_key in self.pool):
                        tmp_key = tmp_key + '_r'
                    self.pool[tmp_key] = grad_out[0].detach()

            return backward_hook_

        def backward_in_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                # Save featuremaps
                out = [i.detach() for i in filter(lambda x: x is not None, grad_in)]
                
                if self.replace or key not in self.pool:
                    self.pool[key] = out
                else:
                    tmp_key = key
                    while(tmp_key in self.pool):
                        tmp_key = tmp_key + '_r'
                    self.pool[tmp_key] = out

            return backward_hook_

        if hook_name == 'forward_out':
            hook = forward_hook
        elif hook_name == 'backward_out':
            hook = backward_hook
        elif hook_name == 'forward_in':
            hook = forward_in_hook
        elif hook_name == 'backward_in':
            hook = backward_in_hook
        else:
            raise ValueError("Unknow hook name: {}".format(hook_name))

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                if 'forward' in hook_name:
                    self.handlers.append(module.register_forward_hook(hook(name)))
                if 'backward' in hook_name:
                    self.handlers.append(module.register_backward_hook(hook(name)))
                
    def remove(self):
        del self.pool
        self.pool = OrderedDict()
        self.remove_hook()
        torch.cuda.empty_cache()

    def _find(self, target_layer):
        if target_layer in self.pool.keys():
            return self.pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()
