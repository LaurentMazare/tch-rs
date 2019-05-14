import torch

class Foo(torch.jit.ScriptModule):
    def __init__(self, v):
        super(Foo, self).__init__()
        self.register_buffer('value', v)

    @torch.jit.script_method
    def forward(self, x, y):
        return 2 * x + y + self.value

foo = Foo(torch.Tensor([42.0]))
foo.save('foo.pt')
