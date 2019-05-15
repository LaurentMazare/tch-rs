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

class Foo1(torch.jit.ScriptModule):
    def __init__(self):
        super(Foo1, self).__init__()

    def forward(self, x, y):
        return 2 * x + y

foo = Foo1()
traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))
traced_foo.save('foo1.pt')

class Foo2(torch.jit.ScriptModule):
    def __init__(self):
        super(Foo2, self).__init__()

    def forward(self, x, y):
        return (2 * x + y, x - y)

foo = Foo2()
traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))
traced_foo.save('foo2.pt')

class Foo3(torch.jit.ScriptModule):
    def __init__(self):
        super(Foo3, self).__init__()

    @torch.jit.script_method
    def forward(self, x):
        result = x[0]
        for i in range(x.size(0)):
            if i: result = result * x[i]
        return result

foo = Foo3()
foo.save('foo3.pt')

