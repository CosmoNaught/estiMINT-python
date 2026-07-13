from flax import nnx

# TODO: can props remove residual if block
class MLP(nnx.Module):
    def __init__(self, din, dout, *, width=128, depth=4, residual=False, dropout_rate=0.1, rngs):
        self.residual = residual
        self.inp = nnx.Linear(din, width, rngs=rngs)
        if residual:
            self.norms = nnx.List([nnx.LayerNorm(width, rngs=rngs) for _ in range(depth)])
            self.fc1 = nnx.List([nnx.Linear(width, width, rngs=rngs) for _ in range(depth)])
            self.fc2 = nnx.List([nnx.Linear(width, width, rngs=rngs) for _ in range(depth)])
        else:
            self.hidden = nnx.List([nnx.Linear(width, width, rngs=rngs) for _ in range(depth)])
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.out = nnx.Linear(width, dout, rngs=rngs)

    def __call__(self, x):
        x = self.inp(x)
        if self.residual:
            for ln, f1, f2 in zip(self.norms, self.fc1, self.fc2):
                x = x + self.dropout(f2(nnx.gelu(f1(ln(x)))))  # pre-LN residual block
            x = nnx.gelu(x)
        else:
            x = nnx.gelu(x)
            for h in self.hidden:
                x = self.dropout(nnx.gelu(h(x)))
        return self.out(x)