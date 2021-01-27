import numpy as np

def groupby_np(X, groups, axis = 0, uf = np.add, out = None, minlength = 0, identity = None):
    if minlength < groups.max() + 1:
        minlength = groups.max() + 1
    if identity is None:
        identity = uf.identity
    i = list(range(X.ndim))
    del i[axis]
    i = tuple(i)
    n = out is None
    if n:
        if identity is None:  # fallback to loops over 0-index for identity
            assert np.all(np.in1d(np.arange(minlength), groups)), "No valid identity for unassinged groups"
            s = [slice(None)] * X.ndim
            for i_ in i:
                s[i_] = 0
            out = np.array([uf.reduce(X[tuple(s)][groups == i]) for i in range(minlength)])
        else:
            out = np.full((minlength,), identity, dtype = X.dtype)
    uf.at(out, groups, uf.reduce(X, i))
    if n:
        return out