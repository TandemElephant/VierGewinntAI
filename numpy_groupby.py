import numpy as np

def groupby_np(X, groups, axis = 0, uf = np.add, out = None, minlength = 0, identity = None):
    # https://stackoverflow.com/questions/49141969/vectorized-groupby-with-numpy
    unique, groups_idx = np.unique(groups, return_inverse=True)
    
    if minlength < groups_idx.max() + 1:
        minlength = groups_idx.max() + 1
    if identity is None:
        identity = uf.identity
    i = list(range(X.ndim))
    del i[axis]
    i = tuple(i)
    n = out is None
    if n:
        if identity is None:  # fallback to loops over 0-index for identity
            assert np.all(np.in1d(np.arange(minlength), groups_idx)), "No valid identity for unassinged groups_idx"
            s = [slice(None)] * X.ndim
            for i_ in i:
                s[i_] = 0
            out = np.array([uf.reduce(X[tuple(s)][groups_idx == i]) for i in range(minlength)])
        else:
            out = np.full((minlength,), identity, dtype = X.dtype)
    uf.at(out, groups_idx, uf.reduce(X, i))
    if n:
        return out

if __name__ == '__main__':
    groups= np.array(['001', '001', '01', '01', '01', '010'])
    nums = np.array([1, 2, 2, 4, 5, -3])
    print(groupby_np(nums, groups, uf=np.minimum))
