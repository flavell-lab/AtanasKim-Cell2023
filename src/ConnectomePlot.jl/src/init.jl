
const py_scipy = PyNULL() # pyimport("scipy")
const py_nx = PyNULL() # pyimport("networkx")
const py_copy = PyNULL() # pyimport("copy");

function __init__()
    copy!(py_scipy, pyimport("scipy"))
    copy!(py_nx, pyimport("networkx"))
    copy!(py_copy, pyimport("copy"))

    return nothing
end
