const py_nd2reader = PyNULL()

function __init__()
    # import nd2reader (https://github.com/JuliaPy/PyCall.jl)
    copy!(py_nd2reader, pyimport_conda("nd2reader", "nd2reader==3.2.3", "conda-forge"))
end
