const py_ski_morphology = PyNULL()
const py_skl_neighbors = PyNULL()
const py_nx = PyNULL()

function __init__()
    copy!(py_ski_morphology, pyimport_conda("skimage.morphology","scikit-image",
            "conda-forge"))
    copy!(py_skl_neighbors, pyimport_conda("sklearn.neighbors", "scikit-learn",
                "conda-forge"))
    copy!(py_nx, pyimport_conda("networkx", "networkx"))
end