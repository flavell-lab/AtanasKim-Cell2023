const ski_morphology = PyNULL()

function __init__()
    copy!(ski_morphology, pyimport_conda("skimage.morphology", "scikit-image"))

    nothing
end