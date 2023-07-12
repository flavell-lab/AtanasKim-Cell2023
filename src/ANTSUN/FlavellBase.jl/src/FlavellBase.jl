module FlavellBase

using Statistics

include("io.jl")
include("array.jl")
include("dict.jl")

export read_txt,
    write_txt,
    create_dir,

    # array.jl
    standardize,
    maxprj,
    meanprj,
    minprj,
    stdprj,
    medianprj,
    map_data,
    aggregate_var,
    rescale_to_range,
    # dict.jl
    add_list_dict!

end # module
