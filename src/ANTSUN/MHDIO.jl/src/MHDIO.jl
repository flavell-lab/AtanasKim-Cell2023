module MHDIO

using DataStructures

const DTYPE_DICT = Dict("MET_SHORT" => Int16, "MET_USHORT" => UInt16,
    "MET_UCHAR" =>UInt8, "MET_DOUBLE"=>Float64);
const DTYPE_DICT_WRITE = Dict([v => k for (k,v) = DTYPE_DICT])


include("mhd.jl")

export MHD,
    read_img,
    write_raw,
    write_MHD_spec

end # module
