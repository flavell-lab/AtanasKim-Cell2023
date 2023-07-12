module NRRDIO
using CodecZlib, TranscodingStreams, DataStructures

const DICT_DTYPE_R = Dict{String,DataType}("uint8"=>UInt8,
    "int8"=>Int8,
    "uint16"=>UInt16,
    "int16"=>Int16,
    "uin32"=>UInt32,
    "int32"=>Int32,
    "uint64"=>UInt64,
    "int64"=>Int64,
    "float"=>Float32,
    "double"=>Float64,
    
    "unsigned char"=>UInt8,
    "char"=>Int8,
    "unsigned short"=>UInt16,
    "short"=>Int16)

const DICT_DTYPE_W = Dict{DataType,String}(UInt8=>"uint8",
    Int8=>"int8",
    UInt16=>"uint16",
    Int16=>"int16",
    UInt32=>"uin32",
    Int32=>"int32",
    UInt64=>"uint64",
    Int64=>"int64",
    Float32=>"float",
    Float64=>"double");
include("nrrd.jl")
include("util.jl")
export nrrd_header,
    write_nrrd,
    read_header_str,
    nrrd_header,
    NRRD,
    read_img,
    spacing,
    # util.jl
    compress_nrrd
end # module
