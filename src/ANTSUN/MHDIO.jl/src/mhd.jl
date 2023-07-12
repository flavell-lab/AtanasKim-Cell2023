mutable struct MHD
    filename::String
    path_mhd::String
    path_raw::String
    mhd_spec_dict::OrderedDict
    img_shape::Tuple{Vararg{Int,3}}
    img_type::Any

    function MHD(path_mhd)

        str_list = filter(x->length(x)>0, split(read(path_mhd, String), "\n"))

        mhd_spec_dict = OrderedDict()
        for line = str_list
            k, v = split(line, " = ")
            mhd_spec_dict[k] = v
        end

        img_shape = Tuple(parse.(Int, split(mhd_spec_dict["DimSize"], " ")))
        img_type = DTYPE_DICT[mhd_spec_dict["ElementType"]]

        path_raw = replace(path_mhd, ".mhd" => ".raw")
        filename = replace(basename(path_mhd), ".mhd" => "")

        new(filename, path_mhd, path_raw, mhd_spec_dict, img_shape, img_type)
    end
end

function read_img(file_path, img_type, img_dim)

    open(file_path) do f
        read!(f, Array{img_type}(undef, img_dim))
    end
end

"""
    read_img(mhd::MHD)

Read .mhd file into array
"""
function read_img(mhd::MHD)
    read_img(mhd.path_raw, mhd.img_type, mhd.img_shape)
end

"""
	write_raw(file_path, array::Array)

Write .raw file for MHD
"""
function write_raw(file_path, array::Array)
    open(file_path, "w") do f
        write(f, array)
    end
end

"""
	generate_MHD_spec(spacing_lat, spacing_axi, size_x, size_y, size_z,
        filename_raw)

Generate specification/metadata text for MHD


Arguments
---------
* `spacing_lat`: lateral spacing
* `spacing_axi`: axial spacing
* `size_x`: x dimension
* `size_y`: y dimension
* `size_z`: z dimension
* `filename_raw`: .raw file name
"""
function generate_MHD_spec(spacing_lat, spacing_axi, size_x, size_y, size_z,
    filename_raw)
"ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
CompressedData = False
TransformMatrix = 1 0 0 0 1 0 0 0 1
Offset = 0 0 0
CenterOfRotation = 0 0 0
AnatomicalOrientation = RAI
ElementSpacing = $(spacing_lat) $(spacing_lat) $(spacing_axi)
DimSize = $(size_x) $(size_y) $(size_z)
ElementType = MET_USHORT
ElementDataFile = $(filename_raw)"
end

function write_MHD_spec(path, spacing_lat, spacing_axi, size_x, size_y, size_z,
    filename_raw)
    MHD_str = generate_MHD_spec(spacing_lat, spacing_axi, size_x, size_y,
        size_z, filename_raw)
    open(path, "w") do f
        write(f, MHD_str)
    end

end
