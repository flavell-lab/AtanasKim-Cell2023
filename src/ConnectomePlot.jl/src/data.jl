# connectome
const path_connectome_white = joinpath(@__DIR__, "..", "data", "white_1986_whole.json")
const data_connectome_white = JSON.parsefile(path_connectome_white, dicttype=Dict, inttype=Int64)
const path_connectome_witvliet_7 = joinpath(@__DIR__, "..", "data", "witvliet_2020_7.json")
const data_connectome_witvliet_7 = JSON.parsefile(path_connectome_witvliet_7, dicttype=Dict, inttype=Int64)
const path_connectome_witvliet_8 = joinpath(@__DIR__, "..", "data", "witvliet_2020_8.json")
const data_connectome_witvliet_8 = JSON.parsefile(path_connectome_witvliet_8, dicttype=Dict, inttype=Int64)

# 2d embedding data
const path_connectome_plot = joinpath(@__DIR__, "..", "data", "connectome_diagram_pos.h5")
const path_connectome_p_plot = joinpath(@__DIR__, "..", "data", "connectome_diagram_pharynx_pos.h5")
const path_connectome_combined_plot = joinpath(@__DIR__, "..", "data", "connectome_diagram_combined_pos.h5")

const dict_pos_z_non_p = h5read(path_connectome_plot, "z")
const dict_pos_v2_non_p = h5read(path_connectome_plot, "v2")
const dict_pos_v3_non_p = h5read(path_connectome_plot, "v3")
const dict_pos_z_p = h5read(path_connectome_p_plot, "z")
const dict_pos_v2_p = h5read(path_connectome_p_plot, "v2")
const dict_pos_v3_p = h5read(path_connectome_p_plot, "v3")
const dict_pos_combined_z = h5read(path_connectome_combined_plot, "z")
const dict_pos_combined_v2 = h5read(path_connectome_combined_plot, "z")
const dict_pos_combined_v3 = h5read(path_connectome_combined_plot, "z")

# witvliet neuron type reference
const path_witvliet_type = joinpath(@__DIR__, "..", "data", "witvliet_table_s1.csv")
const witvliet_type = let
    witvliet_s1 = readdlm(path_witvliet_type, ',');
    witvliet_type = Dict{String,String}()
    for i = 1:(size(witvliet_s1,1)-1)
        neuron = witvliet_s1[i+1,1]
        type_ = witvliet_s1[i+1,3]
        witvliet_type[neuron] = type_
    end

    witvliet_type
end

function get_dict_pos_patched(Δ_v2=0.032, scale_v2=0.05)
    dict_pos_z_patched = Dict{String,Float64}()
    dict_pos_v2_patched = Dict{String,Float64}()

    for (k,z) = dict_pos_combined_z
        dict_pos_z_patched[k] = z

        if haskey(dict_pos_z_p, k)
            dict_pos_v2_patched[k] = Δ_v2 + scale_v2 * dict_pos_v2_p[k]
        elseif haskey(dict_pos_z_non_p, k)
            dict_pos_v2_patched[k] = dict_pos_v2_non_p[k]
        else
            println("neuron $k is missing in the position dictionary")
        end
    end
    
    dict_pos_v2_patched, dict_pos_z_patched
end