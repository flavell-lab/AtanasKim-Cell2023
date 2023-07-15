function import_jld2_data(path_jld2, dict_key="combined_data_dict")
    JLD2.jldopen(path_jld2, "r") do jf
        jf[dict_key]
    end
end
