function import_data(path_h5)
    dict_ = Dict{}()

    h5open(path_h5, "r") do h5f
        for key_ = names(h5f)
            dict_[key_] = read(h5f, key_)
        end
    end

    dict_
end
