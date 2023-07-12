function compress_nrrd(path_nrrd)
    path_nrrd_backup = splitext(path_nrrd)[1] * ".backup.nrrd"
    mv(path_nrrd, path_nrrd_backup)

    header_str, pos = read_header_str(path_nrrd_backup)
    data = read_img(NRRD(path_nrrd_backup))
    header_str = "NRRD0004\n" * replace(header_str, "raw"=>"gzip")
    n_bytes = write_nrrd(path_nrrd, data, header_str, true)
    
    if read_img(NRRD(path_nrrd_backup)) == data
        rm(path_nrrd_backup)
    end
    
    n_bytes
end
