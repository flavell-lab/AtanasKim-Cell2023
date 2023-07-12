function create_model(n_ch_input, n_class, n_feature_init, path_weights=nothing; device=torch_device, eval_mode=true)
    model = py_unet2d.model.unet_model.UNet2D(n_ch_input, n_class, n_feature_init, false)
    
    model.to(device)
    !isnothing(path_weights) && model.load_state_dict(py_torch.load(path_weights, map_location=device))
    eval_mode && model.eval()
    
    return model
end

function eval_model(img::Union{Array{Float32,2}, Array{Float32,3}}, model; device=torch_device)
    img_x = reshape_array(img)
    img_x = py_torch.from_numpy(img_x).to(device)
    
    @pywith py_torch.no_grad() begin
        img_y_pred = model(img_x)
        img_y_pred = py_torch.sigmoid(img_y_pred)
        img_y_pred = img_y_pred.cpu().numpy()
        
        if size(img_y_pred)[1] == 1
            return img_y_pred[1,1,:,:]
        else
            return img_y_pred[:,1,:,:]
        end
    end
end
