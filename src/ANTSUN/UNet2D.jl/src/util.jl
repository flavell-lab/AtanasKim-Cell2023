function standardize(array::Array{<:AbstractFloat,2})
    μ = mean(array)
    σ = std(array)
    (array .- μ) / σ
end

function standardize(array::Array{<:AbstractFloat,3}; dims=(2,3))
    μ = mean(array, dims=dims)
    σ = std(array, dims=dims)
    (array .- μ) ./ σ 
end

reshape_array(array::Array{<:AbstractFloat,2}) = reshape(array, (1,1,size(array)...))
reshape_array(array::Array{<:AbstractFloat,3}) = reshape(array, (size(array,1),1,size(array)[2:3]...))