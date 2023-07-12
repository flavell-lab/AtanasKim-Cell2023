"""
    calculate_ncc(moving, fixed)

Computes the NCC of two image arrays `moving` and `fixed` corresponding to a registration.
"""
function calculate_ncc(moving, fixed)
    @assert(size(fixed) == size(moving))
    
    med_f = median(maximum(fixed, dims=3))
    med_m = median(maximum(moving, dims=3))
    fixed_new = map(x->max(x,0), fixed .- med_f)
    moving_new = map(x->max(x,0), moving .- med_m)
    
    mu_f = Statistics.mean(fixed_new)
    mu_m = Statistics.mean(moving_new)
    fixed_new = fixed_new ./ mu_f .- 1
    moving_new = moving_new ./ mu_m .- 1
   
    sum(fixed_new .* moving_new) /
    sqrt(sum(fixed_new .^ 2) * sum(moving_new .^ 2))
end

"""
    metric_tfm(ncc; threshold=0.9)

Applies a function to `ncc` to make it a cost that increases to infinity if `ncc` decreases below `threshold` (default 0.9)
"""
function metric_tfm(ncc; threshold=0.9)
    if ncc <= threshold + 1e-6
        return Inf
    end
    return (1 - ncc) / (ncc - threshold)
end

