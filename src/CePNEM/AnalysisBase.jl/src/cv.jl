"""
Generate 2-way continuous cross-validation splits. Returns list of [train, test]
# Arguments:
- `idx_splits`: list of time points for videos. e.g. if 2 videos of total length 1600 splitted at 800, `[1:800, 801:1600]`
- `trim`: time points to trim in the beginning of each video
- `frac_train`: training set fraction (default=0.7)
"""

function continuous_cv_split(idx_splits, trim::Int=50, frac_train=0.7)
    n_trim = trim * (length(idx_splits) + 1)
    t_max = idx_splits[end][end]
    n_trimmed = t_max - n_trim
    
    n_train = round(Int, frac_train * n_trimmed)
    n_test = round(Int, (1 - frac_train) * n_trimmed)
    
    @assert(n_train + n_test == n_trimmed)
    
    idx_splits_trim = vcat(trim_idx_splits(idx_splits, (trim,0))...)
    
    list_split = Vector{Vector{Int}}[]
    train = idx_splits_trim[1:n_train]
    test = idx_splits_trim[n_train+51:n_train+50+n_test]
    push!(list_split, [train, test])
    
    test = idx_splits_trim[1:n_test]
    train = idx_splits_trim[n_test+51:n_test+50+n_train]
    push!(list_split, [train, test])
    
    list_split
end

"""
Generate K fold cross-validation splits. Returns list of [train, test]
# Arguments:
- `k`: K fold splits
- `idx_splits`: list of time points for videos. e.g. if 2 videos of total length 1600 splitted at 800, `[1:800, 801:1600]`
- `trim`: number of time points to trim for ewma. default: 50
- `gap`: gap between train and test. default: 50
"""
function kfold_cv_split(k, idx_splits, trim::Int=50, gap::Int=50)
    list_t_split_trim = []
    for i = 1:length(idx_splits)
        rg = idx_splits[i]
        rg = (rg[1]+trim[1]):rg[end]
        push!(list_t_split_trim, rg)
    end
    rg_combined = union(list_t_split_trim...)

    list_split = Vector{Vector{Int64}}[]
    for split = Base.Iterators.partition(rg_combined, round(Int, length(rg_combined) / k))
        idx_test = split
        idx_train = setdiff(rg_combined, split)
        
        # train begin
        idx_test_gap = idx_test[1]:(idx_test[1]+round(Int, gap/2)-1)
        idx_train_gap = (idx_test[1]-round(Int, gap/2)):(idx_test[1]-1)
        idx_test = setdiff(idx_test, idx_test_gap)
        idx_train = setdiff(idx_train, idx_train_gap)
        
        # train end
        idx_test_gap = (idx_test[end]-round(Int, gap/2)+1):idx_test[end]
        idx_train_gap = (idx_test[end]+1):(idx_test[end]+round(Int, gap/2))
        idx_test = setdiff(idx_test, idx_test_gap)
        idx_train = setdiff(idx_train, idx_train_gap)
        
        # println("train: $(length(idx_train)) test: $(length(idx_test))")
        push!(list_split, [idx_train, idx_test])
    end
    
    list_split
end

"""
Generates uniformly-spaced cross-validation splits, subject to the constraint that
the testing data must be contiguous.

# Arguments:
- `t_range`: Time points to use for training and testing
- `n_splits`: Number of cross-validation splits
- `train_frac`: Fraction of data that should be testing
"""
function generate_cv_splits(t_range, n_splits, train_frac)
    @assert(0 < train_frac <= 1, "Training fraction must be between 0 and 1.")
    @assert(n_splits > 1, "Must have at least two splits.")
    splits = []
    train_len = Int32(ceil(train_frac * length(t_range)))
    test_len = length(t_range) - train_len
    loc = 1
    spacing = (length(t_range) - test_len) / (n_splits - 1)
    for i=1:n_splits
        test_rng = t_range[loc:loc+test_len-1]
        train_rng = [t for t in t_range if !(t in test_rng)]
        push!(splits, (train_rng, test_rng))
        loc += spacing
    end
    return splits
end

"""
Adds sampling to a `training set` with a certain `sample_density` at time points in `t_range`.
"""
function add_sampling(training_set, sample_density, t_range)
    len_train = length(training_set)
    len_test = length(t_range) - len_train
    frac_train = len_train / length(t_range)
    frac_sample = frac_train * sample_density
    n_sample = 1 / frac_sample
    new_train = deepcopy(training_set)
    count = 1
    for t=t_range
        if count > n_sample
            count = 1
            push!(new_train, t)
        end
        count += 1
    end
    new_test = [t for t in t_range if !(t in training_set)]
    return (new_train, new_test)
end

"""
Removes beginning of each video from total timepoints `t_range` given video splits `idx_splits`,
    to avoid `ewma` issues.
"""
function rm_dataset_begin(t_range, idx_splits; n_remove=50)
    return setdiff(t_range, union(map(x->x[1]:x[1]+(n_remove-1), idx_splits)...))
end

function trim_idx_splits(idx_splits::Vector{UnitRange{Int64}}, trim=(50,0))
    idx_splits_trim = UnitRange{Int64}[]
    for i = 1:length(idx_splits)
        start = idx_splits[i][1] + trim[1]
        stop =  idx_splits[i][end] - trim[2]
        push!(idx_splits_trim, start:stop)
    end

    idx_splits_trim
end

        
"""
Removes splits with low training behavior variation.

# Arguments:
- `splits`: Proposed train/test splits
- `behaviors`: Behaviors
- `variation_thresh`: Fraction of total variation that must be included in training set.
- `varation_function` (default `cost_rss`): Function to use to compute variation.
"""
function rm_low_variation(splits, behaviors, variation_thresh; variation_fn=cost_rss)
    new_splits = []
    for split in splits
        train = splits[1]
        if !any([variation_fn(behavior[train], mean(behavior)) < variation_thresh * variation_fn(behavior, mean(behavior))
                for behavior in behaviors])
            push!(new_splits, split)
        end
    end
    return new_splits
end


