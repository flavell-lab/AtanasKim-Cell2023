"""
    rank_test(ground_truth, inferred, n_bins; print_extremes=false, normalize=true)

This function computes the rank test for a given ground truth and inferred values. It returns the ranks for each bin and each column of the inferred matrix.

Arguments:
- `ground_truth`: a matrix of size (n_samples, n_features) representing the ground truth values.
- `inferred`: a tensor of size (n_samples, n_inferred, n_features) representing the inferred values from the model fit.
- `n_bins`: an integer representing the number of bins to use for the rank test.
- `print_extremes`: a boolean indicating whether to print the ground truth values that are at the extremes of the inferred values.
- `normalize`: a boolean indicating whether to normalize the ranks by the number of samples.

Returns:
- `ranks`: a matrix of size (n_bins, n_features) representing the ranks for each bin and each column of the inferred matrix.
"""
function rank_test(ground_truth, inferred, n_bins; print_extremes=false, normalize=true)
    ranks = zeros(n_bins, size(ground_truth, 2))
    for i=1:size(ground_truth,1)
        for j=1:size(ground_truth,2)
            L = sum(ground_truth[i,j] .< inferred[i,:,j])
            if ((L == 0) || (L == length(inferred[i,:,j]))) && print_extremes
                println(ground_truth[i,:])
            end
            delta = Int((size(inferred,2)+1) / n_bins)
            curr = 0
            for k=1:n_bins
                if curr <= L < curr + delta
                    if normalize
                        ranks[k,j] += 1/size(ground_truth,1)
                    else
                        ranks[k,j] += 1
                    end
                end
                curr += delta
            end
        end
    end
    return ranks
end

"""
    χ2_uniformtest(raw_ranks)

This function performs a chi-squared test for uniformity on the given ranks.

Arguments:
- `raw_ranks`: a vector of raw ranks.

Returns:
- The p-value of the chi-squared test.
"""
function χ2_uniformtest(raw_ranks)
    J = length(raw_ranks)
    tot_samples = sum(raw_ranks)
    exp_samples = tot_samples / J
    χ2 = sum((exp_samples .- raw_ranks).^2 ./ exp_samples)
    return 1 - cdf(Chisq(J-1), χ2)
end
