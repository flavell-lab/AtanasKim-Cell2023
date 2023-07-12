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

function χ2_uniformtest(raw_ranks)
    J = length(raw_ranks)
    tot_samples = sum(raw_ranks)
    exp_samples = tot_samples / J
    χ2 = sum((exp_samples .- raw_ranks).^2 ./ exp_samples)
    return 1 - cdf(Chisq(J-1), χ2)
end
