module TotalVariation

using  DSP, SparseArrays, LinearAlgebra
export gstv, tv
#See ``Total Variation Denoising With Overlapping Group Sparsity'' by
# Ivan Selesnick and Po-Yu Chen (2013)

#Group Sparse Total Variation Denoising
function gstv(y::Vector{Float64}, k::Int, λ::Float64;
    show_cost::Bool=false, iter::Int=100)
    #Initialize Solution
    N  = length(y)
    x  = copy(y)

    #Differential of input
    b = diff(y)

    #Precalculate D D' where D is the first-order difference matrix
    DD::SparseMatrixCSC{Float64,Int} = spdiagm(-1=>-ones(N-2), 0=>2*ones(N-1), 1=>-ones(N-2))

    #Value To Prevent Singular Matrices
    epsilon = 1e-15

    #Convolution Mask - spreads D x over a larger area
    #This regularizes the problem with applying a gradient to a larger area.
    #at k=1 the normal total variational sparse solution (for D x) is found.
    h = ones(k)

    for i=1:iter
        u::Vector{Float64}              = diff(x)
        r::Vector{Float64}              = sqrt.(max.(epsilon, DSP.conv(u.^2,h)))
        Λ::Vector{Float64}              = DSP.conv(1 ./ r, h)[k:end-(k-1)]
        F::SparseMatrixCSC{Float64,Int} = sparse(Diagonal(1 ./ Λ))/λ + DD
        if(show_cost)
            #1/2||y-x||_2^2 + λΦ(Dx)
            #Where Φ(.) is the group sparse regularizer
            println("Cost at iter ",i," is ", 0.5*sum(abs2.(x.-y)) + λ*sum(r))
        end

        tmp::Vector{Float64} = F\b
        dfb::Vector{Float64} = diff(tmp)

        x[1]       = y[1]       + tmp[1]
        x[2:end-1] = y[2:end-1] + dfb[:] 
        x[end]     = y[end]     - tmp[end]
    end

    return x
end

#Normal Total Variation Problem
function tv(y::Vector{Float64}, λ::Float64;
    show_cost::Bool=false, iter::Int=100)
    gstv(y, 1, λ, show_cost=show_cost, iter=iter)
end

# package code goes here
end # module
