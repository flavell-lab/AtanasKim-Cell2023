Initially cloned from https://github.com/fundamental/TotalVariation.jl/

# TotalVariation

An implementation of Total Variation Denoising and Group Sparse Total Variation
Denoising.

[![Build
Status](https://travis-ci.org/fundamental/TotalVariation.jl.png)](https://travis-ci.org/fundamental/TotalVariation.jl)

Total Variation (TV) minimization uses the TV norm to reduce excess variation in
1D signals. Using TV for denoising will result in a piecewise constant function
with fewer pieces at higher levels of denoising.

Group sparse TV is an extension on the TV norm which models signals which have
several localized transitions. Larger group sizes help model smoother signals
with slow transitions.

For more information see src/example.jl and the source publication:

``Total Variation Denoising With Overlapping Group Sparsity'' by
Ivan Selesnick and Po-Yu Chen (2013)

