"""
    align(curve1_x, curve1_y, curve2_x, curve2_y; headpt::Integer=4, tailpt::Integer=7)

Aligns the points `headpt` and `tailpt` of curve 2 to match curve 1, and returns the transformed curve 2.
Translates, rotates, and scales all other curve-points accordingly.
# Arguments:
- `curve1_x`: Array of x-coordinates of first worm
- `curve1_y`: Array of y-coordinates of first worm
- `curve2_x`: Array of x-coordinates of second worm
- `curve1_y`: Array of y-coordinates of second worm
- `headpt::Integer`: First position from head (in index of curves) to be aligned. Default 4.
- `tailpt::Integer`: Second position from head (in index of curves) to be aligned. Default 7.
"""
function align(curve1_x, curve1_y, 
        curve2_x, curve2_y; headpt::Integer=4, tailpt::Integer=7)
    # Make tip of the nose be the origin
    c1_x = curve1_x .- curve1_x[headpt]    
    c1_y = curve1_y .- curve1_y[headpt]
    c2_x = curve2_x .- curve2_x[headpt]    
    c2_y = curve2_y .- curve2_y[headpt]
    # compute rotation angle
    theta_1 = atan(c1_y[tailpt]/c1_x[tailpt])
    theta_2 = atan(c2_y[tailpt]/c2_x[tailpt])
    delta_theta = theta_1 - theta_2
    # compute scaling factor
    scale_1 = sqrt(c1_x[tailpt]^2 + c1_y[tailpt]^2)
    scale_2 = sqrt(c2_x[tailpt]^2 + c2_y[tailpt]^2)
    scale = scale_1/scale_2
    # transform curve 2
    M_rot = scale * [cos(delta_theta) -sin(delta_theta); sin(delta_theta) cos(delta_theta)]
    c2 = M_rot * transpose([c2_x c2_y])
    # we need to rotate by an additional pi
    if abs(c2[1,tailpt] - c1_x[tailpt]) > 1
        c2 = -c2
    end
    # add back curve 1's origin-position
    return (c2[1,:] .+ curve1_x[headpt], c2[2,:] .+ curve1_y[headpt])
end

"""
    curve_distance(x1_c, y1_c, x2_c, y2_c; headpt::Integer=4, tailpt::Integer=7)

Computes the difficulty of an elastix transform
using the heuristic that more worm-unbending is harder.
# Arguments:
- `x1_c`: Array of x-coordinates of first worm
- `y1_c`: Array of y-coordinates of first worm
- `x2_c`: Array of x-coordinates of second worm
- `y2_c`: Array of y-coordinates of second worm
- `headpt::Integer`: First position from head (in index of curves) to be aligned. Default 4.
- `tailpt::Integer`: Second position from head (in index of curves) to be aligned. Default 7.
"""
function curve_distance(x1_c, y1_c, x2_c, y2_c;
        headpt::Integer=4, tailpt::Integer=7)
    x2, y2 = align(x1_c, y1_c, x2_c, y2_c; headpt=headpt, tailpt=tailpt)
    delta = sum(sqrt.((x2 .- x1_c).^2 + (y2 .- y1_c).^2))
    return delta
end




