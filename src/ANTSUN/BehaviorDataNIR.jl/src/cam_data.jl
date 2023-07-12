"""
    nmp_vec(pos_feature)

Given a 3x2xN array `pos_feature` of 2D coordinates of the nose, metacorpus and pharynx, this function returns three arrays:
- `mn`: an array of shape 2xN representing the vector from the metacorpus to the nose for each frame.
- `mp`: an array of shape 2xN representing the vector from the metacorpus to the pharynx for each frame.
- `mp⊥`: an array of shape 2xN representing the vector perpendicular to `mp` for each frame.
"""
function nmp_vec(pos_feature)
    n = pos_feature[1,1:2,:] # nose
    m = pos_feature[2,1:2,:] # metacorpus
    p = pos_feature[3,1:2,:] # pharynx
    
    mn = n .- m
    mp = p .- m
    mp⊥ = Array(hcat(mp[2,:], - mp[1,:])')
    
    mn, mp, mp⊥
end