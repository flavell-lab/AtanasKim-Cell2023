# ConnectomePlot.jl
Example usage:
```julia
using ConnectomePlot, PyPlot

g = get_graph_white(0)
list_sensory, list_muscle = get_sensory_muscle(g)

dict_pos_z_non_p = ConnectomePlot.dict_pos_z_non_p
dict_pos_v2_non_p = ConnectomePlot.dict_pos_v2_non_p
dict_pos_v3_non_p = ConnectomePlot.dict_pos_v3_non_p

dict_pos_z_p = ConnectomePlot.dict_pos_z_p
dict_pos_v2_p = ConnectomePlot.dict_pos_v2_p
dict_pos_v3_p = ConnectomePlot.dict_pos_v3_p;

# main circuit
let
    dict_x = dict_pos_v2_non_p
    dict_y = dict_pos_z_non_p
    
    figure(figsize=(3,3))
    dict_rgba = Dict("AVA"=>[1,0,0,1], "SMDD"=>[0,1,0,1],
        "RIP"=>[0,0,1,1])
    color_connectome(g, list_muscle, dict_x, dict_y, dict_rgba)
end

# pharyngeal circuit
let
    dict_x = dict_pos_v2_p
    dict_y = dict_pos_z_p
    
    figure(figsize=(3,3))
    dict_rgba = Dict("I2"=>[1,0,0,1], "MC"=>[0,1,0,1])
    color_connectome(g, list_muscle, dict_x, dict_y, dict_rgba)
end
```
