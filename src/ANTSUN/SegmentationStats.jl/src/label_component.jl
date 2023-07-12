function mark_search_pix!(img_bin, img_label, x, y, n, img_size)
    img_label[x, y] = n # object number n

    # recursively search for connected neighboring pixels
    @inbounds for i = -1:1
        @inbounds for j = -1:1
                x_t = x + i
                y_t = y + j

                # checking if it's valid index
                if !(1 <= x_t <= img_size[1] && 1 <= y_t <= img_size[2])
                    continue
                end

                # check if already searched
                if img_label[x_t, y_t] != 0
                    continue
                end

                if  img_bin[x_t, y_t] == 1
                    mark_search_pix!(img_bin, img_label, x_t, y_t, n, img_size)
                else
                    img_label[x_t, y_t] = -1
                end
        end # j
    end # i
end

"""
    label_2d!(img_bin, img_label)
Label components given binary image.
`img_label` is marked either -1 or n> 0
0: not seached; -1: searched, nothing to segment; [1, n]: object number
Arguments
---------
* `img_bin`: binary image segmented
* `img_label`: array to label
"""
function label_2d!(img_bin, img_label)
    # img_label: marking searched pixel and object number
    # 0: not seached; -1: searched, nothing to segment; [1, n]: object number
    @assert size(img_bin) == size(img_label)

    img_size = size(img_bin) # x, y
    n = UInt32(1) # initial object id

    # interate over every pixel
    @inbounds for x=1:img_size[1]
        @inbounds for y=1:img_size[2]
                if img_label[x, y] == 0 # has not been searched
                    if img_bin[x, y] == 1 # object found
                        mark_search_pix!(img_bin, img_label, x, y, n, img_size)
                        n += 1
                    else # mark as searched and empty
                        img_label[x, y] = -1
                    end
                end
         end # y
    end # x

    nothing
end
