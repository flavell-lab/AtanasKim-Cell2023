function bilerp(img, x, y)
    W, H = size(img)
    x = clamp(x, 1, W)
    y = clamp(y, 1, H)
    x0, y0 = floor(Int, x), floor(Int, y)
    xα, yα = x - x0, y - y0
    x1, y1 = min(x0 + 1, W), min(y0 + 1, H)
    @inbounds img00, img01 = img[x0, y0], img[x1, y0]
    @inbounds img10, img11 = img[x0, y1], img[x1, y1]
    (1-xα)*(1-yα)*img00 +
        (xα)*(1-yα)*img01 +
        (1-xα)*(yα)*img10 +
        (xα)*(yα)*img11
end
