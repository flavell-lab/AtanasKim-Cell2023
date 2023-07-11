# camera - FLIR BFS
# const FLIR_BFS_PIX_SIZE = 1  / 624 # [mm/px] 0.55x
const FLIR_BFS_PIX_SIZE = 1 / 791 # [mm/px] 

const IMG_SIZE_X = 968
const IMG_SIZE_Y = 732
const IMG_CROP_SIZE_X = 352
const IMG_CROP_SIZE_Y = 352
const IMG_CROP_RG_X = round(Int, (IMG_SIZE_X - IMG_CROP_SIZE_X) / 2) .+ (0:IMG_CROP_SIZE_X-1)
const IMG_CROP_RG_Y = round(Int, (IMG_SIZE_Y - IMG_CROP_SIZE_Y) / 2) .+ (0:IMG_CROP_SIZE_Y-1)
const IMG_CENTER_X = round(Int, IMG_SIZE_X / 2)
const IMG_CENTER_Y = round(Int, IMG_SIZE_Y / 2)
const IMG_CROP_CENTER_X = round(Int, IMG_CROP_SIZE_X / 2)
const IMG_CROP_CENTER_Y = round(Int, IMG_CROP_SIZE_Y / 2)

# stage
const STAGE_UNIT = 10000 # stage unit / mm

function convert_bfs_pix_to_stage_unit(pix)
    pix * FLIR_BFS_PIX_SIZE * STAGE_UNIT
end

function covert_stage_unit_to_mm(stage_unit)
    stage_unit / STAGE_UNIT
end

function convert_mm_to_bfs_pix(dist)
    dist / FLIR_BFS_PIX_SIZE
end

function convert_bfs_pix_to_mm(pix)
    FLIR_BFS_PIX_SIZE * pix
end