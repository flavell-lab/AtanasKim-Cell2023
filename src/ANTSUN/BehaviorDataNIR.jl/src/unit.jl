const FLIR_BFS_PIX_SIZE = 1 / 791 # [mm/px] 
const FLIR_FPS = 20 # fps
const STAGE_UNIT = 10000 # stage unit / mm

unit_bfs_pix_to_stage_unit(pix) = pix * FLIR_BFS_PIX_SIZE * STAGE_UNIT
unit_bfs_pix_to_mm(pix) = FLIR_BFS_PIX_SIZE * pix
unit_mm_to_bfs_pix(dist) = dist / FLIR_BFS_PIX_SIZE
unit_mm_to_stage_unit(dist) = dist * STAGE_UNIT
unit_stage_unit_to_mm(stage_unit) = stage_unit / STAGE_UNIT
unit_stage_unit_to_bfs_pix(stage_unit) = stage_unit / STAGE_UNIT / FLIR_BFS_PIX_SIZE
