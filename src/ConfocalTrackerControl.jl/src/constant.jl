# loop
const LOOP_INTERVAL_SAVE_STAGE = 20 # Hz
const LOOP_INTERVAL_CONTROL = 40 # Hz

# serial stage
const SERIAL_BAUD_RATE = 115200
const DEFAULT_PORT = "/dev/ttyACM0"

# PID
const PID_X_P = 180.
const PID_X_I = 0.
const PID_X_D = 0.
const PID_Y_P = 180.
const PID_Y_I = 0.
const PID_Y_D = 0.

# neural net
const PATH_MODEL = "/home/imaging/dl_model/nir-track-jsk-2020-10-02/exported-models/DLC_nir-track_resnet_50_iteration-0_shuffle-1"

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

# gui
const FLIR_BFS_PIX_100UM = round(Int, 0.1 / FLIR_BFS_PIX_SIZE)
const RULER_RG_X = sort(union(IMG_CENTER_X:-FLIR_BFS_PIX_100UM:1,
        IMG_CENTER_X:FLIR_BFS_PIX_100UM:IMG_SIZE_X));
const RULER_RG_Y = sort(union(IMG_CENTER_Y:-FLIR_BFS_PIX_100UM:1,
        IMG_CENTER_Y:FLIR_BFS_PIX_100UM:IMG_SIZE_Y))

const path_qmlfile = joinpath(dirname(@__FILE__), "gui.qml")

# NIDAQ
const NIDAQ_SAMPLE_RATE_AI = 5000 # Hz
const NIDAQ_BUFFER_SIZE = 100 * NIDAQ_SAMPLE_RATE_AI
const NIDAQ_DEV_NAME = "Dev1"