
struct TAIL_NEAR_HEAD <: Exception
    idx::Int
    HEAD_TAIL_NEAR_HEAD(idx) = new(idx)
end

struct HEAD_OUT_OF_VIEW <: Exception
    idx::Int
    HEAD_OUT_OF_VIEW(idx) = new(idx)
end

struct TAIL_NEAR_VC <: Exception
    idx::Int
    TAIL_NEAR_VC(idx) = new(idx)
end

struct OUT_OF_FOCUS <: Exception
    idx::Int
    OUT_OF_FOCUS(idx) = new(idx)
end

struct NOT_ENOUGH_NEURONS <: Exception
    idx::Int
    NOT_ENOUGH_NEURONS(idx) = new(idx)
end
