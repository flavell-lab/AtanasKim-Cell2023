dt_now_str(format_str="yyyy-mm-dd HH:MM:SS") = Dates.format(now(), format_str)
str_with_dt(str) = "$(dt_now_str()): $str"
println_dt(str) = println(str_with_dt(str))
