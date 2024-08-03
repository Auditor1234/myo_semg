
#!/bin/bash

startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

# 执行多进程python程序
python main_db5_moe_all.py --subject_range 1 2 --fusion --uncertainty_type=DST --device=0 > /dev/null  &
python main_db5_moe_all.py --subject_range 3 4 --fusion --uncertainty_type=DST --device=0 > /dev/null  &
python main_db5_moe_all.py --subject_range 5 6 --fusion --uncertainty_type=DST --device=0 > /dev/null &
python main_db5_moe_all.py --subject_range 7 8 --fusion --uncertainty_type=DST --device=1  > /dev/null &
python main_db5_moe_all.py --subject_range 9 9 --fusion --uncertainty_type=DST --device=1  > /dev/null &
python main_db5_moe_all.py --subject_range 10 10 --fusion --uncertainty_type=DST --device=1  > /dev/null
# python main_db5_moe_all.py --subject_range 7 7 --fusion=True --uncertainty_type=DST --device=0 > /dev/null  &
# python main_db5_moe_all.py --subject_range 8 8 --fusion=True --uncertainty_type=DST --device=0 > /dev/null  &
# python main_db5_moe_all.py --subject_range 9 9 --fusion=True --uncertainty_type=DST --device=1 > /dev/null  &
# python main_db5_moe_all.py --subject_range 10 10 --fusion=True --uncertainty_type=DST --device=1 > /dev/null 

endTime=`date +%Y%m%d-%H:%M:%S`
endTime_s=`date +%s`

sumTime=$[ $endTime_s - $startTime_s ]

echo "$startTime ---> $endTime" "Total:$sumTime seconds"

