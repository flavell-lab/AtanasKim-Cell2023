count=1
while read line; do
    if [[ $count == $2 ]];
    then
        chmod u+x $line
        $line
    fi
    count=$((count+1))
done < $1

