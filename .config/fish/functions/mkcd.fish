function mkcd
    mkdir -p $argv
    if test $status = 0
        cd $argv
    end
end
