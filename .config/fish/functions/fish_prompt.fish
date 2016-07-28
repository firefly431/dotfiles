function fish_prompt
    set_color $fish_color_cwd
    set -l suffix
    switch $USER
    case root toor
        set suffix '#'
    case '*'
        set suffix '$'
    end
    echo -n (prompt_pwd)
    set_color normal
    echo -n "$suffix "
end
