# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs

PATH=$PATH:$HOME/bin:$HOME/local/bin

export PATH

PS1='\[\e[34m\]\W\[\e[1m\]\$\[\e[0m\] '
