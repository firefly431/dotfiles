set nocompatible
set backspace=indent,eol,start
set history=50
set ruler
set showcmd
set incsearch
set hlsearch
set number
set expandtab
set autowrite
set wildmenu
set hidden
if has("gui_running")
  set cursorline
endif
set undofile
set ignorecase
set smartcase
set gdefault
set tabstop=4
set shiftwidth=4
set softtabstop=4
set cino=:0l1g0(0
set encoding=utf-8
set guifont=Fira\ Mono\ for\ Powerline
set cole=2
set completeopt-=preview
set scrolloff=3
set list
set listchars=tab:▸\ ,trail:•
set diffopt+=vertical
set foldlevelstart=99
set guioptions-=T
set laststatus=2

let mapleader = ","

syntax on
filetype plugin indent on

au BufNewFile,BufRead *.cpp set syntax=cpp11
au BufRead,BufNewFile *.html setlocal tabstop=2 shiftwidth=2 softtabstop=2
au BufLeave * silent! :wa

map Q gq
nnoremap ; :
nnoremap : ;
nnoremap <silent>j gj
nnoremap <silent>k gk
inoremap <C-Space> <C-N>
inoremap <C-U> <C-G>u<C-U>
inoremap <C-W> <C-G>u<C-W>
inoremap kj <Esc>
nnoremap <silent> <backspace> :noh<CR>
inoremap <C-[> {<CR><BS>}<Esc>O
inoremap <C-[>; {<CR><BS>};<Esc>O
inoremap <S-CR> <Esc>o
inoremap <C-E> <Esc>A

nnoremap <silent><C-j> m`:silent +g/\m^\s*$/d<CR>``:noh<CR>
nnoremap <silent><C-k> m`:silent -g/\m^\s*$/d<CR>``:noh<CR>
nnoremap <silent><A-j> :set paste<CR>m`o<Esc>``:set nopaste<CR>
nnoremap <silent><A-k> :set paste<CR>m`O<Esc>``:set nopaste<CR>

if has("vms")
  set nobackup
else
  set backup
endif

if has('mouse')
  set mouse=a
endif

if has("gui_running")
  set background=light
  colorscheme bclear
else
  set background=dark
  colorscheme harlequin
endif

hi Conceal guibg=White guifg=#dddddd

let g:tex_conceal='adgm'

let g:jedi#use_tabs_not_buffers = 0
let g:jedi#use_splits_not_buffers = "left"
let g:jedi#force_py_version = 3

execute pathogen#infect()
