" Functionally Psychedelic
" Vim colorscheme, dark background, very contrasty and colorful
" Maintainer:	kba <unixprog@gmail.com>
" Latest Change:    Thu Feb 12 17:52:28 CET 2009
"{{{ Commento
" TODO:
"   - make those commands functions and make them dependent
"     on variables, so the values aren't reset every time the scheme
"     loads
"   - integrate a version for 8 colors (linux console)
"   - separate gui and cterm colors
"   - make functions from the commands
"   - implement some variables
"   - find a good name :-)
"}}}
"{{{ INITIALIZATION
"==================
" "{{{ Iniitialization
set background=dark
hi clear
if exists("syntax_on")
  syntax reset
endif
let g:colors_name = "krxt-256"
"}}}
"}}}
"{{{ COLOR DEFINITIONS
"=====================
"{{{ Normal and comment default values .

hi Normal  ctermfg=245 ctermbg=0 cterm=none guifg=#999 guibg=black
hi Comment  ctermfg=101 ctermbg=233 cterm=none guibg=#111 guifg=#888956
"}}}
"{{{ Code syntax elements .
hi Constant ctermfg=2 guifg=green
hi Identifier ctermfg=62 cterm=none guifg=#55f
hi Error  ctermfg=203  ctermbg=none guibg=DarkRed
hi Ignore ctermfg=236 guifg=#444
" hi Keyword  ctermfg=33 guifg=#55f
hi link Keyword Statement
hi Function ctermfg=13 guifg=magenta
hi!  link Repeat Function
hi!  link Conditional Function
" hi Include  ctermfg=203
hi Special  ctermfg=9 ctermbg=233 cterm=none
hi Number  ctermfg=202 guifg=#c51
hi String ctermfg=10  guifg=#2c2
hi Type ctermfg=31 guifg=#088 gui=none
hi Todo ctermbg=1  ctermfg=15
hi Statement ctermfg=214 guifg=#d92 gui=none
hi Title cterm=bold ctermfg=247 guifg=magenta
" hi Statement  ctermfg=178 guifg=#d92 gui=none
hi! link PreProc Function
"}}}
"{{{ vim internals .
hi Visual ctermbg=17 guibg=#004
hi Directory ctermfg=12 ctermbg=bg
" hi IncSearch  ctermfg=195  ctermbg=17
hi IncSearch  ctermfg=none  ctermbg=none cterm=reverse,bold
" hi Search ctermfg=16 ctermbg=220  cterm=none,bold
hi Search ctermfg=16  ctermbg=2  cterm=none,bold guifg=black guibg=DarkGreen gui=bold
hi ErrorMsg ctermfg=16  ctermbg=202 cterm=bold guibg=#c14232 guifg=black gui=bold
" hi WarningMsg ctermfg=16  ctermbg=190 cterm=bold guibg=#990 guifg=black gui=bold
hi WarningMsg ctermfg=9  ctermbg=bg cterm=bold guibg=#990 guifg=black gui=bold
hi ModeMsg  ctermfg=179 ctermbg=234 cterm=none guibg=#111 guifg=#b84 gui=none
hi MoreMsg  ctermfg=15  ctermbg=17 cterm=bold guibg=#004 guifg=#fa0
hi WildMenu ctermfg=11  ctermbg=233 cterm=bold guifg=#fa0 guibg=#181818 gui=bold
hi MatchParen ctermfg=231  ctermbg=52 cterm=bold guifg=Red guibg=black
hi SpellBad    term=reverse   ctermbg=52 gui=undercurl guisp=Red       " badly spelled word 
hi SpellCap    term=reverse   ctermbg=9  gui=undercurl guisp=Blue      " word with wrong caps
hi SpellRare   term=reverse   ctermbg=13 gui=undercurl guisp=Magenta   " rare word
hi SpellLocale term=underline ctermbg=11 gui=undercurl guisp=DarkCyan  " word only exists in other region
"}}}
"{{{ plugins
hi! link TagListFilename Folded
"}}}
"{{{ Folding .
hi Fold   ctermfg=5  ctermbg=black guifg=DarkMagenta guibg=black
hi FoldColumn ctermfg=6 ctermbg=black guifg=DarkCyan guibg=black
hi Folded ctermfg=131  ctermbg=234 guifg=#a66 guibg=#1c1c1c
"}}}
"{{{ Popup Menu .
hi PmenuSel ctermfg=11 ctermbg=17 guifg=#d90 guibg=#004
hi Pmenu  ctermfg=11  ctermbg=0 guibg=black guifg=#d90
hi PmenuSbar  ctermfg=11 ctermbg=233 cterm=none guifg=#d90 guibg=#1c1c1c gui=none
hi PmenuThumb ctermfg=0  ctermbg=11  cterm=none guifg=black guibg=#d90 gui=none
"}}}
"{{{ Status Line and VertSplit .
" hi StatusLine ctermfg=16 ctermbg=22 cterm=none
hi StatusLine ctermfg=84 ctermbg=22 cterm=none 
hi Statusline guifg=LightGreen guibg=#040 gui=none
hi StatusLineNC ctermfg=204 ctermbg=52  cterm=none
hi StatusLineNC guibg=#400 guifg=#c66 gui=none
" User Highlights in statusline
" Modified flag
hi User1  ctermbg=12  ctermfg=black cterm=none
hi User1  guifg=black guibg=#55c gui=none
" parentheses
hi User2  ctermbg=232  ctermfg=12  cterm=none
hi User2  guifg=#55c guibg=#060606
" Labels
hi User3  ctermbg=232  ctermfg=6 cterm=none
hi User3  guibg=#060606   guifg=DarkCyan
" Values
hi User4  ctermbg=232  ctermfg=3 cterm=none
" Filetype
hi User5  ctermbg=232  ctermfg=11  cterm=none
hi User5  guibg=#060606  guifg=#d90  gui=none
" Modified flag (red on black version
hi User6  ctermfg=15   ctermbg=52 cterm=bold
hi User6  guifg=red guibg=#222 gui=bold
" Filename
" hi User6  ctermfg=3  cterm=bold
" hi link User6 StatusLine
" hi link User6 StatusLineNC


" Vertical Split Line
hi VertSplit ctermfg=0 ctermbg=8  cterm=NONE
hi VertSplit guifg=black guibg=#444 gui=NONE
"}}}
"{{{ Tab Line, Line Number, Cursor color .
hi LineNr  ctermfg=3  ctermbg=232 guifg=#aa0 guibg=#070707
hi TabLine  ctermfg=0 ctermbg=240  cterm=none 
hi TabLine  guifg=black guibg=#555 gui=none
hi TabLineFill  ctermfg=0 ctermbg=0  cterm=none
hi TabLineFill  guifg=black guibg=black  gui=none
hi TabLineSel ctermfg=3 ctermbg=237  cterm=bold
hi TabLineSel guifg=#d90 guibg=#121  gui=bold
hi CursorLine ctermbg=235  cterm=none term=none
hi CursorLine guibg=#131313  gui=none term=underline
hi Cursor guifg=#300 guibg=#666
"}}}
"{{{ List and Outline Levels
"}}}
"}}}
"{{{ COMMANDS AND FUNCTIONS
"==========================
"{{{ Commands to change coloring flavor .
" those should probably be functions
command! ColorStatusOrange    exec( "hi! StatusLine ctermbg=166 ctermfg=16 cterm=bold" )
command! ColorStatusPurple    exec( "hi! StatusLine ctermbg=53  ctermfg=84" )
command! ColorStatusBlue      exec( "hi! StatusLine ctermbg=17  ctermfg=38" )
command! ColorStatusWhite     exec( "hi! StatusLine ctermbg=white ctermfg=black" )
command! ColorStatusGreen     exec( "hi! StatusLine ctermbg=22  ctermfg=84" )
command! ColorBGGray          exec( "let g:krxt_bg_gray=1|colo krxt-256" )
command! ColorBGBlack         exec( "unlet g:krxt_bg_gray|colo krxt-256" )
command! ColorCommentBGGray   exec( "unlet g:krxt_comment_black|colo krxt-256" )
command! ColorCommentBGBlack  exec( "let g:krxt_comment_black=1|colo krxt-256" )
"}}}
"{{{ variables
" Dark gray background instead of black, if let
" let krxt_bg_gray = 1
" unlet krxt_bg_gray
" Dark gray background for comments if let
" let krxt_comment_gray = 0
" unlet krxt_comment_gray

" enable these to quickly change values and reload
" map <buffer> <c-x> <c-x>:w\|colo krxt_urxvt<cr>
" map <buffer> <c-a> <c-a>:w\|colo krxt_urxvt<cr>
" if exists("krxt_bg_gray")
  " hi Normal  ctermfg=245 ctermbg=232 cterm=none guifg=#999 guibg=#050505
" else
  " hi Normal  ctermfg=245 ctermbg=0 cterm=none guifg=#999 guibg=black
" end

" Syntax
" if exists("krxt_comment_black")
  " hi Comment  ctermfg=238 ctermbg=none cterm=none
" else
  " hi Comment  ctermfg=101 ctermbg=233 cterm=none guibg=#111 guifg=#888956
" end
"}}}
"}}}
"{{{ VIMOUTLINER COLORS

" colors for tags
hi outlTags guifg=darkred	ctermfg=darkred

" color for tables 
hi TA1 guifg=darkviolet	ctermfg=cyan
hi TA2 guifg=darkviolet	ctermfg=cyan
hi TA3 guifg=darkviolet	ctermfg=cyan
hi TA4 guifg=darkviolet	ctermfg=cyan
hi TA5 guifg=darkviolet	ctermfg=cyan
hi TA6 guifg=darkviolet	ctermfg=cyan
hi TA7 guifg=darkviolet	ctermfg=cyan
hi TA8 guifg=darkviolet	ctermfg=cyan
hi TA9 guifg=darkviolet	ctermfg=cyan

" color for user text (wrapping)
hi UT1 guifg=darkred	ctermfg=cyan
hi UT2 guifg=darkred	ctermfg=cyan
hi UT3 guifg=darkred	ctermfg=cyan
hi UT4 guifg=darkred	ctermfg=cyan
hi UT5 guifg=darkred	ctermfg=cyan
hi UT6 guifg=darkred	ctermfg=cyan
hi UT7 guifg=darkred	ctermfg=cyan
hi UT8 guifg=darkred	ctermfg=cyan
hi UT9 guifg=darkred	ctermfg=cyan

" color for user text (non-wrapping)
hi UB1 guifg=darkgray	ctermfg=cyan
hi UB2 guifg=darkgray	ctermfg=cyan
hi UB3 guifg=darkgray	ctermfg=cyan
hi UB4 guifg=darkgray	ctermfg=cyan
hi UB5 guifg=darkgray	ctermfg=cyan
hi UB6 guifg=darkgray	ctermfg=cyan
hi UB7 guifg=darkgray	ctermfg=cyan
hi UB8 guifg=darkgray	ctermfg=cyan
hi UB9 guifg=darkgray	ctermfg=cyan

" colors for folded sections
hi Folded guifg=darkcyan guibg=bg	ctermfg=cyan ctermbg=black
hi FoldColumn guifg=darkcyan guibg=bg	ctermfg=cyan ctermbg=black

" colors for experimental spelling error highlighting
" this only works for spellfix.vim with will be cease to exist soon
hi spellErr gui=underline guifg=yellow	cterm=underline ctermfg=yellow
hi BadWord gui=underline guifg=yellow	cterm=underline ctermfg=yellow

hi OutlineLevel1 guifg=white	ctermfg=15
hi OutlineLevel2 guifg=red	ctermfg=red
hi OutlineLevel3 guifg=blue	ctermfg=6
hi OutlineLevel4 guifg=darkviolet	ctermfg=11
hi OutlineLevel5 guifg=black	ctermfg=12
hi OutlineLevel6 guifg=red	ctermfg=13
hi OutlineLevel7 guifg=blue	ctermfg=1
hi OutlineLevel8 guifg=darkviolet	ctermfg=2
hi OutlineLevel9 guifg=black	ctermfg=3

hi CodeBlock guifg=cyan ctermfg=105
hi OutlineParagraph guifg=darkgray ctermfg=28

hi otlCheckBoxLow ctermfg=52 cterm=bold
hi otlCheckBoxMiddle ctermfg=166 cterm=bold
hi otlCheckBoxAlmost ctermfg=28 cterm=bold
hi otlCheckBoxFull ctermfg=8 cterm=underline
"
"}}}
"
" vim: sw=2
