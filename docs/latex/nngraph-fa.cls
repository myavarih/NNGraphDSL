% nngraph-fa.cls — Persian RTL variant of amznotes for NNGraph DSL report
% Based on amznotes by Alex M Zhang / David Zhang
% Adapted for XeLaTeX + xepersian (RTL)
%
% USAGE: compile with XeLaTeX
%        \usepackage{xepersian} MUST be the last \usepackage in the document
%        \settextfont and \setdigitfont go AFTER \usepackage{xepersian}

\ProvidesClass{nngraph-fa}[2026/06/16 nngraph-fa]
\NeedsTeXFormat{LaTeX2e}

\LoadClass[12pt]{report}

% ------------------------------------------------------------
% PACKAGES

\RequirePackage[margin=1in]{geometry}
\RequirePackage{etoolbox}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage{hyperref}
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
\RequirePackage{parskip}
\RequirePackage{setspace}
\RequirePackage[explicit]{titlesec}
\RequirePackage[most]{tcolorbox}
\RequirePackage{multicol}
\RequirePackage{enumitem}
\RequirePackage{tabularx}
\RequirePackage{booktabs}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{graphicx}
\RequirePackage{minted}
\tcbuselibrary{minted}

% ------------------------------------------------------------
% PAGE LAYOUT

\onehalfspacing
\renewcommand{\arraystretch}{1.25}

% ------------------------------------------------------------
% COLORS

\definecolor{amzchaptercolor}{HTML}{0984e3}
\definecolor{amzdfnboxcolor}{HTML}{0984e3}
\definecolor{amzthmboxcolor}{HTML}{6c5ce7}
\definecolor{amzexboxcolor}{HTML}{fdcb6e}
\definecolor{amzgenboxcolor}{HTML}{00b894}
\definecolor{amztecboxcolor}{HTML}{e17055}
\definecolor{amzcodeboxcolor}{HTML}{2d3436}
\definecolor{amznoteboxcolor}{HTML}{d63031}

% ------------------------------------------------------------
% HYPERREF

\hypersetup{
    colorlinks=false,
    linkbordercolor=black,
    urlbordercolor=blue,
    pdfborderstyle={/S/U/W 1}
}

% ------------------------------------------------------------
% CHAPTER STYLING

\titleformat{\chapter}[display]
{\normalfont\huge\bfseries}
{}
{20pt}
{
    \begin{tcolorbox}[
        enhanced,
        left=23mm,
        right=25mm,
        top=2mm,
        spread sidewards,
        sharp corners,
        colback=amzchaptercolor,
        colframe=amzchaptercolor,
        boxrule=1mm,
        title=\thechapter,
        attach boxed title to top left={xshift=15mm, yshift=-4mm},
        boxed title style={
            size=small,
            colback=amzchaptercolor,
            colframe=white
        }
    ]
        \strut \textcolor{white}{#1}
    \end{tcolorbox}
}

\titleformat{name=\chapter,numberless}[display]{\normalfont\huge\bfseries}
{}
{20pt}
{
    \begin{tcolorbox}[
        enhanced,
        left=23mm,
        right=25mm,
        top=2mm,
        spread sidewards,
        sharp corners,
        colback=amzchaptercolor,
        colframe=amzchaptercolor,
        boxrule=1mm,
    ]
        \strut \textcolor{white}{#1}
    \end{tcolorbox}
}

\titlespacing*{\chapter}{0pt}{-40mm}{-10mm}

% ------------------------------------------------------------
% HEADERS

\pagestyle{fancy}
\lhead{\footnotesize\nouppercase{\leftmark}}
\rhead{\footnotesize\nouppercase{\rightmark}}
\cfoot{\thepage\ از \pageref{LastPage}}

\fancypagestyle{plain}{
    \renewcommand{\headrulewidth}{0pt}
    \fancyhead{}
    \cfoot[C]{\thepage\ از \pageref{LastPage}}
}

% ------------------------------------------------------------
% COLOR BOXES

\newcommand{\amzcolorboxtitlesep}{\ \raisebox{2.3pt}{\scalebox{0.5}{$\blacktriangleright$}}}

\tcbset{
    amzdefaultstyle/.style={
        enhanced,
        drop fuzzy shadow,
        fonttitle=\strut\bfseries,
        boxrule=0pt,
        frame hidden,
        separator sign=\amzcolorboxtitlesep,
        breakable,
    },
    genboxstyle/.style={
        amzdefaultstyle,
        frame style={fill=amzgenboxcolor!85!black},
        interior style={fill=amzgenboxcolor!5!white},
        segmentation style={solid,draw=amzgenboxcolor!50!black,opacity=0.25,line width=1pt}
    },
    noteboxstyle/.style={
        enhanced,
        boxrule=0pt,
        frame hidden,
        borderline west={4pt}{0pt}{amznoteboxcolor},
        colback=amznoteboxcolor!7!white,
        sharp corners,
    },
    codeboxstyle/.style={
        overlay={\begin{tcbclipinterior}\fill[darkgray!20!white]
            (frame.south west) rectangle ([xshift=6mm]frame.north west);
        \end{tcbclipinterior}},
        skin=bicolor,
        collower=white,
        colbacklower=darkgray,
        before upper={\setLTR},
    },
}

% Fix: switch upper content to LTR so minted/verbatim aligns correctly
% in RTL (xepersian) documents. \setLTR is defined by bidi (loaded by
% xepersian), so this must run after \begin{document}.
\AtBeginDocument{%
    \tcbset{codeboxstyle/.append style={before upper={\setLTR}}}%
}

\renewcommand{\l@tcolorbox}{\@dottedtocline{1}{0pt}{3em}}
\setlength{\columnsep}{3em}

\newcounter{amzboxcounter}
\NewDocumentCommand{\newamzbox}{m m m m O{}}{
    \newtcbtheorem[use counter=amzboxcounter,number within=section,list inside={#3}]{tc#1}{#2}{
        amzdefaultstyle,
        frame style={fill=#4!85!black},
        interior style={fill=#4!5!white},
        segmentation style={solid,draw=#4!50!black,opacity=0.25,line width=1pt},
        #5
    }{#3}

    \providetoggle{#3}
    \settoggle{#3}{false}

    \newenvironment{#1}[2]{
        \global\settoggle{#3}{true}
        \begin{tc#1}{##1}{##2}
    }{
        \end{tc#1}
    }

    \newenvironment{#1*}[1]{
        \begin{tc#1*}{##1}
    }{
        \end{tc#1*}
    }
}

\newamzbox{dfnbox}{تعریف}{dfn}{amzdfnboxcolor}
\newamzbox{exbox}{مثال}{ex}{amzexboxcolor}
\newamzbox{tecbox}{تکنیک}{tec}{amztecboxcolor}

\newtcolorbox{genbox}[1]{amzdefaultstyle, genboxstyle, title={#1}}
\newtcolorbox{notebox}{noteboxstyle}

\renewcommand{\theFancyVerbLine}{\ttfamily\scriptsize{\arabic{FancyVerbLine}}}
\newamzbox{codebox}{قطعه کد}{code}{amzcodeboxcolor}[codeboxstyle]

\newenvironment{amzcode}[2][]{
    \VerbatimEnvironment
    \begin{minted}[
        linenos,
        breaklines,
        autogobble,
        xleftmargin=3.75mm,
        #1
    ]{#2}}
{\end{minted}}

% ------------------------------------------------------------
% TEXT MACROS

\newcommand{\dfntxt}[1]{\textit{\textbf{#1}}}
% LTR wrappers for English/code inside Persian text
% defined AFTER \usepackage{xepersian} gives us \LR
% so we delay definition to \AtBeginDocument
\AtBeginDocument{
    \providecommand{\en}[1]{\LR{#1}}
    \providecommand{\code}[1]{\LR{\texttt{#1}}}
}
