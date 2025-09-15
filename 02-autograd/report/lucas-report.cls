\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{lucas-report}[Lucas's Report Class]

\LoadClass[11pt,oneside]{article}

\RequirePackage{geometry}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{kotex}
\RequirePackage{amsmath,amssymb,mathtools}
% \RequirePackage{sidenotes}
\RequirePackage{derivative,fixdif}
\RequirePackage{algorithm,algpseudocode}
\RequirePackage[inline]{enumitem}
\RequirePackage{microtype}
\RequirePackage{tcolorbox}
\RequirePackage{titlesec}
\RequirePackage{parskip}
\RequirePackage{bytefield}
\RequirePackage{listings}
\RequirePackage{marginfix}
\RequirePackage{xspace}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage{wasysym}
\RequirePackage{hyperref}
\RequirePackage{datetime2}
\RequirePackage{tikz}
\RequirePackage{eso-pic}
\RequirePackage{tabularx}
\RequirePackage{wrapfig}
\RequirePackage{fancyvrb}
\RequirePackage{enumitem}
% \RequirePackage{lmodern}
\RequirePackage{siunitx}
\RequirePackage[backend=biber,sorting=none]{biblatex}
\RequirePackage{caption}
\RequirePackage{titling}
\RequirePackage{setspace}
\RequirePackage{subcaption}
\RequirePackage{tabularray}

\UseTblrLibrary{booktabs}

\geometry{
  a4paper,
  % textwidth=14cm,
  left=2.5cm,
  top=2.5cm,
  bottom=3cm,
  right=2.5cm,
  % marginparsep=0.5cm,
  % marginparwidth=4cm,
}

%%% Titling
\setlength{\droptitle}{-3em}

\pretitle{
  \begin{flushleft}\LARGE\bfseries
    }
    \posttitle{
  \end{flushleft}
}

\preauthor{
  \begin{flushleft}
    \large
    \onehalfspacing
    }
    \postauthor{
  \end{flushleft}
}

\predate{
  \begin{flushleft}
    }
    \postdate{
  \end{flushleft}
}

% \makeatletter
% \renewcommand{\maketitle}{\bgroup\setlength{\parindent}{0pt}
% \begin{flushleft}
%   {\LARGE\textbf{\@title}}

%   {\large\@author}

%   \@date

%   \vspace{2em}
% \end{flushleft}\egroup
% }
% \makeatother

\titleclass{\problem}{straight}[\part]
\newcounter{problem}
\renewcommand{\theproblem}{\arabic{problem}}
\titleformat{\problem}
{\bfseries\scshape\large}
{Problem \theproblem}
{0.75em}
{\mdseries}
\titlespacing{\problem}{0pt}{\baselineskip}{.5\baselineskip}

%%% Listing style

\lstset{numbers=left,numberstyle=\tiny,basicstyle=\ttfamily,columns=fixed,frame=tb,captionpos=b}

%%% Algorithm (algorithmicx) style

\makeatletter
\RenewDocumentCommand{\Comment}{s m}{
  \IfBooleanT{#1}{\Statex}\IfBooleanF{#1}{\hfill}\hspace*{\ALG@tlm}\(\triangleright\) #2
}
\makeatother

\algnewcommand{\IfThenElse}[3]{% \IfThenElse{<if>}{<then>}{<else>}
  \State \algorithmicif\ #1\ \algorithmicthen\ #2\ \algorithmicelse\ #3}

%%% Margin note style

% \makeatletter
% \newcommand{\marginparstyle}{\footnotesize\raggedright} % initialize style with start value
% \long\def\@ympar#1{% redefine margin par to avoid too many macros in the document
%   \@savemarbox\@marbox{\marginparstyle#1}% marginparstyle is a prefix to the marginpar text now
%   \global\setbox\@currbox\copy\@marbox % the rest of the definition is taken from original LaTeX
%   \@xympar}
% \renewcommand*{\marginfont}{\marginparstyle} % adjust marginnote font in the same way
% \makeatother

\newcommand*{\algorithmautorefname}{Algorithm}

%%% Full width (text + margin)

\newlength{\fullwidth}
\setlength{\fullwidth}{\textwidth}
\addtolength{\fullwidth}{\marginparwidth}
\addtolength{\fullwidth}{\marginparsep}

%%% Custom operation commands

\newcommand*\Xor{\mathbin{{}^\wedge{}}}
\newcommand*\Or{\mathbin{|}}
\renewcommand*\And{\mathbin{\&}}
\newcommand*\LShift{\ll}
\newcommand*\RShift{\gg}
\newcommand*\Not{\ensuremath{\mathord{\sim}}}
\newcommand*\LogicalNot{\ensuremath{\mathord{!}}}
\newcommand*\nan{\textsc{NaN}\xspace}
\NewDocumentCommand{\inc}{ s m }{\IfBooleanTF{#1}{\mathrel{++}\!#2}{#2\!\mathrel{++}}}
\NewDocumentCommand{\dec}{ s m }{\IfBooleanTF{#1}{\mathrel{--}\!#2}{#2\!\mathrel{--}}}

%%% Colors

\definecolor{lightcyan}{rgb}{0.84,1,1}
\definecolor{lightgreen}{rgb}{0.64,1,0.71}
\definecolor{lightred}{rgb}{1,0.7,0.71}

\setcounter{secnumdepth}{4}

\def\subsectionautorefname{section}
\def\subsubsectionautorefname{section}

%%% Vectors and Matrices

\DeclareDocumentCommand\vb{ m }{ \mathbf{#1} } % Vector bold [star for Greek and italic Roman]


\letdif{\grad}{nabla}
