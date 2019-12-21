The following files are contained in this package:

HD.cpp     -  Source code of the algorithms to compute the halfspace depth
              as described in "Exact computation of the halfspace depth" 
              by Rainer Dyckerhoff and Pavlo Mozharovskyi (arXiv:1411:6927)   
HD.h       -  Header file
HDTest.cpp -  Source code of a small console program to demonstrate the use
              of the routines in HD.cpp
makefile   -  Makefile to build the program
ReadMe.txt -  This file

The program was built with Visual Studio 2013 as well as with the GNU C++ 
compiler. The main program 'HDTest.cpp' uses features of C++11. However,
the algorithms in 'HD.cpp' use only features of C++98.