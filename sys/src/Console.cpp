/*
 * Console.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/Console.h"


vislib::sys::Console stuff; // Delete me!

/*
 * Coloring under windows with: 

    if (! GetConsoleScreenBufferInfo(hStdout, &csbiInfo)) 
    {
        MessageBox(NULL, TEXT("GetConsoleScreenBufferInfo"), 
            TEXT("Console Error"), MB_OK); 
        return 0;
    }

    wOldColorAttrs = csbiInfo.wAttributes; 

    // Set the text attributes to draw red text on black background. 

    if (! SetConsoleTextAttribute(hStdout, FOREGROUND_RED | 
            FOREGROUND_INTENSITY))
    {
        MessageBox(NULL, TEXT("SetConsoleTextAttribute"), 
            TEXT("Console Error"), MB_OK);
        return 0;
    }


	SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN);
	printf("weiss ");
	SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
	printf("weisser ");
	SetConsoleTextAttribute(hStdout, FOREGROUND_RED);
	printf("rot ");
	SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_INTENSITY);
	printf("roter ");
	SetConsoleTextAttribute(hStdout, BACKGROUND_BLUE | BACKGROUND_RED | BACKGROUND_GREEN);
	printf("andersrum\n");

	//SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN);
	SetConsoleTextAttribute(hStdout, wOldColorAttrs);

 *
 */

/*
 * vislib::sys::Console::GetConsole
 */
vislib::sys::Console& vislib::sys::Console::GetConsole(void) {
    // TODO: Implement
    return stuff;
}


/*
 * vislib::sys::Console::Console
 */
vislib::sys::Console::Console(void) {
    // TODO: Implement
}


/*
 * vislib::sys::Console::~Console
 */
vislib::sys::Console::~Console(void) {
    // TODO: Implement
}
