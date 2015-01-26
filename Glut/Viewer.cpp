/*
 * Viewer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Viewer.h"
#include "visglut.h"
#include "Window.h"
#include "vislib/sys/CmdLineProvider.h"


/*
 * megamol::viewer::Viewer::Viewer
 */
megamol::viewer::Viewer::Viewer(void) : ApiHandle(), windows(), appTerminate() {
}


/*
 * megamol::viewer::Viewer::~Viewer
 */
megamol::viewer::Viewer::~Viewer(void) {
    vislib::SingleLinkedList<megamol::viewer::Window*>::Iterator iter
        = this->windows.GetIterator();
    vislib::SingleLinkedList<megamol::viewer::Window*> helper;
    while (iter.HasNext()) { helper.Add(iter.Next()); }
    this->windows.Clear();
    iter = helper.GetIterator();
    while (iter.HasNext()) { delete iter.Next(); }
    try {
		this->ProcessEvents();
    } catch(...) {
    }
    try {
	    ::glutMainLoopEvent();
    } catch(...) {
    }
    try {
		// It's a freeglut bug which makes this method explode for what reason ever.
		// Perhaps because I cleaned up my windows (with menus) before ...
		// To Debug: run the application (outside VS) and then attach the debugger
        //::glutExit();
    } catch(...) {
    }
}


/*
 * megamol::viewer::Viewer::Initialise
 */
bool megamol::viewer::Viewer::Initialise(unsigned int hints) {
    vislib::sys::CmdLineProviderA cmdLine("dummyApp -direct");
    unsigned int displayMode = GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA;

    ::glutInit(&cmdLine.ArgC(), cmdLine.ArgV());

    if ((hints & MMV_VIEWHINT_QUADBUFFER) == MMV_VIEWHINT_QUADBUFFER) {
        displayMode |= GLUT_STEREO;
    }

    if ((hints & MMV_VIEWHINT_ALPHABUFFER) == MMV_VIEWHINT_ALPHABUFFER) {
        displayMode |= GLUT_ALPHA;
    }

    ::glutInitDisplayMode(displayMode);
    ::glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
        GLUT_ACTION_CONTINUE_EXECUTION);

    return true;
}


/*
 * megamol::viewer::Viewer::ProcessEvents
 */
bool megamol::viewer::Viewer::ProcessEvents(void) {
    if (this->windows.IsEmpty()) return false;
    ::glutMainLoopEvent();
    return true;
}


/*
 * megamol::viewer::Viewer::CloseAllWindows
 */
void megamol::viewer::Viewer::CloseAllWindows(void) {
    megamol::viewer::Window** winz = new megamol::viewer::Window*[this->windows.Count()];
    unsigned int idx = 0;
    vislib::SingleLinkedList<megamol::viewer::Window*>::Iterator iter
        = this->windows.GetIterator();
    while (iter.HasNext()) {
        winz[idx++] = iter.Next();
    }
    for (unsigned int i = 0; i < idx; i++) {
        winz[i]->Close();
    }
}


/*
 * megamol::viewer::Viewer::RequestAppTermination
 */
void megamol::viewer::Viewer::RequestAppTermination(void) {
    this->appTerminate.Call(*this);
}


/*
 * megamol::viewer::Viewer::PresentationModeSwap
 */
void megamol::viewer::Viewer::PresentationModeSwap(void) {
    // this method is only called if there is at least one presentation window
    vislib::SingleLinkedList<Window*>::Iterator iter = this->windows.GetIterator();
    while (iter.HasNext()) {
        if (iter.Next()->GetPresentationMode() == 1) {
            return; // at least one presentation window is not ready
        }
    }
    // if we reach here all presentation windows are ready
    int ow = ::glutGetWindow();

    iter = this->windows.GetIterator();
    while (iter.HasNext()) { iter.Next()->PresentationModeSwap(); }
    iter = this->windows.GetIterator();
    while (iter.HasNext()) { iter.Next()->PresentationModeUpdate(); }
    iter = this->windows.GetIterator();
    while (iter.HasNext()) { iter.Next()->PresentationModeRefresh(); }

    if (ow != 0) ::glutSetWindow(ow);
}
