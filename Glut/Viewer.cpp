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
}


/*
 * megamol::viewer::Viewer::Initialise
 */
bool megamol::viewer::Viewer::Initialise(void) {
    int argc = 1;
    char *argv[] = {
#ifdef _WIN32
        "dummyApp",
        "-direct",  // force direct rendering context. no indirect crap!
#else /* _WIN32 */
        strdup("dummyApp"),
        strdup("-direct"),
#endif /* _WIN32 */
        NULL}; // faked command line
    ::glutInit(&argc, argv);
    // TODO: glutInitDisplayMode should be configurable?
    ::glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    ::glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
        GLUT_ACTION_CONTINUE_EXECUTION);
#ifndef _WIN32
    free(argv[0]);
#endif /* !_WIN32 */
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
