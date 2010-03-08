/*
 * Runnable.cpp  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/Runnable.h"

#include "vislib/StackTrace.h"


/*
 * vislib::sys::Runnable::~Runnable
 */
vislib::sys::Runnable::~Runnable(void) {
    VLSTACKTRACE("Runnable::~Runnable", __FILE__, __LINE__);
}


/*
 * vislib::sys::Runnable::OnThreadStarted
 */
void vislib::sys::Runnable::OnThreadStarted(void *userData) {
    VLSTACKTRACE("Runnable::OnThreadStarted", __FILE__, __LINE__);
}


/*
 * vislib::sys::Runnable::OnThreadStarting
 */
void vislib::sys::Runnable::OnThreadStarting(void *userData) {
    VLSTACKTRACE("Runnable::OnThreadStarting", __FILE__, __LINE__);
}


/*
 * vislib::sys::Runnable::Terminate
 */
bool vislib::sys::Runnable::Terminate(void) {
    VLSTACKTRACE("Runnable::Terminate", __FILE__, __LINE__);
    return false;
}
