/*
 * Runnable.cpp  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/sys/Runnable.h"


/*
 * vislib::sys::Runnable::~Runnable
 */
vislib::sys::Runnable::~Runnable(void) {}


/*
 * vislib::sys::Runnable::OnThreadStarted
 */
void vislib::sys::Runnable::OnThreadStarted(void* userData) {}


/*
 * vislib::sys::Runnable::OnThreadStarting
 */
void vislib::sys::Runnable::OnThreadStarting(void* userData) {}


/*
 * vislib::sys::Runnable::Terminate
 */
bool vislib::sys::Runnable::Terminate(void) {
    return false;
}
