/*
 * Runnable.cpp  15.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/Runnable.h"

#include "the/stack_trace.h"


/*
 * vislib::sys::Runnable::~Runnable
 */
vislib::sys::Runnable::~Runnable(void) {
    THE_STACK_TRACE;
}


/*
 * vislib::sys::Runnable::OnThreadStarted
 */
void vislib::sys::Runnable::OnThreadStarted(void *userData) {
    THE_STACK_TRACE;
}


/*
 * vislib::sys::Runnable::OnThreadStarting
 */
void vislib::sys::Runnable::OnThreadStarting(void *userData) {
    THE_STACK_TRACE;
}


/*
 * vislib::sys::Runnable::Terminate
 */
bool vislib::sys::Runnable::Terminate(void) {
    THE_STACK_TRACE;
    return false;
}
