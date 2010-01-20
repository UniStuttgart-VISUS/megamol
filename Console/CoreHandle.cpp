/*
 * CoreHandle.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CoreHandle.h"
#include "MegaMolCore.h"
#include "MegaMolViewer.h"
#ifndef _WIN32
#include <string.h> // for memset
#endif
#include "vislib/mathfunctions.h"
#include "vislib/memutils.h"


/*
 * megamol::console::CoreHandle::MegaMolHandle
 */
megamol::console::CoreHandle::CoreHandle(void) 
        : hndl(NULL), size(0) {
    // intentionally empty
}


/*
 * megamol::console::CoreHandle::~MegaMolHandle
 */
megamol::console::CoreHandle::~CoreHandle(void) {
    this->DestroyHandle();
    ARY_SAFE_DELETE(this->hndl);
    this->size = 0;
}


/*
 * megamol::console::CoreHandle::DestroyHandle
 */
void megamol::console::CoreHandle::DestroyHandle(void) {
    ::mmcDisposeHandle(this->hndl); // this is ugly, but it works, since it is
#ifndef MEGAMOLVIEWER_USESTATIC     // save to dispose invalid handles.
    if (::mmvDisposeHandle != NULL) {
#endif /* !MEGAMOLVIEWER_USESTATIC */
        ::mmvDisposeHandle(this->hndl);
#ifndef MEGAMOLVIEWER_USESTATIC
    }
#endif /* !MEGAMOLVIEWER_USESTATIC */
    ::memset(this->hndl, 0, this->size);
}


/*
 * megamol::console::CoreHandle::operator void *
 */
megamol::console::CoreHandle::operator void *(void) const { 
    if (this->hndl == NULL) {
        unsigned int size = ::mmcGetHandleSize();
#ifndef MEGAMOLVIEWER_USESTATIC
        if (::mmvGetHandleSize != NULL) {
#endif /* !MEGAMOLVIEWER_USESTATIC */
            if (size < ::mmvGetHandleSize()) {
                size = ::mmvGetHandleSize();
            }
#ifndef MEGAMOLVIEWER_USESTATIC
        }
#endif /* !MEGAMOLVIEWER_USESTATIC */
        if (this->size < size) {
            ASSERT(!this->IsValid());

            this->size = size;

            delete[] this->hndl;
            this->hndl = new unsigned char[this->size];
            ::memset(this->hndl, 0, this->size);
        }
    }
    return static_cast<void*>(this->hndl);
}
