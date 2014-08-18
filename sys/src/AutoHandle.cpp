/*
 * AutoHandle.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AutoHandle.h"

#include "vislib/SystemException.h"


#ifdef _WIN32
/*
 * vislib::sys::AutoHandle::AutoHandle
 */
vislib::sys::AutoHandle::AutoHandle(const bool isNull) 
        : handle(isNull ? NULL : INVALID_HANDLE_VALUE) {
}


/*
 * vislib::sys::AutoHandle::AutoHandle
 */
vislib::sys::AutoHandle::AutoHandle(HANDLE handle, const bool takeOwnership) 
        : handle(INVALID_HANDLE_VALUE) {
    this->Set(handle, takeOwnership);
}


/*
 * vislib::sys::AutoHandle::AutoHandle
 */
vislib::sys::AutoHandle::AutoHandle(const AutoHandle& rhs) 
        : handle(INVALID_HANDLE_VALUE) {
    this->Set(rhs.handle, false);
}


/*
 * vislib::sys::AutoHandle::~AutoHandle
 */
vislib::sys::AutoHandle::~AutoHandle(void) {
    this->Close();
}


/*
 * vislib::sys::AutoHandle::Close
 */
void vislib::sys::AutoHandle::Close(void) {
    if (this->IsValid()) {
        ::CloseHandle(this->handle);
        this->handle = INVALID_HANDLE_VALUE;
    }
}


/*
 * vislib::sys::AutoHandle::Set
 */
void vislib::sys::AutoHandle::Set(HANDLE handle, const bool takeOwnership) {
    if (this->handle != handle) {
        this->Close();

        if (takeOwnership) {
            this->handle = handle;
        } else {
            // Note: ::GetCurrentProcess() returns a pseudo-handle which needs 
            // not to be closed.
            if (!::DuplicateHandle(::GetCurrentProcess(), handle, 
                    ::GetCurrentProcess(), &(this->handle), 0, FALSE, 
                    DUPLICATE_SAME_ACCESS)) {
                SystemException(__FILE__, __LINE__);
            }   
        }
    }
}

#endif /* _WIN32 */
