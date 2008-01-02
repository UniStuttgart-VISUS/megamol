/*
 * SharedMemory.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/SharedMemory.h"

#ifndef _WIN32
#include <sys/shm.h>
#endif /* _WIN32 */

#include "vislib/IllegalParamException.h"
#include "vislib/memutils.h"
#include "vislib/StringConverter.h"
#include "vislib/sysfunctions.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::sys::SharedMemory::SharedMemory
 */
vislib::sys::SharedMemory::SharedMemory(void) : mapping(NULL) {
#ifdef _WIN32
    this->hSharedMem = NULL;
#else /* _WIN32 */
    this->id = -1;
#endif /* _WIN32 */
}


/*
 * vislib::sys::SharedMemory::~SharedMemory
 */
vislib::sys::SharedMemory::~SharedMemory(void) {
    this->Close();
}


/*
 * vislib::sys::SharedMemory::Close
 */
void vislib::sys::SharedMemory::Close(void) {
#ifdef _WIN32
    if (this->mapping != NULL) {
        if (!::UnmapViewOfFile(this->mapping)) {
            throw SystemException(__FILE__, __LINE__);
        }
        this->mapping = NULL;
    }

    if (this->hSharedMem != NULL) {
        if (!::CloseHandle(this->hSharedMem)) {
            throw SystemException(__FILE__, __LINE__);
        }
        this->hSharedMem = NULL;
    }

#else /* _WIN32 */
    if (this->mapping != NULL) {
        if (::shmdt(this->mapping) == -1) {
            throw SystemException(__FILE__, __LINE__);
        }
        this->mapping = NULL;
    }

    if (this->id != -1) {
        // TODO: das könnte auf der "client-seite" in die luft gehen.
        if (::shmctl(this->id, IPC_RMID, 0) == -1) {
            throw SystemException(__FILE__, __LINE__);
        }
        this->id = -1;
    }
#endif /* _WIN32 */
}


/*
 * vislib::sys::SharedMemory::IsOpen
 */
bool vislib::sys::SharedMemory::IsOpen(void) const {
#ifdef _WIN32
    return ((this->hSharedMem != NULL) && (this->mapping != NULL));
#else /* _WIN32 */
    return ((this->id != -1) && (this->mapping != NULL));
#endif /* _WIN32 */
}


/*
 * vislib::sys::SharedMemory::Open
 */
void vislib::sys::SharedMemory::Open(const char *name, const AccessMode accessMode, 
        const CreationMode creationMode, const FileSize size) {
#ifdef _WIN32
    DWORD protect = 0;
    DWORD access = 0;
    switch (accessMode) {
        case READ_ONLY: 
            protect = PAGE_READONLY; 
            access = FILE_MAP_READ;
            break;
        case READ_WRITE: 
            protect = PAGE_READWRITE;
            access = FILE_MAP_WRITE;    // WRITE implies READ.
            break;
    }

    if (creationMode == CREATE_ONLY) {
        this->hSharedMem = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, 
            protect, static_cast<DWORD>(size >> 32), 
            static_cast<DWORD>(size & 0xFFFFFFFF), name);
        if (this->hSharedMem == NULL) {
            throw SystemException(__FILE__, __LINE__);
        }

    } else {
        ASSERT(creationMode == OPEN_ONLY);
        this->hSharedMem = ::OpenFileMappingA(access, FALSE, name);
        if (this->hSharedMem == NULL) {
            throw SystemException(__FILE__, __LINE__);
        }
    }

    this->mapping = ::MapViewOfFile(this->hSharedMem, access, 0, 0, 
        static_cast<SIZE_T>(size));
    if (this->mapping == NULL) {
        throw SystemException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    int flags = 0;
    switch (accessMode) {
        case READ_WRITE: 
            flags |= SHM_W;
            /* falls through. */
        case READ_ONLY: 
            flags |= SHM_R; 
            break;
    }
    switch (creationMode) {
        case CREATE_ONLY:
            flags |= IPC_CREAT;
            break;

        default:
            break;
    }

    this->id = ::shmget(TranslateIpcName(name), static_cast<size_t>(size), 
        flags);
    if (this->id == -1) {
        throw SystemException(__FILE__, __LINE__);
    }

    this->mapping = ::shmat(this->id, NULL, 0);
    if (this->mapping == reinterpret_cast<void *>(-1)) {
        throw SystemException(__FILE__, __LINE__);
    }
#endif /* _WIN32 */
}


/*
 * vislib::sys::SharedMemory::Open
 */
void vislib::sys::SharedMemory::Open(const wchar_t *name, const AccessMode accessMode, 
        const CreationMode creationMode, const FileSize size) {
#ifdef _WIN32
    DWORD protect = 0;
    DWORD access = 0;
    switch (accessMode) {
        case READ_ONLY: 
            protect = PAGE_READONLY; 
            access = FILE_MAP_READ;
            break;
        case READ_WRITE: 
            protect = PAGE_READWRITE;
            access = FILE_MAP_WRITE;    // WRITE implies READ.
            break;
    }

    if (creationMode == CREATE_ONLY) {
        this->hSharedMem = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, 
            protect, static_cast<DWORD>(size >> 32), 
            static_cast<DWORD>(size & 0xFFFFFFFF), name);
        if (this->hSharedMem == NULL) {
            throw SystemException(__FILE__, __LINE__);
        }

    } else {
        ASSERT(creationMode == OPEN_ONLY);
        this->hSharedMem = ::OpenFileMappingW(access, FALSE, name);
        if (this->hSharedMem == NULL) {
            throw SystemException(__FILE__, __LINE__);
        }
    }

    this->mapping = ::MapViewOfFile(this->hSharedMem, access, 0, 0, 
        static_cast<SIZE_T>(size));
    if (this->mapping == NULL) {
        throw SystemException(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    this->Open(W2A(name), accessMode, creationMode, size);
#endif /* _WIN32 */
}


/*
 * vislib::sys::SharedMemory::SharedMemory
 */
vislib::sys::SharedMemory::SharedMemory(const SharedMemory& rhs) {
    throw UnsupportedOperationException("SharedMemory", __FILE__, __LINE__);
}


/*
 * vislib::sys::SharedMemory::operator =
 */
vislib::sys::SharedMemory& vislib::sys::SharedMemory::operator =(
        const SharedMemory& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
