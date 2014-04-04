/*
 * SharedMemory.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/SharedMemory.h"

#ifndef _WIN32
#include <sys/mman.h>
#include <fcntl.h>
#endif /* _WIN32 */

#include "the/argument_exception.h"
#include "the/memory.h"
#include "the/text/string_converter.h"
#include "vislib/sysfunctions.h"
#include "the/system/system_exception.h"
#include "the/trace.h"
#include "the/not_supported_exception.h"


/*
 * vislib::sys::SharedMemory::SharedMemory
 */
vislib::sys::SharedMemory::SharedMemory(void) : mapping(NULL) {
#ifdef _WIN32
    this->hSharedMem = NULL;
#else /* _WIN32 */
    this->hSharedMem = -1;
    this->size = 0;
#endif /* _WIN32 */
}


/*
 * vislib::sys::SharedMemory::~SharedMemory
 */
vislib::sys::SharedMemory::~SharedMemory(void) {
    try {
        this->Close();
    } catch (...) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Exception in SharedMemory dtor.\n");
    }
}


/*
 * vislib::sys::SharedMemory::Close
 */
void vislib::sys::SharedMemory::Close(void) {
#ifdef _WIN32
    if (this->mapping != NULL) {
        if (!::UnmapViewOfFile(this->mapping)) {
            throw the::system::system_exception(__FILE__, __LINE__);
        }
        this->mapping = NULL;
    }

    if (this->hSharedMem != NULL) {
        if (!::CloseHandle(this->hSharedMem)) {
            throw the::system::system_exception(__FILE__, __LINE__);
        }
        this->hSharedMem = NULL;
    }

#else /* _WIN32 */
    if (this->mapping != NULL) {
        if (::munmap(this->mapping, this->size) == -1) {
            throw the::system::system_exception(__FILE__, __LINE__);
        }
        this->mapping = NULL;
        this->size = 0;
    }

    if (this->hSharedMem != -1) {
        if (::shm_unlink(this->name.c_str()) == -1) {
            throw the::system::system_exception(__FILE__, __LINE__);
        }
        if (::close(this->hSharedMem) == -1) {
            throw the::system::system_exception(__FILE__, __LINE__);
        }
        this->hSharedMem = -1;
        this->name.clear();
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
    return ((this->hSharedMem != -1) && (this->mapping != NULL));
#endif /* _WIN32 */
}


/*
 * vislib::sys::SharedMemory::Open
 */
void vislib::sys::SharedMemory::Open(const char *name, const AccessMode accessMode, 
        const CreationMode creationMode, const FileSize size) {
#ifdef _WIN32
    unsigned int protect = 0;
    unsigned int access = 0;
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

    if (creationMode == OPEN_ONLY) {
        this->hSharedMem = ::OpenFileMappingA(access, FALSE, name);
        if (this->hSharedMem == NULL) {
            throw the::system::system_exception(__FILE__, __LINE__);
        }

    } else {
        this->hSharedMem = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, 
            protect, static_cast<unsigned int>(size >> 32), 
            static_cast<unsigned int>(size & 0xFFFFFFFF), name);
        if (this->hSharedMem == NULL) {
            throw the::system::system_exception(__FILE__, __LINE__);
        } else if (creationMode == CREATE_ONLY) {
            /* Handle exclusive open by closing any exisiting mapping. */
            unsigned int error = ::GetLastError();
            if (error == ERROR_ALREADY_EXISTS) {
                ::CloseHandle(this->hSharedMem);
                this->hSharedMem = NULL;
                throw the::system::system_exception(error, __FILE__, __LINE__);
            }
        }
    }

    this->mapping = ::MapViewOfFile(this->hSharedMem, access, 0, 0, 
        static_cast<size_t>(size));
    if (this->mapping == NULL) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    int oflags = 0;
    int protect = 0;
    switch (accessMode) {
        case READ_ONLY: 
            oflags |= O_RDONLY;
            protect |= PROT_READ;
            break;

        case READ_WRITE: 
            oflags |= O_RDWR;
            protect |= PROT_READ | PROT_WRITE;
            break;
    }
    switch (creationMode) {
        case CREATE_ONLY:
            oflags |= O_EXCL;
            /* falls through. */
        case OPEN_CREATE:
            oflags |= O_CREAT;
            break;

        default:
            break;
    }

    this->name = TranslateWinIpc2PosixName(name);
    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Open POSIX shared memory \"%s\"\n", 
        this->name.c_str());
    this->hSharedMem = ::shm_open(this->name.c_str(), oflags, DFT_MODE);
    if (this->hSharedMem == -1) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    this->size = static_cast<size_t>(size);
    // TODO: This is problematic in conjunction with open
    if (::ftruncate(this->hSharedMem, this->size) == -1) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    this->mapping = ::mmap(NULL, this->size, protect, MAP_SHARED, 
        this->hSharedMem, 0);
    if (this->mapping == MAP_FAILED) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

#endif /* _WIN32 */
}


/*
 * vislib::sys::SharedMemory::Open
 */
void vislib::sys::SharedMemory::Open(const wchar_t *name, const AccessMode accessMode, 
        const CreationMode creationMode, const FileSize size) {
#ifdef _WIN32
    unsigned int protect = 0;
    unsigned int access = 0;
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

    if (creationMode == OPEN_ONLY) {
        this->hSharedMem = ::OpenFileMappingW(access, FALSE, name);
        if (this->hSharedMem == NULL) {
            throw the::system::system_exception(__FILE__, __LINE__);
        }

    } else {
        this->hSharedMem = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, 
            protect, static_cast<unsigned int>(size >> 32), 
            static_cast<unsigned int>(size & 0xFFFFFFFF), name);
        if (this->hSharedMem == NULL) {
            throw the::system::system_exception(__FILE__, __LINE__);
        } else if (creationMode == CREATE_ONLY) {
            /* Handle exclusive open by closing any exisiting mapping. */
            unsigned int error = ::GetLastError();
            if (error == ERROR_ALREADY_EXISTS) {
                ::CloseHandle(this->hSharedMem);
                this->hSharedMem = NULL;
                throw the::system::system_exception(error, __FILE__, __LINE__);
            }
        }
    }

    this->mapping = ::MapViewOfFile(this->hSharedMem, access, 0, 0, 
        static_cast<size_t>(size));
    if (this->mapping == NULL) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

#else /* _WIN32 */
    this->Open(THE_W2A(name), accessMode, creationMode, size);
#endif /* _WIN32 */
}


#ifndef _WIN32
/*
 * vislib::sys::SharedMemory::DFT_MODE
 */
const mode_t vislib::sys::SharedMemory::DFT_MODE = 0666;
#endif /* !_WIN32 */


/*
 * vislib::sys::SharedMemory::SharedMemory
 */
vislib::sys::SharedMemory::SharedMemory(const SharedMemory& rhs) {
    throw the::not_supported_exception("SharedMemory", __FILE__, __LINE__);
}


/*
 * vislib::sys::SharedMemory::operator =
 */
vislib::sys::SharedMemory& vislib::sys::SharedMemory::operator =(
        const SharedMemory& rhs) {
    if (this != &rhs) {
        throw the::argument_exception("rhs", __FILE__, __LINE__);
    }

    return *this;
}
