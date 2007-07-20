/*
 * NamedPipe.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/NamedPipe.h"
#include "vislib/File.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/MissingImplementationException.h"
#include "vislib/Path.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemException.h"

#ifndef _WIN32
#include <sys/types.h>
#include <sys/stat.h>
#include <linux/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include "vislib/Trace.h"
#include "vislib/error.h"
#endif /* !_WIN32 */


/** size of the pipes buffer */
#define PIPE_BUFFER_SIZE 4096


#ifndef _WIN32
/*
 * linuxConsumeBrokenPipes
 */
void linuxConsumeBrokenPipes(int) {
    /* do nothing! More important do not exit!!! */
}


/** the file system base directory for the pipe nodes */
const char vislib::sys::NamedPipe::baseDir[] = "/tmp/vislibpipes";
#endif /* !_WIN32 */


/*
 * vislib::sys::NamedPipe::NamedPipe
 */
vislib::sys::NamedPipe::NamedPipe(void) 
        : handle(
#ifdef _WIN32
        INVALID_HANDLE_VALUE
#else /* _WIN32 */ 
        0
#endif /* _WIN32 */ 
        ), mode(PIPE_MODE_NONE) {

#ifndef _WIN32
    ::signal(SIGPIPE, linuxConsumeBrokenPipes);
#endif /* !_WIN32 */ 
}


/*
 * vislib::sys::NamedPipe::~NamedPipe
 */
vislib::sys::NamedPipe::~NamedPipe(void) {
    if (this->IsOpen()) {
        this->Close();
    }
}


/*
 * vislib::sys::NamedPipe::Close
 */
void vislib::sys::NamedPipe::Close(void) {
    if (this->IsOpen()) {

#ifdef _WIN32

        ::FlushFileBuffers(this->handle);
        if (!this->isClient) {
            // flush server side pipe
            ::DisconnectNamedPipe(this->handle); 
        }
        ::CloseHandle(this->handle);

        this->handle = INVALID_HANDLE_VALUE;


#else /* _WIN32 */

        ::close(this->handle);
        this->handle = 0;

        // TODO: Think of a way of cleaning up the pipe file

#endif /* _WIN32 */ 

        this->mode = PIPE_MODE_NONE;
    }
}
        

/*
 * vislib::sys::NamedPipe::Open
 */
void vislib::sys::NamedPipe::Open(vislib::StringA name, 
        vislib::sys::NamedPipe::PipeMode mode) {

    // check parameters
    if (!this->checkPipeName(name)) {
        throw IllegalParamException("name", __FILE__, __LINE__);
    }
    if (mode == vislib::sys::NamedPipe::PIPE_MODE_NONE) {
        throw IllegalParamException("mode", __FILE__, __LINE__);
    }

    // close old pipe
    if (this->IsOpen()) {
        this->Close();
    }

    vislib::StringA pipeName = PipeSystemName(name);

#ifdef _WIN32
    
    this->handle = ::CreateNamedPipeA(pipeName.PeekBuffer(), 
        FILE_FLAG_FIRST_PIPE_INSTANCE | 
        ((mode == PIPE_MODE_READ) ? PIPE_ACCESS_INBOUND : PIPE_ACCESS_OUTBOUND),
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE, 2, PIPE_BUFFER_SIZE, 
        PIPE_BUFFER_SIZE, NMPWAIT_USE_DEFAULT_WAIT, NULL);

    this->isClient = false;    

    if (this->handle == INVALID_HANDLE_VALUE) {
        DWORD lastError = ::GetLastError();

        if (lastError == ERROR_ACCESS_DENIED) {
            // pipe already created! So we are the client

            this->handle = ::CreateFileA(pipeName.PeekBuffer(), 
                ((mode == PIPE_MODE_WRITE) ? GENERIC_WRITE : GENERIC_READ), 
                0, NULL, OPEN_EXISTING, 0, 0);
    
            this->isClient = true;

            if (this->handle == INVALID_HANDLE_VALUE) {
                throw vislib::sys::SystemException(__FILE__, __LINE__);
            }

        } else {
            throw vislib::sys::SystemException(__FILE__, __LINE__);
        }
    }

    if (!this->isClient) {
        DWORD lastError;
        bool connected = ::ConnectNamedPipe(this->handle, NULL) ? true 
            : ((lastError = GetLastError()) == ERROR_PIPE_CONNECTED); 

        if (!connected) {
            // pipe broken before even creating
            ::CloseHandle(this->handle);
            this->handle = INVALID_HANDLE_VALUE;
            this->mode = PIPE_MODE_NONE;

            throw vislib::sys::SystemException(lastError, __FILE__, __LINE__);
        }
    }

#else /* _WIN32 */

//    TRACE(VISLIB_TRCELVL_INFO, "vislib::sys::NamedPipe::Open\n");

    if (!vislib::sys::File::IsDirectory(this->baseDir)) {
        vislib::sys::Path::MakeDirectory(this->baseDir);
    }

    // Create the FIFO if it does not exist
    mode_t oldMask = ::umask(0);
    if (::mknod(pipeName.PeekBuffer(), S_IFIFO | 0666, 0) != 0) {

//        TRACE(vislib::Trace::LEVEL_VL_INFO, "mknod failed\n");

        DWORD lastError = ::GetLastError();
        if (lastError != EEXIST) {
            ::umask(oldMask);
            throw vislib::sys::SystemException(lastError, __FILE__, __LINE__);
        }
    }
    ::umask(oldMask);

    this->handle = ::open(pipeName.PeekBuffer(), O_SYNC | 
        ((mode == PIPE_MODE_READ) ? O_RDONLY : O_WRONLY));
    if (!this->handle) {

//        TRACE(vislib::Trace::LEVEL_VL_INFO, "open failed\n");

        throw vislib::sys::SystemException(__FILE__, __LINE__);
    }

    /* Tricky.
     * This works since 'open' blocks until both ends of the pipe are opend.
     */
    ::unlink(pipeName.PeekBuffer());

#endif /* _WIN32 */ 

    this->mode = mode;

}


/*
 * vislib::sys::NamedPipe::Open
 */
void vislib::sys::NamedPipe::Open(vislib::StringW name, 
        vislib::sys::NamedPipe::PipeMode mode) {

#ifdef _WIN32

    // check parameters
    if (!this->checkPipeName(name)) {
        throw IllegalParamException("name", __FILE__, __LINE__);
    }
    if (mode == vislib::sys::NamedPipe::PIPE_MODE_NONE) {
        throw IllegalParamException("mode", __FILE__, __LINE__);
    }

    // close old pipe
    if (this->IsOpen()) {
        this->Close();
    }

    vislib::StringW pipeName = PipeSystemName(name);
    
    this->handle = ::CreateNamedPipeW(pipeName.PeekBuffer(), 
        FILE_FLAG_FIRST_PIPE_INSTANCE | 
        ((mode == PIPE_MODE_READ) ? PIPE_ACCESS_INBOUND : PIPE_ACCESS_OUTBOUND),
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE, 2, PIPE_BUFFER_SIZE, 
        PIPE_BUFFER_SIZE, NMPWAIT_USE_DEFAULT_WAIT, NULL);

    this->isClient = false;    

    if (this->handle == INVALID_HANDLE_VALUE) {
        DWORD lastError = ::GetLastError();

        if (lastError == ERROR_ACCESS_DENIED) {
            // pipe already created! So we are the client

            this->handle = ::CreateFileW(pipeName.PeekBuffer(), 
                ((mode == PIPE_MODE_WRITE) ? GENERIC_WRITE : GENERIC_READ), 
                0, NULL, OPEN_EXISTING, 0, 0);
    
            this->isClient = true;

            if (this->handle == INVALID_HANDLE_VALUE) {
                throw vislib::sys::SystemException(__FILE__, __LINE__);
            }

        } else {
            throw vislib::sys::SystemException(__FILE__, __LINE__);
        }
    }

    if (!this->isClient) {
        DWORD lastError;
        bool connected = ::ConnectNamedPipe(this->handle, NULL) ? true 
            : ((lastError = GetLastError()) == ERROR_PIPE_CONNECTED); 

        if (!connected) {
            // pipe broken before even creating
            ::CloseHandle(this->handle);
            this->handle = INVALID_HANDLE_VALUE;
            this->mode = PIPE_MODE_NONE;

            throw vislib::sys::SystemException(lastError, __FILE__, __LINE__);
        }
    }

    this->mode = mode;

#else /* _WIN32 */

    // since linux and unicode do not really work (as always)           
    this->Open(W2A(name), mode);

#endif /* _WIN32 */ 
}


/*
 * vislib::sys::NamedPipe::Read
 */
void vislib::sys::NamedPipe::Read(void *buffer, unsigned int size) {

    if (this->mode != PIPE_MODE_READ) {
        throw IllegalStateException("Pipe not in read mode", __FILE__, __LINE__);
    }
    if (size == 0) return;
    if (buffer == NULL) {
        throw IllegalParamException("buffer", __FILE__, __LINE__);
    }

#ifdef _WIN32
    DWORD outRead;
#else /* _WIN32 */
    ssize_t outRead;
#endif /* _WIN32 */ 

    while (size > 0) {

#ifdef _WIN32

        if (!::ReadFile(this->handle, buffer, size, &outRead, NULL)) {
            // pipe broken
            ::FlushFileBuffers(this->handle);
            if (!this->isClient) {
                ::DisconnectNamedPipe(this->handle); 
            }
            ::CloseHandle(this->handle);
            this->handle = INVALID_HANDLE_VALUE;
            this->mode = PIPE_MODE_NONE;
            throw vislib::sys::SystemException(__FILE__, __LINE__);
        }

#else /* _WIN32 */

        //TRACE(vislib::Trace::LEVEL_VL_INFO, "Read (%d)\n", size);

        outRead = ::read(this->handle, buffer, size);

        //TRACE(vislib::Trace::LEVEL_VL_INFO, "Read returned %d\n", outRead);

        if (outRead <= 0) {

            //TRACE(vislib::Trace::LEVEL_VL_INFO, "feof or ferror\n");

            ::close(this->handle);
            this->handle = 0;
            this->mode = PIPE_MODE_NONE;

            //TRACE(vislib::Trace::LEVEL_VL_INFO, "feof or ferror ... Done!\n");

            throw vislib::sys::SystemException(__FILE__, __LINE__);
        }

#endif /* _WIN32 */ 

        buffer = static_cast<void*>(static_cast<char*>(buffer) + outRead);
        size -= outRead;
    }
}


/*
 * vislib::sys::NamedPipe::PipeSystemName
 */
vislib::StringA vislib::sys::NamedPipe::PipeSystemName(const vislib::StringA &name) {
    if (!checkPipeName(name)) {
        throw IllegalParamException("name", __FILE__, __LINE__);
    }

#ifdef _WIN32
    vislib::StringA pipeName = "\\\\.\\pipe\\";
    pipeName += name;
#else /* _WIN32 */
    vislib::StringA pipeName = baseDir;
    pipeName += "/";
    pipeName += name;
#endif /* _WIN32 */
    return pipeName;
}


/*
 * vislib::sys::NamedPipe::PipeSystemName
 */
vislib::StringW vislib::sys::NamedPipe::PipeSystemName(const vislib::StringW &name) {
    if (!checkPipeName(name)) {
        throw IllegalParamException("name", __FILE__, __LINE__);
    }

#ifdef _WIN32
    vislib::StringW pipeName = L"\\\\.\\pipe\\";
    pipeName += name;
#else /* _WIN32 */
    vislib::StringW pipeName = A2W(baseDir);
    pipeName += L"/";
    pipeName += name;
#endif /* _WIN32 */
    return pipeName;
}


/*
 * vislib::sys::NamedPipe::Write
 */
void vislib::sys::NamedPipe::Write(void *buffer, unsigned int size) {

    if (this->mode != PIPE_MODE_WRITE) {
        throw IllegalStateException("Pipe not in write mode", __FILE__, __LINE__);
    }
    if (size == 0) return;
    if (buffer == NULL) {
        throw IllegalParamException("buffer", __FILE__, __LINE__);
    }

#ifdef _WIN32
    DWORD outWritten;
#else /* _WIN32 */
    ssize_t outWritten;
#endif /* _WIN32 */ 

    while (size > 0) {

#ifdef _WIN32

        if (!::WriteFile(this->handle, buffer, size, &outWritten, NULL)) {
            // pipe broken
            ::CloseHandle(this->handle);
            this->handle = INVALID_HANDLE_VALUE;
            this->mode = PIPE_MODE_NONE;
            throw vislib::sys::SystemException(__FILE__, __LINE__);
        }

#else /* _WIN32 */

        //TRACE(vislib::Trace::LEVEL_VL_INFO, "Write (%d)\n", size);

        outWritten = ::write(this->handle, buffer, size);

        //TRACE(vislib::Trace::LEVEL_VL_INFO, "Write returned %d\n", outWritten);

        if (outWritten <= 0) {

            //TRACE(vislib::Trace::LEVEL_VL_INFO, "feof or ferror\n");

            ::close(this->handle);
            this->handle = 0;
            this->mode = PIPE_MODE_NONE;

            //TRACE(vislib::Trace::LEVEL_VL_INFO, "feof or ferror ... Done!\n");

            throw vislib::sys::SystemException(__FILE__, __LINE__);

        }

#endif /* _WIN32 */ 

        buffer = static_cast<void*>(static_cast<char*>(buffer) + outWritten);
        size -= outWritten;
    }
}


/*
 * vislib::sys::NamedPipe::checkPipeName
 */
template<class T> bool vislib::sys::NamedPipe::checkPipeName(
        const vislib::String<T> &name) {
    if (name.Find(static_cast<typename T::Char>(vislib::sys::Path::SEPARATOR_A))
           != vislib::String<T>::INVALID_POS) {
        return false;
    }
    if (name.Find(static_cast<typename T::Char>('<'))
           != vislib::String<T>::INVALID_POS) {
        return false;
    }
    if (name.Find(static_cast<typename T::Char>('>'))
           != vislib::String<T>::INVALID_POS) {
        return false;
    }
    if (name.Find(static_cast<typename T::Char>('|'))
           != vislib::String<T>::INVALID_POS) {
        return false;
    }
    if (name.Find(static_cast<typename T::Char>(':'))
           != vislib::String<T>::INVALID_POS) {
        return false;
    }
    if (name.Find(static_cast<typename T::Char>('*'))
           != vislib::String<T>::INVALID_POS) {
        return false;
    }
    if (name.Find(static_cast<typename T::Char>('?'))
           != vislib::String<T>::INVALID_POS) {
        return false;
    }

    return true;
}
