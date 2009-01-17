/*
 * NamedPipe.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/NamedPipe.h"
#include "vislib/error.h"
#include "vislib/File.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/MissingImplementationException.h"
#include "vislib/Path.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"

#ifndef _WIN32
#include <sys/types.h>
#include <sys/stat.h>
#include <linux/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include "vislib/Thread.h"
#include "vislib/PerformanceCounter.h"
#endif /* !_WIN32 */


/** size of the pipes buffer */
#define PIPE_BUFFER_SIZE 4096


#ifndef _WIN32
namespace vislib {
namespace sys {

/*
 * linuxConsumeBrokenPipes
 */
void linuxConsumeBrokenPipes(int) {
    /* do nothing! More important do not exit!!! */
}

} /* end namespace sys */
} /* end namespace vislib */


/** the file system base directory for the pipe nodes */
const char vislib::sys::NamedPipe::baseDir[] = "/tmp/vislibpipes";


/*****************************************************************************/


/*
 * vislib::sys::NamedPipe::Exterminatus::Exterminatus
 */
vislib::sys::NamedPipe::Exterminatus::Exterminatus(unsigned int timeout, 
        const char *pipename, bool readend) 
        : Runnable(), connected(false), terminated(false), timeout(timeout),
        pipename(pipename), readend(readend) {
}


/*
 * vislib::sys::NamedPipe::Exterminatus::~Exterminatus
 */
vislib::sys::NamedPipe::Exterminatus::~Exterminatus(void) {
}


/*
 * vislib::sys::NamedPipe::Exterminatus::Run
 */
DWORD vislib::sys::NamedPipe::Exterminatus::Run(void *userData) {
    PerformanceCounter counter;

    while (!this->connected && (static_cast<unsigned int>(counter.Difference()) < this->timeout)) {
        Thread::Sleep(1);
    }

    if (!this->connected) {
        // not connected after timeout
        this->terminated = true;

        int terminator = ::open(pipename, O_SYNC | (this->readend ? O_RDONLY : O_WRONLY));
        if (terminator >= 0) {
            ::close(terminator);
        }
    }

    return 0;
}


/*****************************************************************************/
#endif /* !_WIN32 */


/*
 * vislib::sys::NamedPipe::NamedPipe
 */
vislib::sys::NamedPipe::NamedPipe(void) 
        : handle(
#ifdef _WIN32
        INVALID_HANDLE_VALUE
#else /* _WIN32 */ 
        -1
#endif /* _WIN32 */ 
        ), mode(PIPE_MODE_NONE), cleanupLock() {

#ifdef _WIN32
    ::ZeroMemory(&this->overlapped, sizeof(OVERLAPPED));
#else /* _WIN32 */
    ::signal(SIGPIPE, linuxConsumeBrokenPipes);
#endif /* _WIN32 */ 
}


/*
 * vislib::sys::NamedPipe::~NamedPipe
 */
vislib::sys::NamedPipe::~NamedPipe(void) {
    this->Close();
}


/*
 * vislib::sys::NamedPipe::Close
 */
void vislib::sys::NamedPipe::Close(void) {

    this->cleanupLock.Lock();

#ifdef _WIN32
    if (this->handle != INVALID_HANDLE_VALUE) {
#else /* _WIN32 */
    if (this->handle >= 0) {
#endif /* _WIN32 */

#ifdef _WIN32

        try {
            ::FlushFileBuffers(this->handle);
        } catch (...) { }

        if (!this->isClient) {
            // flush server side pipe
            ::DisconnectNamedPipe(this->handle);
        }
        ::CancelIo(this->handle);
        ::CloseHandle(this->handle);

        if (this->overlapped.hEvent != NULL) {
            ::CloseHandle(this->overlapped.hEvent);
            this->overlapped.hEvent = NULL;
        }

        this->handle = INVALID_HANDLE_VALUE;

#else /* _WIN32 */

        ::close(this->handle);
        this->handle = -1;
#endif /* _WIN32 */ 

        this->mode = PIPE_MODE_NONE;
    }

    this->cleanupLock.Unlock();
}
        

/*
 * vislib::sys::NamedPipe::Open
 */
bool vislib::sys::NamedPipe::Open(vislib::StringA name, 
        vislib::sys::NamedPipe::PipeMode mode, unsigned int timeout) {

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

    if (timeout == 0) {
        timeout = INFINITE;
    }

    this->isClient = false;

    // create the overlapped structure allowing us to produce timeouts
    ::ZeroMemory(&this->overlapped, sizeof(OVERLAPPED));
    this->overlapped.hEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL); // manual reset event
    
    // create the pipe
    this->handle = ::CreateNamedPipeA(pipeName.PeekBuffer(), 
        FILE_FLAG_FIRST_PIPE_INSTANCE | FILE_FLAG_OVERLAPPED | 
        ((mode == PIPE_MODE_READ) ? PIPE_ACCESS_INBOUND : PIPE_ACCESS_OUTBOUND),
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE, 2, PIPE_BUFFER_SIZE, 
        PIPE_BUFFER_SIZE, 0, NULL);

    if (this->handle == INVALID_HANDLE_VALUE) {
        // creation failed
        DWORD lastError = ::GetLastError();

        if (lastError == ERROR_ACCESS_DENIED) {
            // pipe already created! So we should be the client

            this->handle = ::CreateFileA(pipeName.PeekBuffer(), 
                ((mode == PIPE_MODE_WRITE) ? GENERIC_WRITE : GENERIC_READ), 
                0, NULL, OPEN_EXISTING, 0, 0);
    
            this->isClient = true;

            if (this->handle == INVALID_HANDLE_VALUE) {
                throw vislib::sys::SystemException(__FILE__, __LINE__);
            }

        } else {
            // creation really failed!
            throw vislib::sys::SystemException(__FILE__, __LINE__);
        }
    }

    if (!this->isClient) {
        // continue initialising the server
        ::ConnectNamedPipe(this->handle, &this->overlapped);

        switch (::WaitForSingleObject(this->overlapped.hEvent, timeout)) {
            case WAIT_OBJECT_0: {
                // event signaled so connection should be established
                DWORD dummy;
                if (::GetOverlappedResult(this->handle, &this->overlapped, &dummy, FALSE) == 0) {
                    throw vislib::sys::SystemException(__FILE__, __LINE__);
                }
                ::ResetEvent(this->overlapped.hEvent);

            } break;
            default: {
                // unknown error
                vislib::sys::SystemMessage msg(::GetLastError());
                VLTRACE(Trace::LEVEL_VL_INFO, 
                    "NamedPipe Open WaitForSigleObject Error: \"%s\"", 
                    static_cast<const char*>(msg));
            } // NO BREAK;
            case WAIT_TIMEOUT: {
                // timeout
                if (::CancelIo(this->handle) == 0) {
                    vislib::sys::SystemMessage msg(::GetLastError());
                    VLTRACE(Trace::LEVEL_VL_INFO, 
                        "NamedPipe Open Timeout: CancelIo failed \"%s\"", 
                        static_cast<const char*>(msg));
                }
                this->Close();
                return false;
            } break;
        }
    }
    ASSERT(this->handle != INVALID_HANDLE_VALUE);
    this->mode = mode;
    return true; // pipe opened

#else /* _WIN32 */
    Exterminatus ext(timeout, pipeName.PeekBuffer(), (mode != PIPE_MODE_READ));
    vislib::sys::Thread tot(&ext);

    mode_t oldMask = ::umask(0);
    if (!vislib::sys::File::IsDirectory(this->baseDir)) {
        vislib::sys::Path::MakeDirectory(this->baseDir);
    }

    // Create the FIFO if it does not exist
    if (::mknod(pipeName.PeekBuffer(), S_IFIFO | 0666, 0) != 0) {

//        VLTRACE(vislib::Trace::LEVEL_VL_INFO, "mknod failed\n");

        DWORD lastError = ::GetLastError();
        if (lastError != EEXIST) {
            ::umask(oldMask);
            throw vislib::sys::SystemException(lastError, __FILE__, __LINE__);
        }
    }
    ::umask(oldMask);

//    VLTRACE(vislib::Trace::LEVEL_VL_INFO, "vislib::sys::NamedPipe::Open\n");

    if (timeout > 0) {
        tot.Start();
    }

    this->handle = ::open(pipeName.PeekBuffer(), O_SYNC | 
        ((mode == PIPE_MODE_READ) ? O_RDONLY : O_WRONLY));
    ext.MarkConnected();

    if (timeout > 0) {
        tot.Join();
    }

    if (this->handle < 0) {
//        VLTRACE(vislib::Trace::LEVEL_VL_INFO, "open failed.\n");
        throw vislib::sys::SystemException(__FILE__, __LINE__);
    } else if (ext.IsTerminated()) {
//        VLTRACE(vislib::Trace::LEVEL_VL_INFO, "open timed out.\n");
        ::close(this->handle);
        this->handle = -1;
    }

    // Tricky.
    // This works since 'open' blocks until both ends of the pipe are opend.
    ::unlink(pipeName.PeekBuffer());

    if (this->handle >= 0) {
        this->mode = mode;
    }
    return (this->handle >= 0); // pipe opened

#endif /* _WIN32 */ 
}


/*
 * vislib::sys::NamedPipe::Open
 */
bool vislib::sys::NamedPipe::Open(vislib::StringW name, 
        vislib::sys::NamedPipe::PipeMode mode, unsigned int timeout) {

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

    if (timeout == 0) {
        timeout = INFINITE;
    }

    this->isClient = false;    

    // create the overlapped structure allowing us to produce timeouts
    ::ZeroMemory(&this->overlapped, sizeof(OVERLAPPED));
    this->overlapped.hEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL); // manual reset event
    
    // create the pipe
    this->handle = ::CreateNamedPipeW(pipeName.PeekBuffer(), 
        FILE_FLAG_FIRST_PIPE_INSTANCE | FILE_FLAG_OVERLAPPED | 
        ((mode == PIPE_MODE_READ) ? PIPE_ACCESS_INBOUND : PIPE_ACCESS_OUTBOUND),
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE, 2, PIPE_BUFFER_SIZE, 
        PIPE_BUFFER_SIZE, 0, NULL);

    if (this->handle == INVALID_HANDLE_VALUE) {
        // creation failed
        DWORD lastError = ::GetLastError();

        if (lastError == ERROR_ACCESS_DENIED) {
            // pipe already created! So we should be the client

            this->handle = ::CreateFileW(pipeName.PeekBuffer(), 
                ((mode == PIPE_MODE_WRITE) ? GENERIC_WRITE : GENERIC_READ), 
                0, NULL, OPEN_EXISTING, 0, 0);
    
            this->isClient = true;

            if (this->handle == INVALID_HANDLE_VALUE) {
                throw vislib::sys::SystemException(__FILE__, __LINE__);
            }

        } else {
            // creation really failed!
            throw vislib::sys::SystemException(__FILE__, __LINE__);
        }
    }

    if (!this->isClient) {
        // continue initialising the server
        ::ConnectNamedPipe(this->handle, &this->overlapped);

        switch (::WaitForSingleObject(this->overlapped.hEvent, timeout)) {
            case WAIT_OBJECT_0: {
                // event signaled so connection should be established
                DWORD dummy;
                if (::GetOverlappedResult(this->handle, &this->overlapped, &dummy, FALSE) == 0) {
                    throw vislib::sys::SystemException(__FILE__, __LINE__);
                }
                ::ResetEvent(this->overlapped.hEvent);

            } break;
            default: {
                // unknown error
                vislib::sys::SystemMessage msg(::GetLastError());
                VLTRACE(Trace::LEVEL_VL_INFO, 
                    "NamedPipe Open WaitForSigleObject Error: \"%s\"", 
                    static_cast<const char*>(msg));
            } // NO BREAK;
            case WAIT_TIMEOUT: {
                // timeout
                if (::CancelIo(this->handle) == 0) {
                    vislib::sys::SystemMessage msg(::GetLastError());
                    VLTRACE(Trace::LEVEL_VL_INFO, 
                        "NamedPipe Open Timeout: CancelIo failed \"%s\"", 
                        static_cast<const char*>(msg));
                }
                this->Close();
                return false;
            } break;
        }
    }
    
#ifdef _WIN32
    ASSERT(this->handle != INVALID_HANDLE_VALUE);
#endif /* _WIN32 */

    this->mode = mode;
    return true;

#else /* _WIN32 */

    // since linux and unicode do not really work (as always)           
    return this->Open(W2A(name), mode, timeout);

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

        ::ResetEvent(this->overlapped.hEvent);
        if (!::ReadFile(this->handle, buffer, size, &outRead, &this->overlapped)) {
            DWORD readAll = 0;
            DWORD le = ::GetLastError();
            if (le == ERROR_IO_PENDING) {
                while (le == ERROR_IO_PENDING) {
                    ::WaitForSingleObject(this->overlapped.hEvent, INFINITE);
                    BOOL gor = ::GetOverlappedResult(this->handle, &this->overlapped, &outRead, FALSE);
                    le = (gor != 0) ? NO_ERROR : ::GetLastError();
                    readAll += outRead;

                    if (le == ERROR_IO_INCOMPLETE) {
                        le = ERROR_IO_PENDING;
                    }
                    if (le == ERROR_IO_PENDING) {
                        ::Sleep(1);
                    } else if (le != NO_ERROR) {
                        // problem! (pipe broken)
                        this->cleanupLock.Lock();
                        ::FlushFileBuffers(this->handle);
                        ::CancelIo(this->handle);
                        this->cleanupLock.Unlock();
                        this->Close();
                        throw vislib::sys::SystemException(le, __FILE__, __LINE__);
                    }
                }
            } else {
                // problem! (pipe broken)
                this->cleanupLock.Lock();
                ::FlushFileBuffers(this->handle);
                ::CancelIo(this->handle);
                this->cleanupLock.Unlock();
                this->Close();
                throw vislib::sys::SystemException(le, __FILE__, __LINE__);
            }
            outRead = readAll;

        }

#else /* _WIN32 */

        //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "Read (%d)\n", size);

        outRead = ::read(this->handle, buffer, size);

        //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "Read returned %d\n", outRead);

        if (outRead <= 0) {

            //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "feof or ferror\n");

            this->cleanupLock.Lock();
            ::close(this->handle);
            this->handle = -1;
            this->mode = PIPE_MODE_NONE;
            this->cleanupLock.Unlock();

            //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "feof or ferror ... Done!\n");

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

        ::ResetEvent(this->overlapped.hEvent);
        if (!::WriteFile(this->handle, buffer, size, &outWritten, &this->overlapped)) {
            DWORD writeAll = 0;
            DWORD le = ::GetLastError();
            if (le == ERROR_IO_PENDING) {
                while (le == ERROR_IO_PENDING) {
                    ::WaitForSingleObject(this->overlapped.hEvent, INFINITE);
                    BOOL gor = ::GetOverlappedResult(this->handle, &this->overlapped, &outWritten, FALSE);
                    le = (gor != 0) ? NO_ERROR : ::GetLastError();
                    writeAll += outWritten;

                    if (le == ERROR_IO_INCOMPLETE) {
                        le = ERROR_IO_PENDING;
                    }
                    if (le == ERROR_IO_PENDING) {
                        ::Sleep(1);
                    } else if (le != NO_ERROR) {
                        // problem! (pipe broken)
                        this->cleanupLock.Lock();
                        ::FlushFileBuffers(this->handle);
                        ::CancelIo(this->handle);
                        this->cleanupLock.Unlock();
                        this->Close();
                        throw vislib::sys::SystemException(le, __FILE__, __LINE__);
                    }
                }
            } else {
                // problem! (pipe broken)
                this->cleanupLock.Lock();
                ::FlushFileBuffers(this->handle);
                ::CancelIo(this->handle);
                this->cleanupLock.Unlock();
                this->Close();
                throw vislib::sys::SystemException(le, __FILE__, __LINE__);
            }
            outWritten = writeAll;

        }

#else /* _WIN32 */

        //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "Write (%d)\n", size);

        outWritten = ::write(this->handle, buffer, size);

        //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "Write returned %d\n", outWritten);

        if (outWritten <= 0) {

            //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "feof or ferror\n");

            this->cleanupLock.Lock();
            ::close(this->handle);
            this->handle = -1;
            this->mode = PIPE_MODE_NONE;
            this->cleanupLock.Unlock();

            //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "feof or ferror ... Done!\n");

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
