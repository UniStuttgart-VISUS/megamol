/*
 * NamedPipe.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/NamedPipe.h"
#include "vislib/error.h"
#include "vislib/File.h"
#include "the/argument_exception.h"
#include "the/invalid_operation_exception.h"
#include "the/not_implemented_exception.h"
#include "vislib/Path.h"
#include "the/string.h"
#include "the/text/string_converter.h"
#include "the/system/system_exception.h"
#include "the/trace.h"

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
unsigned int vislib::sys::NamedPipe::Exterminatus::Run(void *userData) {
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
bool vislib::sys::NamedPipe::Open(the::astring name, 
        vislib::sys::NamedPipe::PipeMode mode, unsigned int timeout) {

    // check parameters
    if (!this->checkPipeName(name)) {
        throw the::argument_exception("name", __FILE__, __LINE__);
    }
    if (mode == vislib::sys::NamedPipe::PIPE_MODE_NONE) {
        throw the::argument_exception("mode", __FILE__, __LINE__);
    }

    // close old pipe
    if (this->IsOpen()) {
        this->Close();
    }

    the::astring pipeName = PipeSystemName(name);

#ifdef _WIN32

    if (timeout == 0) {
        timeout = INFINITE;
    }

    this->isClient = false;

    // create the overlapped structure allowing us to produce timeouts
    ::ZeroMemory(&this->overlapped, sizeof(OVERLAPPED));
    this->overlapped.hEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL); // manual reset event
    
    // create the pipe
    this->handle = ::CreateNamedPipeA(pipeName.c_str(), 
        FILE_FLAG_FIRST_PIPE_INSTANCE | FILE_FLAG_OVERLAPPED | 
        ((mode == PIPE_MODE_READ) ? PIPE_ACCESS_INBOUND : PIPE_ACCESS_OUTBOUND),
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE, 2, PIPE_BUFFER_SIZE, 
        PIPE_BUFFER_SIZE, 0, NULL);

    if (this->handle == INVALID_HANDLE_VALUE) {
        // creation failed
        DWORD lastError = ::GetLastError();

        if (lastError == ERROR_ACCESS_DENIED) {
            // pipe already created! So we should be the client

            this->handle = ::CreateFileA(pipeName.c_str(), 
                ((mode == PIPE_MODE_WRITE) ? GENERIC_WRITE : GENERIC_READ), 
                0, NULL, OPEN_EXISTING, 0, 0);
    
            this->isClient = true;

            if (this->handle == INVALID_HANDLE_VALUE) {
                throw the::system::system_exception(__FILE__, __LINE__);
            }

        } else {
            // creation really failed!
            throw the::system::system_exception(__FILE__, __LINE__);
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
                    throw the::system::system_exception(__FILE__, __LINE__);
                }
                ::ResetEvent(this->overlapped.hEvent);

            } break;
            default: {
                // unknown error
                the::system::system_message msg(::GetLastError());
                THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, 
                    "NamedPipe Open WaitForSigleObject Error: \"%s\"", 
                    msg.operator the::astring().c_str());
            } // NO BREAK;
            case WAIT_TIMEOUT: {
                // timeout
                if (::CancelIo(this->handle) == 0) {
                    the::system::system_message msg(::GetLastError());
                    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, 
                        "NamedPipe Open Timeout: CancelIo failed \"%s\"", 
                        msg.operator the::astring().c_str());
                }
                this->Close();
                return false;
            } break;
        }
    }
    THE_ASSERT(this->handle != INVALID_HANDLE_VALUE);
    this->mode = mode;
    return true; // pipe opened

#else /* _WIN32 */
    Exterminatus ext(timeout, pipeName.c_str(), (mode != PIPE_MODE_READ));
    vislib::sys::Thread tot(&ext);

    mode_t oldMask = ::umask(0);
    if (!vislib::sys::File::IsDirectory(this->baseDir)) {
        vislib::sys::Path::MakeDirectory(this->baseDir);
    }

    // Create the FIFO if it does not exist
    if (::mknod(pipeName.c_str(), S_IFIFO | 0666, 0) != 0) {

//        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "mknod failed\n");

        auto lastError = ::GetLastError();
        if (lastError != EEXIST) {
            ::umask(oldMask);
            throw the::system::system_exception(lastError, __FILE__, __LINE__);
        }
    }
    ::umask(oldMask);

//    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "vislib::sys::NamedPipe::Open\n");

    if (timeout > 0) {
        tot.Start();
    }

    this->handle = ::open(pipeName.c_str(), O_SYNC | 
        ((mode == PIPE_MODE_READ) ? O_RDONLY : O_WRONLY));
    ext.MarkConnected();

    if (timeout > 0) {
        tot.Join();
    }

    if (this->handle < 0) {
//        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "open failed.\n");
        throw the::system::system_exception(__FILE__, __LINE__);
    } else if (ext.IsTerminated()) {
//        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "open timed out.\n");
        ::close(this->handle);
        this->handle = -1;
    }

    // Tricky.
    // This works since 'open' blocks until both ends of the pipe are opend.
    ::unlink(pipeName.c_str());

    if (this->handle >= 0) {
        this->mode = mode;
    }
    return (this->handle >= 0); // pipe opened

#endif /* _WIN32 */ 
}


/*
 * vislib::sys::NamedPipe::Open
 */
bool vislib::sys::NamedPipe::Open(the::wstring name, 
        vislib::sys::NamedPipe::PipeMode mode, unsigned int timeout) {

#ifdef _WIN32

    // check parameters
    if (!this->checkPipeName(name)) {
        throw the::argument_exception("name", __FILE__, __LINE__);
    }
    if (mode == vislib::sys::NamedPipe::PIPE_MODE_NONE) {
        throw the::argument_exception("mode", __FILE__, __LINE__);
    }

    // close old pipe
    if (this->IsOpen()) {
        this->Close();
    }

    the::wstring pipeName = PipeSystemName(name);

    if (timeout == 0) {
        timeout = INFINITE;
    }

    this->isClient = false;    

    // create the overlapped structure allowing us to produce timeouts
    ::ZeroMemory(&this->overlapped, sizeof(OVERLAPPED));
    this->overlapped.hEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL); // manual reset event
    
    // create the pipe
    this->handle = ::CreateNamedPipeW(pipeName.c_str(), 
        FILE_FLAG_FIRST_PIPE_INSTANCE | FILE_FLAG_OVERLAPPED | 
        ((mode == PIPE_MODE_READ) ? PIPE_ACCESS_INBOUND : PIPE_ACCESS_OUTBOUND),
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE, 2, PIPE_BUFFER_SIZE, 
        PIPE_BUFFER_SIZE, 0, NULL);

    if (this->handle == INVALID_HANDLE_VALUE) {
        // creation failed
        DWORD lastError = ::GetLastError();

        if (lastError == ERROR_ACCESS_DENIED) {
            // pipe already created! So we should be the client

            this->handle = ::CreateFileW(pipeName.c_str(), 
                ((mode == PIPE_MODE_WRITE) ? GENERIC_WRITE : GENERIC_READ), 
                0, NULL, OPEN_EXISTING, 0, 0);
    
            this->isClient = true;

            if (this->handle == INVALID_HANDLE_VALUE) {
                throw the::system::system_exception(__FILE__, __LINE__);
            }

        } else {
            // creation really failed!
            throw the::system::system_exception(__FILE__, __LINE__);
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
                    throw the::system::system_exception(__FILE__, __LINE__);
                }
                ::ResetEvent(this->overlapped.hEvent);

            } break;
            default: {
                // unknown error
                the::system::system_message msg(::GetLastError());
                THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, 
                    "NamedPipe Open WaitForSigleObject Error: \"%s\"", 
                    msg.operator the::astring().c_str());
            } // NO BREAK;
            case WAIT_TIMEOUT: {
                // timeout
                if (::CancelIo(this->handle) == 0) {
                    the::system::system_message msg(::GetLastError());
                    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, 
                        "NamedPipe Open Timeout: CancelIo failed \"%s\"", 
                        msg.operator the::astring().c_str());
                }
                this->Close();
                return false;
            } break;
        }
    }
    
#ifdef _WIN32
    THE_ASSERT(this->handle != INVALID_HANDLE_VALUE);
#endif /* _WIN32 */

    this->mode = mode;
    return true;

#else /* _WIN32 */

    // since linux and unicode do not really work (as always)           
    return this->Open(THE_W2A(name), mode, timeout);

#endif /* _WIN32 */ 
}


/*
 * vislib::sys::NamedPipe::Read
 */
void vislib::sys::NamedPipe::Read(void *buffer, unsigned int size) {

    if (this->mode != PIPE_MODE_READ) {
        throw the::invalid_operation_exception("Pipe not in read mode", __FILE__, __LINE__);
    }
    if (size == 0) return;
    if (buffer == NULL) {
        throw the::argument_exception("buffer", __FILE__, __LINE__);
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
                        throw the::system::system_exception(le, __FILE__, __LINE__);
                    }
                }
            } else {
                // problem! (pipe broken)
                this->cleanupLock.Lock();
                ::FlushFileBuffers(this->handle);
                ::CancelIo(this->handle);
                this->cleanupLock.Unlock();
                this->Close();
                throw the::system::system_exception(le, __FILE__, __LINE__);
            }
            outRead = readAll;

        }

#else /* _WIN32 */

        //THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Read (%d)\n", size);

        outRead = ::read(this->handle, buffer, size);

        //THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Read returned %d\n", outRead);

        if (outRead <= 0) {

            //THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "feof or ferror\n");

            this->cleanupLock.Lock();
            ::close(this->handle);
            this->handle = -1;
            this->mode = PIPE_MODE_NONE;
            this->cleanupLock.Unlock();

            //THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "feof or ferror ... Done!\n");

            throw the::system::system_exception(__FILE__, __LINE__);
        }

#endif /* _WIN32 */ 

        buffer = static_cast<void*>(static_cast<char*>(buffer) + outRead);
        size -= outRead;
    }
}


/*
 * vislib::sys::NamedPipe::PipeSystemName
 */
the::astring vislib::sys::NamedPipe::PipeSystemName(const the::astring &name) {
    if (!checkPipeName(name)) {
        throw the::argument_exception("name", __FILE__, __LINE__);
    }

#ifdef _WIN32
    the::astring pipeName = "\\\\.\\pipe\\";
    pipeName += name;
#else /* _WIN32 */
    the::astring pipeName = baseDir;
    pipeName += "/";
    pipeName += name;
#endif /* _WIN32 */
    return pipeName;
}


/*
 * vislib::sys::NamedPipe::PipeSystemName
 */
the::wstring vislib::sys::NamedPipe::PipeSystemName(const the::wstring &name) {
    if (!checkPipeName(name)) {
        throw the::argument_exception("name", __FILE__, __LINE__);
    }

#ifdef _WIN32
    the::wstring pipeName = L"\\\\.\\pipe\\";
    pipeName += name;
#else /* _WIN32 */
    the::wstring pipeName = THE_A2W(baseDir);
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
        throw the::invalid_operation_exception("Pipe not in write mode", __FILE__, __LINE__);
    }
    if (size == 0) return;
    if (buffer == NULL) {
        throw the::argument_exception("buffer", __FILE__, __LINE__);
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
                        throw the::system::system_exception(le, __FILE__, __LINE__);
                    }
                }
            } else {
                // problem! (pipe broken)
                this->cleanupLock.Lock();
                ::FlushFileBuffers(this->handle);
                ::CancelIo(this->handle);
                this->cleanupLock.Unlock();
                this->Close();
                throw the::system::system_exception(le, __FILE__, __LINE__);
            }
            outWritten = writeAll;

        }

#else /* _WIN32 */

        //THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Write (%d)\n", size);

        outWritten = ::write(this->handle, buffer, size);

        //THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Write returned %d\n", outWritten);

        if (outWritten <= 0) {

            //THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "feof or ferror\n");

            this->cleanupLock.Lock();
            ::close(this->handle);
            this->handle = -1;
            this->mode = PIPE_MODE_NONE;
            this->cleanupLock.Unlock();

            //THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "feof or ferror ... Done!\n");

            throw the::system::system_exception(__FILE__, __LINE__);

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
        const T &name) {
    if (std::find(name.begin(), name.end(), static_cast<typename T::value_type>(vislib::sys::Path::SEPARATOR_A)) != name.end()) {
        return false;
    }
    if (std::find(name.begin(), name.end(), static_cast<typename T::value_type>('<')) != name.end()) {
        return false;
    }
    if (std::find(name.begin(), name.end(), static_cast<typename T::value_type>('>')) != name.end()) {
        return false;
    }
    if (std::find(name.begin(), name.end(), static_cast<typename T::value_type>('|')) != name.end()) {
        return false;
    }
    if (std::find(name.begin(), name.end(), static_cast<typename T::value_type>(':')) != name.end()) {
        return false;
    }
    if (std::find(name.begin(), name.end(), static_cast<typename T::value_type>('*')) != name.end()) {
        return false;
    }
    if (std::find(name.begin(), name.end(), static_cast<typename T::value_type>('?')) != name.end()) {
        return false;
    }

    return true;
}
