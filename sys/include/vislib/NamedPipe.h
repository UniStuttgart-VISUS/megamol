/*
 * NamedPipe.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_NAMEDPIPE_H_INCLUDED
#define VISLIB_NAMEDPIPE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/CriticalSection.h"
#include "vislib/String.h"
#ifndef _WIN32
#include "vislib/Runnable.h"
#endif /* !_WIN32 */


namespace vislib {
namespace sys {

    /**
     * This class implements a named pipe (1 to 1 fifo ipc mechanism).
     *
     * WARNING: Do not try to use both ends of one pipe in a single thread!
     * This could result in a deadlock!
     *
     * Be aware that 'Open', 'Read', and 'Write' are blocking calles and 
     * require a corresponding call to 'Open', 'Write', or 'Read' on the other
     * end of the pipe before the methods return.
     *
     * Keep in mind that it is almost impossible to close a pipe gracefully. In
     * most cases 'Read' or 'Write' will throw a system exception as soon as 
     * the pipe breaks because one of it's ends is closed.
     *
     * Note: The linux implementation registers a signal handler for the signal
     * SIGPIPE. Do not register another handler or the behaviour of this class
     * is undefined.
     *
     * Note: The linux implementation uses a "hand-shake" message to determine 
     * if the connection was successfully created. These NamedPipe objects are
     * therefore NOT compatible with any other named pipe implementations.
     */
    class NamedPipe {
    public:
        
        /** possible values for PipeMode */
        enum PipeMode {
            PIPE_MODE_NONE,
            PIPE_MODE_WRITE,
            PIPE_MODE_READ
        };

        /** Ctor. */
        NamedPipe(void);

        /** Dtor. */
        ~NamedPipe(void);

        /**
         * Closes a named pipe. If this pipe is not opened, the method returns
         * immediately. 
         */
        void Close(void);

        /**
         * Answers whether the named pipe is open.
         *
         * @return True if the named pipe is open, false otherwise.
         */
        inline bool IsOpen(void) const {
            return this->mode != PIPE_MODE_NONE;
        }
        
        /**
         * Opens a named pipe. If another pipe is already opend, that pipe is 
         * closed. Each pipe may only be opend once with each available 
         * OpenMode (Only one reader and one writer). The creating mode 
         * specifies if the method should fail if the pipe is not yet created.
         * The pipe name may not contain any characters which whould be invalid
         * for a file name.
         *
         * Under windows operating systems the name specifies the pipe
         * '\\.\pipe\<name>' where '<name>' is replace with the given name.
         *
         * Under linux systems the pipe is placed in the file system at
         * '/tmp/vislibpipes/<name>' where '<name>' is replace with the given 
         * name.
         *
         * This method will not return until both ends of the pipe are opened.
         * It's therefore not possible to open both ends from the same thread.
         *
         * @param name The name of the named pipe.
         *             TODO: Document further
         * @param mode The PipeMode to be used. Must not be 'PIPE_MODE_NONE'.
         * @param timeout Specifies the connection timeout in milliseconds. If 
         *                the pipe cannot be opened within this time out the
         *                functions returns false. A value of zero disables the
         *                timeout.
         *
         * @return 'true' if the pipe has been opened, 'false' otherwise.
         *
         * @throws IllegalParamException if name contains any invalid 
         *                               characters.
         * @throws IllegalParamException if openMode == 'PIPE_MODE_NONE'
         * @throws SystemException if the pipe could not be opened.
         */
        bool Open(StringA name, PipeMode mode, unsigned int timeout = 0);

        /**
         * Opens a named pipe. If another pipe is already opend, that pipe is 
         * closed. Each pipe may only be opend once with each available 
         * OpenMode (Only one reader and one writer). The creating mode 
         * specifies if the method should fail if the pipe is not yet created.
         * The pipe name may not contain any characters which whould be invalid
         * for a file name.
         *
         * Under windows operating systems the name specifies the pipe
         * '\\.\pipe\<name>' where '<name>' is replace with the given name.
         *
         * Under linux systems the pipe is placed in the file system at
         * '/tmp/vislibpipes/<name>' where '<name>' is replace with the given 
         * name.
         *
         * This method will not return until both ends of the pipe are opened.
         * It's therefore not possible to open both ends from the same thread.
         *
         * @param name The name of the named pipe.
         *             TODO: Document further
         * @param mode The PipeMode to be used. Must not be 'PIPE_MODE_NONE'.
         * @param timeout Specifies the connection timeout in milliseconds. If 
         *                the pipe cannot be opened within this time out the
         *                functions returns false. A value of zero disables the
         *                timeout.
         *
         * @return 'true' if the pipe has been opened, 'false' otherwise.
         *
         * @throws IllegalParamException if name contains any invalid 
         *                               characters.
         * @throws IllegalParamException if openMode == 'PIPE_MODE_NONE'
         * @throws SystemException if the pipe could not be opened.
         */
        bool Open(StringW name, PipeMode mode, unsigned int timeout = 0);

        /**
         * Answers the OpenMode of the pipe or 'PIPE_MODE_NONE' if the pipe is
         * closed.
         *
         * @return The Mode of the pipe.
         */
        inline PipeMode Mode(void) const {
            return this->mode;
        }

        /**
         * Reads 'size' bytes from the pipe into 'buffer'. The method does not
         * return until 'size' bytes could be read. Communication errors (e. g.
         * when the pipe breaks) result in exceptions.
         *
         * @param buffer Pointer to the buffer to read.
         * @param size The number of bytes to read.
         *
         * @throws IllegalStateException if the Mode of the pipe is not
         *                               'PIPE_MODE_READ'.
         * @throws IllegalParamException if buffer is NULL.
         * @throws SystemException if the pipe could not be read, e. g. if the
         *                         pipe is broke.
         */
        void Read(void *buffer, unsigned int size);

        /**
         * Answer the full system name of a pipe. The pipe name 'name' is 
         * expanded by an os dependent path. On Windows the return value will
         * be '\\.\pipe\name' and on Linux the return value will be
         * '/tmp/vislibpipes/name'.
         *
         * @param name The name of the named pipe.
         *
         * @return The full system name of the named pipe.
         *
         * @throws IllegalParamException if name is not a valid pipe name.
         */
        static vislib::StringA PipeSystemName(const vislib::StringA &name);

        /**
         * Answer the full system name of a pipe. The pipe name 'name' is 
         * expanded by an os dependent path. On Windows the return value will
         * be '\\.\pipe\name' and on Linux the return value will be
         * '/tmp/vislibpipes/name'.
         *
         * @param name The name of the named pipe.
         *
         * @return The full system name of the named pipe.
         *
         * @throws IllegalParamException if name is not a valid pipe name.
         */
        static vislib::StringW PipeSystemName(const vislib::StringW &name);

        /**
         * Writes 'size' bytes from 'buffer' into the pipe. The method does not
         * return until all bytes are written. Communication errors (e. g. when
         * the pipe breaks) result in exceptions.
         *
         * @param buffer Pointer to the buffer to write.
         * @param size The number of bytes to write.
         *
         * @throws IllegalStateException If the Mode of the pipe is not
         *                               'PIPE_MODE_WRITE'.
         * @throws IllegalParamException if buffer is NULL.
         * @throws SystemException if the data could not be written to the 
         *                         pipe, e. g. if the pipe is broke.
         */
        void Write(void *buffer, unsigned int size);

    private:

        /**
         * Checks wether the given name is a valid pipe name. Currently the 
         * following characters are forbidden:
         *  - the path separator
         *  - '>', '<', '|', ':', '?', and '*'
         *
         * @param name The name to check.
         *
         * @return true if the name is valid, false otherwise.
         */
        template<class T> static bool checkPipeName(const String<T> &name);

#ifdef _WIN32
        /** The handle of the pipe */
        HANDLE handle;

        /** flag indicating if the pipe is the client */
        bool isClient;

        /** The overlapped structure of timeout control */
        OVERLAPPED overlapped;

#else /* _WIN32 */

        /**
         * Nested runnable class creating the timeout terminating mechanism.
         */
        class Exterminatus : public Runnable {
        public:

            /** 
             * ctor 
             *
             * @param timeout The timeout in milliseconds
             * @param pipename The name of the pipe to terminate after timeout
             * @param readend 'true' if the termination is performed using the
             *                reading end, 'false' when using the writing end.
             */
            Exterminatus(unsigned int timeout, const char *pipename, bool readend);

            /** dtor */
            ~Exterminatus(void);

            /**
             * Perform the work of a thread.
             *
             * @param userData A pointer to user data that are passed to the thread,
             *                 if it started.
             *
             * @return The application dependent return code of the thread. This 
             *         must not be STILL_ACTIVE (259).
             */
            virtual DWORD Run(void *userData);

            /** Marks the pipe as connected. */
            inline void MarkConnected(void) {
                this->connected = true;
            }

            /**
             * Answer if the pipe has to be terminated due to timeout.
             *
             * @return 'true' if the pipe has to be terminated, 
             *         'false' otherwise.
             */
            inline bool IsTerminated(void) const {
                return this->terminated;
            }

        private:

            /** flag indicating that the pipe is connected. */
            bool connected;

            /** flag indicating that the pipe has been terminated */
            bool terminated;

            /** the timeout value */
            unsigned int timeout;

            /** the name of the pipe */
            const char *pipename;

            /** flag whether to open the reading end of the pipe */
            bool readend;

        };

        /** The handle of the pipe */
        int handle;

        /** The base directory for the named pipe file system nodes */
        static const char baseDir[];

#endif /* _WIN32 */

        /** The mode of the pipe */
        PipeMode mode;

        /** The critical section of the pipe used to synchronise the cleanup */
        CriticalSection cleanupLock;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_NAMEDPIPE_H_INCLUDED */
