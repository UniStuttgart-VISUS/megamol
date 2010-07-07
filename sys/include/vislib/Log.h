/*
 * Log.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_LOG_H_INCLUDED
#define VISLIB_LOG_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include <cstdio>
#include <ctime>


namespace vislib {
namespace sys {


    /**
     * This is a utility class for managing a log file.
     */
    class Log {

    public:

        /** type for time stamps */
        typedef time_t TimeStamp;

        /** type for the file name */
#ifdef _WIN32
        typedef vislib::StringW FileNameString;
#else /* _WIN32 */
        typedef vislib::StringA FileNameString;
#endif /* _WIN32 */

        /** 
         * Set this level to log all messages. If you use this constant for 
         * logging itself, the messages will only be logged, if LEVEL_ALL is 
         * also set as current log level.
         */
        static const UINT LEVEL_ALL;

        /**
         * Use this for logging errors. The value of this constant is 1, i. e.
         * messages with LEVEL_ERROR will always be logged, if any logging is
         * enabled.
         */
        static const UINT LEVEL_ERROR;

        /**
         * Use this for informative messages. The value of this constant 
         * is 200. 
         */
        static const UINT LEVEL_INFO;

        /** 
         * Use this for disabling logging. The value is 0. It cannot be used
         * for logging itself, but only for the current log level.
         */
        static const UINT LEVEL_NONE;

        /**
         * Use this for warning messages. The value of this constant 
         * is 100. 
         */
        static const UINT LEVEL_WARN;

        /** The default log object. */
        static Log& DefaultLog;

        /**
         * Nested abstract base class of echo output targets.
         */
        class EchoTarget {
        public:

            /** ctor */
            EchoTarget() { }

            /** dtor */
            virtual ~EchoTarget() { }

            /**
             * Writes a string to the echo output target. Implementations may 
             * assume that message ends with a new line control sequence.
             *
             * @param level The message level.
             * @param message The message ANSI string.
             */
            virtual void Write(UINT level, const char *message) const = 0;

        };

        /**
         * Implementation of EchoTarget writing the messages to a posix system
         * stream.
         */
        class EchoTargetStream : public EchoTarget {
        public:

            /** Object echoing to stdout */
            static const EchoTargetStream StdOut;

            /** Object echoing to stderr */
            static const EchoTargetStream StdErr;

            /** 
             * ctor 
             *
             * @param stream Specifies the stream used. This target object will
             *               not close the stream 
             */
            EchoTargetStream(FILE *stream);

            /**
             * Writes a string to the echo output target. Implementations may 
             * assume that message ends with a new line control sequence.
             *
             * @param level The message level.
             * @param message The message ANSI string.
             */
            virtual void Write(UINT level, const char *message) const;

            /**
             * Answers the associated stream.
             *
             * @return The associated stream.
             */
            inline FILE * Stream(void) const {
                return this->stream;
            }

            /** dtor */
            virtual ~EchoTargetStream();

        private:

            /** the echo stream */
            FILE *stream;
        };

#ifdef _WIN32
        /**
         * Implementation of EchoTarget writing the messages using windows API 
         * OutputDebugString
         */
        class EchoTargetDebugOutput : public EchoTarget {
        public:

            /** ctor */
            EchoTargetDebugOutput() : EchoTarget() { }

            /** dtor */
            virtual ~EchoTargetDebugOutput() { }

            /**
             * Writes a string to the echo output target. Implementations may 
             * assume that message ends with a new line control sequence.
             *
             * @param level The message level.
             * @param message The message ANSI string.
             */
            virtual void Write(UINT level, const char *message) const;

        };
#endif /* _WIN32 */

        /**
         * Implementation of EchoTarget redirecting the messages to another
         * Log object.
         *
         * Not precautions are made to prohibit cycles in the redirection!
         */
        class EchoTargetRedirect : public EchoTarget {
        public:

            /** ctor */
            EchoTargetRedirect(void);

            /** 
             * ctor
             *
             * @param target The targeted log object to receive the echoed
             *               messages. This redirection object does not take
             *               ownership of the memory of the targeted log 
             *               object. Therefore the caller must ensure that the
             *               pointer to the targeted log object remains valid
             *               as long as it is used by this object.
             */
            EchoTargetRedirect(Log *target);

            /** dtor */
            virtual ~EchoTargetRedirect(void);

            /**
             * Answers the targeted log object.
             *
             * @return The targeted log object.
             */
            inline Log* GetTarget(void) const {
                return this->target;
            }

            /**
             * Sets the targeted log object.
             *
             * The targeted log object to receive the echoed messages. This
             * redirection object does not take ownership of the memory of the
             * targeted log object. Therefore the caller must ensure that the
             * pointer to the targeted log object remains valid as long as it
             * is used by this object.
             *
             * @param target The targeted log object to receive the echoed
             *               messages.
             */
            void SetTarget(Log *target);

            /**
             * Writes a string to the echo output target. Implementations may 
             * assume that message ends with a new line control sequence.
             *
             * @param level The message level.
             * @param message The message ANSI string.
             */
            virtual void Write(UINT level, const char *message) const;

        public:

            /** The targeted log object */
            Log *target;

        };

        /** 
         * Ctor. Constructs a new log file without a physical file. 
         *
         * @param level Sets the current log level.
         * @param msgbufsize The number of messages that will be stored in 
         *                   memory if no physical log file is available.
         */
        Log(UINT level = LEVEL_ERROR, unsigned int msgbufsize = 10);

        /** 
         * Ctor. Constructs a new log file with the specified physical file.
         *
         * @param level Sets the current log level.
         * @param filename The name of the physical log file.
         * @param addSuffix If true a automatically generated suffix is added
         *                  to the name of the physical log file, consisting of
         *                  the name of the computer name, the current date and
         *                  time.
         */
        Log(UINT level, const char *filename, bool addSuffix);

        /**
         * Ctor. Constructs a new log file with the specified physical file.
         *
         * @param level Sets the current log level.
         * @param filename The name of the physical log file.
         * @param addSuffix If true a automatically generated suffix is added
         *                  to the name of the physical log file, consisting of
         *                  the name of the computer name, the current date and
         *                  time.
         */
        Log(UINT level, const wchar_t *filename, bool addSuffix);

        /**
         * Copy ctor.
         *
         * @param source The object which will be copied from.
         */
        Log(const Log& source);

        /** Dtor. */
        ~Log(void);

        /**
         * Writes all offline messages to the echo target.
         *
         * @param remove If 'true' the offline messages will be removed after
         *               the operation returns. If 'false' the offline messages
         *               remain in the offline buffer.
         */
        void EchoOfflineMessages(bool remove = false);

        /**
         * Answer the file name of the physical log file. Use this getter to
         * check if the opening of the physical log file in constructors 
         * succeseeded.
         *
         * @return The name of the current physical log file.
         */
        inline const FileNameString& GetLogFileName(void) const {
            return this->filename;
        }

        /**
         * Behaves like GetLogFileName but returns an ANSI string.
         *
         * @return The name of the current physical log file as ANSI string.
         */
#ifdef _WIN32
        inline const vislib::StringA GetLogFileNameA(void) const {
            return W2A(this->filename);
        }
#else /* _WIN32 */
        inline const vislib::StringA& GetLogFileNameA(void) const {
            return this->filename;
        }
#endif /* _WIN32 */

        /**
         * Behaves like GetLogFileName but returns an unicode string.
         *
         * @return The name of the current physical log file as unicode string.
         */
#ifdef _WIN32
        inline const vislib::StringW& GetLogFileNameW(void) const {
            return this->filename;
        }
#else /* _WIN32 */
        inline const vislib::StringW GetLogFileNameW(void) const {
            return A2W(this->filename);
        }
#endif /* _WIN32 */

        /**
         * Specifies the location of the physical log file. Any physical log
         * file currently in use will be closed.
         *
         * @param filename The name of the physical log file. If this parameter
         *                 is NULL, the current physical log file is closed,
         *                 but no new file will be opened.
         * @param addSuffix If true a automatically generated suffix is added
         *                  to the name of the physical log file, consisting of
         *                  the name of the computer name, the current date and
         *                  time.
         *
         * @return true if the log file name had been successfully changes,
         *         false otherwise.
         */
        bool SetLogFileName(const char *filename, bool addSuffix);

        /**
         * Specifies the location of the physical log file. Any physical log
         * file currently in use will be closed.
         *
         * @param filename The name of the physical log file. If this parameter
         *                 is NULL, the current physical log file is closed,
         *                 but no new file will be opened.
         * @param addSuffix If true a automatically generated suffix is added
         *                  to the name of the physical log file, consisting of
         *                  the name of the computer name, the current date and
         *                  time.
         *
         * @return true if the log file name had been successfully changes,
         *         false otherwise.
         */
        bool SetLogFileName(const wchar_t *filename, bool addSuffix);

        /**
         * Answer the number of messages that will be stored in memory if no 
         * physical log file is available.
         *
         * @return The number of messages that will be stored in memory if no 
         *         physical log file is available.
         */
        inline unsigned int GetOfflineMessageBufferSize(void) const {
            return this->msgbufsize;
        }

        /**
         * Sets the number of messages that will be stored in memory if no 
         * physical log file is available.
         *
         * @param msgbufsize The number of messages that will be stored in 
         *                   memory if no physical log file is available.
         */
        void SetOfflineMessageBufferSize(unsigned int msgbufsize);

        /**
         * Answer the state of the autoflush flag.
         *
         * @return The state of the autoflush flag.
         */
        inline bool IsAutoFlushEnabled(void) const {
            return this->autoflush;
        }

        /**
         * Sets or clears the autoflush flag. If the autoflush flag is set
         * a flush of all data to the physical log file is performed after
         * each message. Autoflush is enabled by default.
         *
         * @param enable New value for the autoflush flag.
         */
        inline void SetAutoFlush(bool enable) {
            this->autoflush = enable;
        }

        /** Enables the autoflush flag. */
        inline void EnableAutoFlush(void) {
            this->SetAutoFlush(true);
        }

        /** Disable the autoflush flag. */
        inline void DisableAutoFlush(void) {
            this->SetAutoFlush(false);
        }

        /** Flushes the physical log file. */
        void FlushLog(void);

        /**
         * Answer the current log level. Messages above this level will be
         * ignored.
         *
         * @return The current log level.
         */
        inline UINT GetLevel(void) const {
            return this->level;
        }

        /**
         * Set a new log level. Messages above this level will be ignored.
         *
         * @param level The new log level.
         */
        inline void SetLevel(const UINT level) {
            this->level = level;
        }

        /**
         * Answer the current echo level. Messages above this level will be
         * ignored, while the other messages will be echoed to the echo output
         * stream.
         *
         * @return The current echo level.
         */
        inline UINT GetEchoLevel(void) const {
            return this->echoLevel;
        }

        /**
         * Set a new echo level. Messages above this level will be ignored, 
         * while the other messages will be echoed to the echo output stream.
         *
         * @param level The new echo level.
         */
        inline void SetEchoLevel(const UINT level) {
            this->echoLevel = level;
        }
        /**
         * Answer the current echo output target. This log object does not own
         * this target.
         *
         * @return The current echo level.
         */
        inline const EchoTarget* GetEchoOutTarget(void) const {
            return this->echoOut;
        }

        /**
         * Set a new echo output target. This Log object does not take 
         * ownership of the targets memory. Therefore the caller must ensure
         * that the pointer of the target object is valid as long as it is used
         * by this Log object. Use 'NULL' pointer to deactivate echo output.
         *
         * @param level The new echo output stream.
         */
        inline void SetEchoOutTarget(const EchoTarget* stream) {
            this->echoOut = stream;
        }

        /**
         * Writes a formatted messages with the specified log level to the log
         * file. The format of the message is similar to the printf functions.
         * A new line character is automatically appended if the last 
         * character of fmt is no new line character.
         *
         * @param level The log level of the message.
         * @param fmt The log message.
         */
        void WriteMsg(const UINT level, const char *fmt, ...);

        /**
         * Writes a formatted messages with the specified log level to the log
         * file. The format of the message is similar to the printf functions.
         * A new line character is automatically appended if the last 
         * character of fmt is no new line character.
         *
         * Note: On Linux systems the unicode implementation is at least buggy.
         * It's not recommended to use these unicode methodes under Linux.
         *
         * @param level The log level of the message.
         * @param fmt The log message.
         */
        void WriteMsg(const UINT level, const wchar_t *fmt, ...);

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this.
         */
        Log& operator=(const Log& rhs);

    private:

        /**
         * Helper class for offline messages
         */
        class OfflineMessage {
        public:

            /** ctor */
            OfflineMessage(void);

            /** dtor */
            ~OfflineMessage(void);

            /**
             * Assignment operator
             *
             * @param rhs The right hand side operand.
             *
             * @return Reference to this.
             */
            OfflineMessage& operator=(const OfflineMessage& rhs);

            /** the log level */
            UINT level;

            /** the time stamp */
            TimeStamp time;

            /** the message */
            vislib::StringA message;

        };

        /**
         * Answer the current time.
         *
         * @return A time stamp representing NOW.
         */
        static TimeStamp currentTimeStamp(void);

        /**
         * Write a message prefix consisting of a time stamp and a log level 
         * to the physical log file. The physical log file must be valid.
         *
         * @param level The log level.
         * @param tiemstamp The time stamp.
         */
        void writeMsgPrefix(UINT level, const TimeStamp& timestamp);

        /**
         * Returns the next free offline message object.
         *
         * @return The next free offline message object, or NULL if there is 
         *         none.
         */
        OfflineMessage *nextOfflineMessage(void);

        /**
         * Returns the file name suffix.
         *
         * @return The file name suffix.
         */
        vislib::StringA getFileNameSuffix(void);

        /** the empty log message string const */
        static const char *emptyLogMsg;

        /** the omitted log messages string const */
        static const char *omittedLogMsgs;

        /** the current log level */
        UINT level;

        /** the filename of the log file */
        FileNameString filename;

        /** the file output stream of the physical log file */
        FILE *logfile;

        /** the size of the offline message buffer */
        unsigned int msgbufsize;

        /** the number of omitted offline messages */
        unsigned int omittedMsgs;

        /** array of the offline messages */
        OfflineMessage *offlineMsgs;

        /** flag whether or not to flush the log file after each message. */
        bool autoflush;

        /** the current echo log level */
        UINT echoLevel;

        /** the echo output stream */
        const EchoTarget *echoOut;

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_LOG_H_INCLUDED */
