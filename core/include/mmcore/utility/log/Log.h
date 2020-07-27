/*
 * Log.h
 *
 * Copyright (C) 2006 - 2010 by Universitaet Stuttgart (VIS).
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/api/MegaMolCore.std.h"

#include <cstdio>
#include <ctime>
#include <string>
#include <iostream>
#include <thread>

#define SPDLOG_EOL ""

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/msvc_sink.h"
#include "spdlog/sinks/ostream_sink.h"
#include "spdlog/spdlog.h"

namespace megamol {
namespace core {
namespace utility {
namespace log {


    /**
     * This is a utility class for managing a log file.
     */
    class MEGAMOLCORE_API Log {
    public:

        /** Default log message pattern for spdlog */
        static const char std_pattern[3];

        /** type for time stamps */
        using TimeStamp = time_t;

        /** type for the id of the source object */
        using SourceID = size_t;

        /** unsigned int alias */
        using UINT = unsigned int;

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
         * Answer the current source id
         *
         * @return A source id representing THIS
         */
        static SourceID CurrentSourceID(void);

        /**
         * Answer the current time.
         *
         * @return A time stamp representing NOW.
         */
        static TimeStamp CurrentTimeStamp(void);

        /**
         * Abstract base class for log targets
         */
        class MEGAMOLCORE_API Target {
        public:

            /** Dtor */
            virtual ~Target(void);

            /** Flushes any buffer */
            virtual void Flush(void);

            /**
             * Answer the log level of this target
             *
             * @return The log level of this target
             */
            inline UINT Level(void) const {
                return this->level;
            }

            /**
             * Writes a message to the log target
             *
             * @param level The level of the message
             * @param time The time stamp of the message
             * @param sid The object id of the source of the message
             * @param msg The message text itself
             */
            virtual void Msg(UINT level, TimeStamp time, SourceID sid,
                const char *msg) = 0;

            /**
             * Writes a message to the log target
             *
             * @param level The level of the message
             * @param time The time stamp of the message
             * @param sid The object id of the source of the message
             * @param msg The message text itself
             */
            virtual void Msg(UINT level, TimeStamp time, SourceID sid, std::string const& msg) = 0;

            /**
             * Sets the log level for this target
             *
             * @param level The new log level
             */
            inline void SetLevel(UINT level) {
                this->level = level;
            }

        protected:

            /**
             * Ctor
             *
             * @param level The log level for this target
             */
            Target(UINT level = Log::LEVEL_ERROR);

        private:

            /** The log level for this target */
            UINT level;

        };

#ifdef _WIN32
        /**
         * Target class echoing the log messages into a stream
         */
        class DebugOutputTarget : public Target {
        public:

            /**
             * Ctor
             *
             * @param level The log level used for this target
             */
            DebugOutputTarget(UINT level = Log::LEVEL_ERROR);

            /** Dtor */
            virtual ~DebugOutputTarget(void);

            /**
             * Writes a message to the log target
             *
             * @param level The level of the message
             * @param time The time stamp of the message
             * @param sid The object id of the source of the message
             * @param msg The message text itself
             */
            void Msg(UINT level, TimeStamp time, SourceID sid,
                const char *msg) override;

            /**
             * Writes a message to the log target
             *
             * @param level The level of the message
             * @param time The time stamp of the message
             * @param sid The object id of the source of the message
             * @param msg The message text itself
             */
            void Msg(UINT level, TimeStamp time, SourceID sid, std::string const& msg) override;

        private:
            std::shared_ptr<spdlog::sinks::msvc_sink_mt> sink;

            std::shared_ptr<spdlog::logger> logger;
        };
#endif /* _WIN32 */

        /**
         * Target class safing message to a ASCII text file
         */
        class FileTarget : public Target {
        public:

            /**
             * Opens a physical log file
             *
             * @param path The path to the physical log file
             * @param level The log level used for this target
             */
            FileTarget(std::string const& path, UINT level = Log::LEVEL_ERROR);

            /** Dtor */
            virtual ~FileTarget(void);

            /** Flushes any buffer */
            void Flush(void) override;

            /**
             * Answer the path to the physical log file
             *
             * @return The path to the physical log file
             */
            inline const std::string& Filename(void) const {
                return this->filename;
            }

            /**
             * Writes a message to the log target
             *
             * @param level The level of the message
             * @param time The time stamp of the message
             * @param sid The object id of the source of the message
             * @param msg The message text itself
             */
            void Msg(UINT level, TimeStamp time, SourceID sid,
                const char *msg) override;

            /**
             * Writes a message to the log target
             *
             * @param level The level of the message
             * @param time The time stamp of the message
             * @param sid The object id of the source of the message
             * @param msg The message text itself
             */
            void Msg(UINT level, TimeStamp time, SourceID sid, std::string const& msg) override;

        private:
            
            /** The file name of the log file used */
            std::string filename;

            std::shared_ptr<spdlog::sinks::basic_file_sink_mt> sink;

            std::shared_ptr<spdlog::logger> logger;

        };

        /**
         * Target class storing message as long as the file is offline
         */
        class OfflineTarget : public Target {
        public:

            /**
             * Ctor
             *
             * @param bufferSize The number of message to be stored in the
             *                   offline buffer.
             * @param level The log level used for this target
             */
            OfflineTarget(unsigned int bufferSize = 20,
                UINT level = Log::LEVEL_ERROR);

            /** Dtor */
            virtual ~OfflineTarget(void);

            /**
             * Answer the size of the offline message buffer
             *
             * @return The number of offline message that will be stored in
             *         the offline buffer
             */
            inline unsigned int BufferSize(void) const {
                return this->bufSize;
            }

            /**
             * Writes a message to the log target
             *
             * @param level The level of the message
             * @param time The time stamp of the message
             * @param sid The object id of the source of the message
             * @param msg The message text itself
             */
            void Msg(UINT level, TimeStamp time, SourceID sid,
                const char *msg) override;

            /**
             * Writes a message to the log target
             *
             * @param level The level of the message
             * @param time The time stamp of the message
             * @param sid The object id of the source of the message
             * @param msg The message text itself
             */
            void Msg(UINT level, TimeStamp time, SourceID sid, std::string const& msg) override;

            /**
             * Answer the number of omitted messages
             *
             * @return The number of omitted messages
             */
            inline unsigned int OmittedMessagesCount(void) const {
                return omittedCnt;
            }

            /**
             * Echoes all stored offline messages to 'target' and optionally
             * deletes the message buffer.
             *
             * @param target The target object to receive the messages
             * @param remove If true the offline message buffer will be freed
             */
            void Reecho(Target &target, bool remove = true);

            /**
             * Sets the size of the buffer for offline messages
             *
             * @param bufferSize The number of message to be stored in the
             *                   offline buffer.
             */
            void SetBufferSize(unsigned int bufferSize);

        private:

            /**
             * Utility struct for offline messages
            */
            typedef struct _offlinemessage_t {

                /** The level of the message */
                UINT level;

                /** The message time stamp */
                TimeStamp time;

                /** The message source ID */
                SourceID sid;

                /** The message text */
                std::string msg;

            } OfflineMessage;

            /** The number of offline messages to be stored */
            unsigned int bufSize;

            /** The number of stored messages */
            unsigned int msgCnt;

            /** buffer of offline messages */
            OfflineMessage *msgs;

            /** The number of omitted messages */
            unsigned int omittedCnt;

        };

        /**
         * Target class echoing the log messages into a stream
         */
        class MEGAMOLCORE_API StreamTarget : public Target {
        public:

            /** Stream target to stdout */
            static const std::shared_ptr<Target> StdOut;

            /** Stream target to stderr */
            static const std::shared_ptr<Target> StdErr;

            /**
             * Ctor
             *
             * @param stream The stream to write the log messages to
             * @param level The log level used for this target
             */
            StreamTarget(std::ostream& stream, UINT level = Log::LEVEL_ERROR);

            /** Dtor */
            virtual ~StreamTarget(void);

            /** Flushes any buffer */
            void Flush(void) override;

            /**
             * Writes a message to the log target
             *
             * @param level The level of the message
             * @param time The time stamp of the message
             * @param sid The object id of the source of the message
             * @param msg The message text itself
             */
            void Msg(UINT level, TimeStamp time, SourceID sid,
                const char *msg) override;

            /**
             * Writes a message to the log target
             *
             * @param level The level of the message
             * @param time The time stamp of the message
             * @param sid The object id of the source of the message
             * @param msg The message text itself
             */
            void Msg(UINT level, TimeStamp time, SourceID sid, std::string const& msg) override;

        private:

             std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink;

             std::shared_ptr<spdlog::logger> logger;

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
         * Copy ctor.
         *
         * @param source The object which will be copied from.
         */
        Log(const Log& source);

        /** Dtor. */
        ~Log(void);

        /**
         * Access the echo log target
         *
         * @return The echo log target
         */
        inline const std::shared_ptr<Target> AccessEchoTarget(void) const {
            return this->echoTarget;
        }

        /**
         * Access the main log target
         *
         * @return The main log target
         */
        inline const std::shared_ptr<Target> AccessMainTarget(void) const {
            return this->mainTarget;
        }

        /** Disable the autoflush flag. */
        inline void DisableAutoFlush(void) {
            this->SetAutoFlush(false);
        }

        /**
         * Writes all offline messages to the echo target.
         *
         * @param remove If 'true' the offline messages will be removed after
         *               the operation returns. If 'false' the offline messages
         *               remain in the offline buffer.
         */
        void EchoOfflineMessages(bool remove = false);

        /** Enables the autoflush flag. */
        inline void EnableAutoFlush(void) {
            this->SetAutoFlush(true);
        }

        /** Flushes the physical log file. */
        void FlushLog(void);

        /**
         * Answer the current echo level. Messages above this level will be
         * ignored, while the other messages will be echoed to the echo output
         * stream.
         *
         * @return The current echo level.
         */
        UINT GetEchoLevel(void) const;

        /**
         * Answer the current log level. Messages above this level will be
         * ignored.
         *
         * @return The current log level.
         */
        UINT GetLevel(void) const;

        /**
         * Answer the file name of the log file as ANSI string.
         *
         * @return The name of the current physical log file as ANSI string.
         */
        std::string GetLogFileNameA(void) const;

        /**
         * Answer the number of messages that will be stored in memory if no 
         * physical log file is available.
         *
         * @return The number of messages that will be stored in memory if no 
         *         physical log file is available.
         */
        unsigned int GetOfflineMessageBufferSize(void) const;

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

        /**
         * Set a new echo level. Messages above this level will be ignored, 
         * while the other messages will be echoed to the echo output stream.
         *
         * @param level The new echo level.
         */
        void SetEchoLevel(UINT level);

        /**
         * Sets the new echo log target
         *
         * @param The new echo log target
         */
        void SetEchoTarget(std::shared_ptr<Target> target);

        /**
         * Set a new log level. Messages above this level will be ignored.
         *
         * @param level The new log level.
         */
        void SetLevel(UINT level);

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
         * Sets the new main log target
         *
         * @param The new main log target
         */
        void SetMainTarget(std::shared_ptr<Target> target);

        /**
         * Sets the number of messages that will be stored in memory if no 
         * physical log file is available.
         *
         * @param msgbufsize The number of messages that will be stored in 
         *                   memory if no physical log file is available.
         */
        void SetOfflineMessageBufferSize(unsigned int msgbufsize);

        /**
         * Connects the internal memory for log targets of this this log with
         * the memory of the 'master' log. Changes to the targets themself are
         * not thread-safe. Log messages as input to the targets may be
         * thead-safe depending on the employed targets.
         *
         * @param master The master log providing the memory for stroing the
         *               log targets.
         */
        void ShareTargetStorage(const Log& master);

        /**
         * Writes a formatted error message to the log. The level will be
         * 'LEVEL_ERROR'.
         *
         * @param fmt The log message
         */
        void WriteError(const char *fmt, ...);

        /**
         * Writes a formatted error message to the log. The level will be
         * 'LEVEL_ERROR + lvlOff'. Not that a high level offset value might
         * downgrade the message to warning or even info level.
         *
         * @param fmt The log message
         * @param lvlOff The log level offset
         */
        void WriteError(int lvlOff, const char *fmt, ...);

        /**
         * Writes a formatted error message to the log. The level will be
         * 'LEVEL_INFO'.
         *
         * @param fmt The log message
         */
        void WriteInfo(const char *fmt, ...);

        /**
         * Writes a formatted error message to the log. The level will be
         * 'LEVEL_INFO + lvlOff'.
         *
         * @param fmt The log message
         * @param lvlOff The log level offset
         */
        void WriteInfo(int lvlOff, const char *fmt, ...);

        /**
         * Writes a pre-formatted message with specified log level, time stamp
         * and source id to the log.
         *
         * @param level The level of the message
         * @param time The time stamp of the message
         * @param sid The object id of the source of the message
         * @param msg The message text itself
         */
        void WriteMessage(UINT level, TimeStamp time, SourceID sid,
            const std::string& msg);

        /**
         * Writes a pre-formatted message with specified log level, time stamp
         * and source id to the log.
         *
         * @param level The level of the message
         * @param time The time stamp of the message
         * @param sid The object id of the source of the message
         * @param msg The message text itself
         */
        void WriteMessageVaA(UINT level, TimeStamp time, SourceID sid,
            const char *fmt, va_list argptr);

        /**
         * Writes a formatted messages with the specified log level to the log
         * file. The format of the message is similar to the printf functions.
         * A new line character is automatically appended if the last 
         * character of fmt is no new line character.
         *
         * @param level The log level of the message.
         * @param fmt The log message.
         */
        void WriteMsg(UINT level, const char *fmt, ...);

        /**
         * Writes a formatted error message to the log. The level will be
         * 'LEVEL_WARN'.
         *
         * @param fmt The log message
         */
        void WriteWarn(const char *fmt, ...);

        /**
         * Writes a formatted error message to the log. The level will be
         * 'LEVEL_WARN + lvlOff'. Not that a high level offset value might
         * downgrade the message to info level.
         *
         * @param fmt The log message
         * @param lvlOff The log level offset
         */
        void WriteWarn(int lvlOff, const char *fmt, ...);

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
         * Answer a file name suffix for log files
         *
         * @return A file name suffix for log files
         */
        std::string getFileNameSuffix(void);

        /** The main log target */
        std::shared_ptr<Target> mainTarget;

        /** The log echo target */
        std::shared_ptr<Target> echoTarget;

        /** Flag whether or not to flush any targets after each message */
        bool autoflush;

    };
    
} // namespace log
} // namespace utility
} // namespace core
} // namespace megamol
