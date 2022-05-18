/*
 * OfflineTarget.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/utility/log/Log.h"

namespace megamol::core::utility::log {

/**
 * Target class storing message as long as the file is offline
 */
class OfflineTarget : public Log::Target {
public:
    /**
     * Ctor
     *
     * @param bufferSize The number of message to be stored in the
     *                   offline buffer.
     * @param level The log level used for this target
     */
    OfflineTarget(unsigned int bufferSize = 20, Log::UINT level = Log::LEVEL_ERROR);

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
    void Msg(Log::UINT level, Log::TimeStamp time, Log::SourceID sid, const char* msg) override;

    /**
     * Writes a message to the log target
     *
     * @param level The level of the message
     * @param time The time stamp of the message
     * @param sid The object id of the source of the message
     * @param msg The message text itself
     */
    void Msg(Log::UINT level, Log::TimeStamp time, Log::SourceID sid, std::string const& msg) override;

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
    void Reecho(Target& target, bool remove = true);

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
        Log::UINT level;

        /** The message time stamp */
        Log::TimeStamp time;

        /** The message source ID */
        Log::SourceID sid;

        /** The message text */
        std::string msg;

    } OfflineMessage;

    /** The number of offline messages to be stored */
    unsigned int bufSize;

    /** The number of stored messages */
    unsigned int msgCnt;

    /** buffer of offline messages */
    OfflineMessage* msgs;

    /** The number of omitted messages */
    unsigned int omittedCnt;
};

} // namespace megamol::core::utility::log
