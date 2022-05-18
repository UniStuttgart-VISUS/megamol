/*
 * OfflineTarget.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/log/OfflineTarget.h"

#include <algorithm>
#include <sstream>

#include "vislib/memutils.h"

/*
 * megamol::core::utility::log::OfflineTarget::OfflineTarget
 */
megamol::core::utility::log::OfflineTarget::OfflineTarget(unsigned int bufferSize, Log::UINT level)
        : Target(level)
        , bufSize(bufferSize)
        , msgCnt(0)
        , msgs(new OfflineMessage[bufferSize])
        , omittedCnt(0) {
    // intentionally empty
}


/*
 * megamol::core::utility::log::OfflineTarget::~OfflineTarget
 */
megamol::core::utility::log::OfflineTarget::~OfflineTarget(void) {
    ARY_SAFE_DELETE(this->msgs);
    this->bufSize = 0;
    this->msgCnt = 0;
}


/*
 * megamol::core::utility::log::OfflineTarget::Msg
 */
void megamol::core::utility::log::OfflineTarget::Msg(Log::UINT level, megamol::core::utility::log::Log::TimeStamp time,
    megamol::core::utility::log::Log::SourceID sid, const char* msg) {
    Msg(level, time, sid, std::string(msg));
}


/*
 * megamol::core::utility::log::OfflineTarget::Msg
 */
void megamol::core::utility::log::OfflineTarget::Msg(Log::UINT level, megamol::core::utility::log::Log::TimeStamp time,
    megamol::core::utility::log::Log::SourceID sid, std::string const& msg) {
    // Do not check the level. We store ALL messages
    if (this->msgCnt < this->bufSize) {
        this->msgs[this->msgCnt].level = level;
        this->msgs[this->msgCnt].time = time;
        this->msgs[this->msgCnt].sid = sid;
        this->msgs[this->msgCnt].msg = msg;
        this->msgCnt++;
    } else {
        this->omittedCnt++;
    }
}


/*
 * megamol::core::utility::log::OfflineTarget::Reecho
 */
void megamol::core::utility::log::OfflineTarget::Reecho(megamol::core::utility::log::Log::Target& target, bool remove) {
    for (unsigned int i = 0; i < this->msgCnt; i++) {
        target.Msg(this->msgs[i].level, this->msgs[i].time, this->msgs[i].sid, this->msgs[i].msg);
    }
    if (remove)
        this->msgCnt = 0;
    if (this->omittedCnt > 0) {
        std::stringstream omg;
        omg << this->omittedCnt << " offline log message" << ((this->omittedCnt == 1) ? "" : "s") << " omitted\n";
        target.Msg(Log::LEVEL_WARN, Log::CurrentTimeStamp(), Log::CurrentSourceID(), omg.str());
        if (remove)
            this->omittedCnt = 0;
    }
}


/*
 * megamol::core::utility::log::OfflineTarget::SetBufferSize
 */
void megamol::core::utility::log::OfflineTarget::SetBufferSize(unsigned int bufferSize) {
    OfflineMessage* om = this->msgs;
    this->msgs = new OfflineMessage[bufferSize];
    unsigned int cnt = std::min(this->msgCnt, bufferSize);
    this->omittedCnt += (this->msgCnt - cnt);
    this->bufSize = bufferSize;
    for (unsigned int i = 0; i < cnt; i++) {
        this->msgs[i].level = om[i].level;
        this->msgs[i].time = om[i].time;
        this->msgs[i].sid = om[i].sid;
        this->msgs[i].msg = om[i].msg;
    }
    delete[] om;
}
