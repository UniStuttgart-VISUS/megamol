/*
 * testthelog.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testthelog.h"
#include "testhelper.h"

#include "vislib/Log.h"

void TestTheLogWithPhun(void) {
    vislib::sys::Log &log = vislib::sys::Log::DefaultLog;
    log.SetOfflineMessageBufferSize(3);

    log.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Error 1");
    log.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Warning 1");
    log.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Info 1");

    log.SetLevel(vislib::sys::Log::LEVEL_ALL);

    log.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Error 2");
    log.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Warning 2");
    log.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Info 2");

    log.SetLogFileName("testlog.log", true);

    log.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Error 3");
    log.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Warning 3");
    log.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Info 3");

    log.SetLogFileName(static_cast<char*>(NULL), true);

    log.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Error 4");
    log.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Warning 4");
    log.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Info 4");

}
