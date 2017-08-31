/*
 * LogEchoTarget.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_LOGECHOTARGET_H_INCLUDED
#define MEGAMOLCORE_LOGECHOTARGET_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/api/MegaMolCore.h"
#include "vislib/sys/Log.h"


namespace megamol {
namespace core {
namespace utility {

    /*
     * internal helper class of external echo targets.
     */
    class LogEchoTarget : public vislib::sys::Log::Target {
    public:

        /** ctor */
        LogEchoTarget(void);

        /** 
         * ctor 
         *
         * @param target The targeted output function
         */
        LogEchoTarget(mmcLogEchoFunction target);

        /** dtor */
        virtual ~LogEchoTarget();

        /**
         * Sets the targeted output function.
         *
         * @param target The targeted output function
         */
        void SetTarget(mmcLogEchoFunction target);

        /**
            * Writes a message to the log target
            *
            * @param level The level of the message
            * @param time The time stamp of the message
            * @param sid The object id of the source of the message
            * @param msg The message text itself
            */
        virtual void Msg(UINT level, vislib::sys::Log::TimeStamp time,
            vislib::sys::Log::SourceID sid, const char *msg);

    private:

        /** the output target */
        mmcLogEchoFunction target;

    };

} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */


#endif /* MEGAMOLCORE_LOGECHOTARGET_H_INCLUDED */
