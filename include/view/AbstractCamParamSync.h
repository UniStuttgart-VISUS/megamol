/*
 * AbstractCamParamSync.h
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTCAMPARAMSYNC_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTCAMPARAMSYNC_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallCamParamSync.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "Module.h"

#include "api/MegaMolCore.std.h"

#include "param/ParamSlot.h"



namespace megamol {
namespace core {
namespace view {

    /**
     * Base class that allows for camera synchronisation between views.
     */
    class MEGAMOLCORE_API AbstractCamParamSync {

    public:

        /** Finalise the instance. */
        virtual ~AbstractCamParamSync(void);

    protected:

        /** Initialise a new instance. */
        AbstractCamParamSync(void);

        /**
         * Incoming call from a slave to this master
         *
         * @param c The incoming call
         *
         * @return Some return value
         */
        virtual bool OnGetCamParams(CallCamParamSync& c) = 0;

        /**
         * If the call for retrieving the camera is registered, invoke it and
         * assign it by value to 'dst'.
         *
         * @param dst A camera parameter smart pointer.
         */
        void SyncCamParams(CallCamParamSync::CamParams dst);

        /** Slot used if the derived class needs to receive the parameters. */
        CallerSlot slotGetCamParams;

        /** Slot used if the derived class must send the parameters. */
        CalleeSlot slotSetCamParams;

    private:

        bool onGetCamParams(Call & c);
    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTCAMPARAMSYNC_H_INCLUDED */
