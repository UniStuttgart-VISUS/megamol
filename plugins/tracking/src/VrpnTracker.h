/*
 * VrpnModule.h
 *
 * Copyright (C) 2014 by Tilo Pfannkuch
 * Copyright (C) 2008-2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLVRPN_VRPNMODULE_H_INCLUDED
#define MEGAMOLVRPN_VRPNMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#define MEGAMOLVRPN_VRPNMODULE_SLEEP_DURATION 0

#include "Manipulator.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vrpn_Tracker.h"
#include "vrpn_Button.h"
#include "VrpnDevice.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/view/CallCamParamSync.h"
#include "vislib/math/Quaternion.h"
#include <atomic>
#include <thread>

using namespace megamol;
using namespace vislib;

namespace megamol {
namespace tracking {

    class VrpnTracker : public core::Module {

    public:

        static const char *ClassName(void) {
            return "VrpnModule";
        }

        static const char *Description(void) {
            return "This module manipulates the camera via tracking.";
        }

        static bool IsAvailable(void) {
            return true;
        }

        VrpnTracker(void);
        ~VrpnTracker(void);

    protected:

        virtual bool create(void);
        virtual void release(void);


    private:

        static void VRPN_CALLBACK onTrackerChanged(void *userData, const vrpn_TRACKERCB vrpnData);
        static void VRPN_CALLBACK onButtonChanged (void *userData, const vrpn_BUTTONCB vrpnData);

        VrpnDevice<vrpn_Tracker_Remote> motionDevice;
        VrpnDevice<vrpn_Button_Remote> buttonDevice;
        core::param::ParamSlot activeNodeSlot;
        core::param::ParamSlot connectSlot;
        core::param::ParamSlot disconnectSlot;
        std::shared_ptr<std::thread> mainLoopThread;
        std::atomic<bool> isRunning;
        core::CallerSlot getCamSlot;
        Manipulator manipulator;

        void mainLoopBody(void);
        bool onConnect(core::param::ParamSlot& slot);
        bool onDisconnect(core::param::ParamSlot& slot);
        void stop(void);
        void start(void);
    };

} /* end namespace tracking */
} /* end namespace megamol */

#endif /* MEGAMOLVRPN_VRPNMODULE_H_INCLUDED */