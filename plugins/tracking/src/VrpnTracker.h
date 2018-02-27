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
#include "Module.h"
#include "param/ParamSlot.h"
#include "vrpn/vrpn_Tracker.h"
#include "vrpn/vrpn_Button.h"
#include "VrpnDevice.h"
#include "CallerSlot.h"
#include "CalleeSlot.h"
#include "view/CallCamParamSync.h"
#include "vislib/math/Quaternion.h"
#include <atomic>
#include <thread>

using namespace megamol;
using namespace vislib;

namespace megamol {
namespace vrpnModule {

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

} /* end namespace vrpnModule */
} /* end namespace megamol */

#endif /* MEGAMOLVRPN_VRPNMODULE_H_INCLUDED */