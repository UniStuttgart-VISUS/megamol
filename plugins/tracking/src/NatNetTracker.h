/*
 * NatNetTracker.h
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/param/ParamSlot.h"

#include <atomic>

#include "vrpn_Button.h"

#include "mmcore/view/Call6DofInteraction.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "Manipulator.h"
#include "mmcore/Module.h"
#include "NatNetDevicePool.h"
#include "VrpnDevice.h"


namespace megamol {
namespace tracking {

    /**
     * 6DOF tracker using NaturalPoint's native SDK.
     */
    class NatNetTracker : public megamol::core::Module {

    public:

        static const char *ClassName(void) {
            return "NatNetTracker";
        }

        static const char *Description(void) {
            return "This module manipulates the camera using NaturalPoint's NatNet library and VRPN for buttons.";
        }

        static bool IsAvailable(void) {
            return true;
        }

        NatNetTracker(void);
        virtual ~NatNetTracker(void);

    protected:

        virtual bool create(void);
        virtual void release(void);

    private:

        typedef megamol::core::Module Base;

        static void VRPN_CALLBACK onButtonChanged(void *userData, const vrpn_BUTTONCB vrpnData);

        bool onConnect(core::param::ParamSlot& slot);
        bool onDisconnect(core::param::ParamSlot& slot);
        bool onGetState(core::Call& call);
        void onStickMotion(const sRigidBodyData& data);
        void onGlassesMotion(const sRigidBodyData& data);

        /** The VRPN device that handles button presses. */
        VrpnDevice<vrpn_Button_Remote> buttonDevice;

        /** Remembers the current button states. */
        core::view::Call6DofInteraction::ButtonMaskType buttonStates;

        /** Determines the running state of the VRPN thread. */
        std::atomic<bool> isRunning;

        /** The manipulator that computes the transformation. */
        Manipulator manipulator;

        /** Handles position changes via NatNet. */
        NatNetDevicePool motionDevices;

        core::param::ParamSlot paramActiveNode;
        core::param::ParamSlot paramConnect;
        core::param::ParamSlot paramDisconnect;
        core::param::ParamSlot paramValidButtons;
        core::param::ParamSlot paramStick;
        core::param::ParamSlot paramGlasses;
        core::param::ParamSlot paramEnableGlasses;
        
        core::CallerSlot slotGetCam;
        core::CalleeSlot slotGetState;

        std::string stick;
        std::string glasses;
    };

} /* end namespace tracking */
} /* end namespace megamol */
