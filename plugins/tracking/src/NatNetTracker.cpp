/*
 * NatNetTracker.cpp
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "NatNetTracker.h"

#include <thread>

#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/BoolParam.h"

#include "vislib/Trace.h"

#include "vislib/sys/Log.h"
#include "vislib/sys/SystemInformation.h"


/*
 * megamol::tracking::NatNetTracker::NatNetTracker
 */
megamol::tracking::NatNetTracker::NatNetTracker(void) :
        buttonDevice("button"), buttonStates(0),
        paramConnect("connect", "Establish the connection to the tracker."),
        paramActiveNode("activeNode", "Enables the tracker only on the node with the specified name."),
        paramDisconnect("disconnect", "Terminate the connection to the tracker."),
        paramValidButtons("validButtons", "Selects the buttons that are reported by the getState call."),
        paramStick("stick", "The name of the rigid body to track as stick."),
        paramGlasses("glasses", "The name of the rigid body to track as glasses."),
        paramEnableGlasses("enableGlasses", "Enables the tracking of the glasses."),
        slotGetCam("getCamera", "Connects the tracker to the module which of the camera should be manipulated."),
        slotGetState("getState", "Retrieves the current state of the 6DOF device.") {
    for (auto slot : this->buttonDevice.GetParams()) {
        this->MakeSlotAvailable(slot);
    }
    for (auto slot : this->manipulator.GetParams()) {
        this->MakeSlotAvailable(slot);
    }
    for (auto slot : this->motionDevices.GetParams()) {
        this->MakeSlotAvailable(slot);
    }

    this->paramActiveNode << new core::param::StringParam(_T(""));
    this->MakeSlotAvailable(&this->paramActiveNode);

    this->paramConnect << new core::param::ButtonParam();
    this->paramConnect.SetUpdateCallback(this, &NatNetTracker::onConnect);
    this->MakeSlotAvailable(&this->paramConnect);

    this->paramDisconnect << new core::param::ButtonParam();
    this->paramDisconnect.SetUpdateCallback(this, &NatNetTracker::onDisconnect);
    this->MakeSlotAvailable(&this->paramDisconnect);

    this->paramValidButtons << new core::param::IntParam(-1);
    this->MakeSlotAvailable(&this->paramValidButtons);

    this->paramStick << new core::param::StringParam(_T(""));
    this->MakeSlotAvailable(&this->paramStick);

    this->paramGlasses << new core::param::StringParam(_T(""));
    this->MakeSlotAvailable(&this->paramGlasses);

    this->paramEnableGlasses << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->paramEnableGlasses);

    this->slotGetCam.SetCompatibleCall<core::view::CallCamParamSyncDescription>();
    this->MakeSlotAvailable(&this->slotGetCam);

    this->slotGetState.SetCallback(
        core::view::Call6DofInteraction::ClassName(),
        core::view::Call6DofInteraction::FunctionName(0),
        &NatNetTracker::onGetState);
    this->MakeSlotAvailable(&this->slotGetState);
}

/*
 * megamol::tracking::NatNetTracker::~NatNetTracker
 */
megamol::tracking::NatNetTracker::~NatNetTracker(void) {
    this->Release();
}


/*
 * megamol::tracking::NatNetTracker::create
 */
bool megamol::tracking::NatNetTracker::create(void) {
    return true;
}


/*
 * megamol::tracking::NatNetTracker::release
 */
void megamol::tracking::NatNetTracker::release(void) {
    this->isRunning.store(false);
    this->motionDevices.Disconnect();
}


/*
 * megamol::tracking::NatNetTracker::onButtonChanged
 */
void VRPN_CALLBACK megamol::tracking::NatNetTracker::onButtonChanged(void *userData, const vrpn_BUTTONCB vrpnData) {
    //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "Button = %d, State = %d\n", vrpnData.button, vrpnData.state);

    auto that = static_cast<NatNetTracker *>(userData);
    that->manipulator.OnButtonChanged(vrpnData.button, (vrpnData.state != 0));

    // Remember the button state.
    core::view::Call6DofInteraction::ButtonMaskType mask = 1 << vrpnData.button;

    if (vrpnData.state != 0) {
        that->buttonStates |= mask;
    } else {
        that->buttonStates &= ~mask;
    }
}


/*
 * megamol::tracking::NatNetTracker::onConnect
 */
bool megamol::tracking::NatNetTracker::onConnect(core::param::ParamSlot& slot) {
    using vislib::sys::Log;

    this->onDisconnect(this->paramDisconnect);

    vislib::TString computerName;
    vislib::sys::SystemInformation::ComputerName(computerName);

    auto activeNode = this->paramActiveNode.Param<core::param::StringParam>()->Value();

    if (!activeNode.IsEmpty() && !computerName.Equals(activeNode, false)) {
        Log::DefaultLog.WriteWarn(_T("%s is not enabled for receiving tracker updates."), computerName.PeekBuffer());
        return true;
    }

    auto call = this->slotGetCam.CallAs<core::view::CallCamParamSync>();

    if (call == nullptr) {
        Log::DefaultLog.WriteWarn(_T("No call for retrieving the camera that should be manipulated by the 6DOF device is registered."));
        return false; 
    }

    Log::DefaultLog.WriteInfo(_T("Retrieving camera ..."));
	
	if (!(*call)(core::view::CallCamParamSync::IDX_GET_CAM_PARAMS)) {
		Log::DefaultLog.WriteWarn(_T("Camera parameters could not be recieved."));
		return false;
	}
    auto camera = call->PeekCamParams();
    this->manipulator.SetCamParams(camera);
	
    Log::DefaultLog.WriteInfo(_T("Starting trackers ..."));
	
    this->buttonDevice.Connect(); // Call before registration!
	this->buttonDevice.Register<vrpn_BUTTONCHANGEHANDLER>(&NatNetTracker::onButtonChanged, this);
	
    this->motionDevices.ClearCallbacks();	
    try {
        this->motionDevices.Connect(); // Call after registration!
        this->stick = T2A(this->paramStick.Param<core::param::StringParam>()->Value());
        this->glasses = T2A(this->paramGlasses.Param<core::param::StringParam>()->Value());
    } catch (Exception e) {
        Log::DefaultLog.WriteError(e.GetMsg());
    }
	this->motionDevices.Register(this->stick, NatNetDevicePool::CallbackType(*this, &NatNetTracker::onStickMotion));
	this->motionDevices.Register(this->glasses, NatNetDevicePool::CallbackType(*this, &NatNetTracker::onGlassesMotion));
	
    this->isRunning = true;
	
	Log::DefaultLog.WriteInfo(_T("Starting NatNet thread ..."));
    std::thread thread([this]() {
        while (this->isRunning) {
            //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "VRPN main loop.\n");
            this->buttonDevice.MainLoop();
            std::this_thread::yield();
        }
    });
	
    thread.detach();
	
    return true;
}


/*
 * megamol::tracking::NatNetTracker::onDisconnect
 */
bool megamol::tracking::NatNetTracker::onDisconnect(core::param::ParamSlot& slot) {
    this->isRunning.store(false);
    this->motionDevices.Disconnect();
    return true;
}


/*
 * megamol::tracking::NatNetTracker::onGetState
 */
bool megamol::tracking::NatNetTracker::onGetState(core::Call& call) {
    auto valid = static_cast<core::view::Call6DofInteraction::ButtonMaskType>(this->paramValidButtons.Param<core::param::IntParam>()->Value());

    try {
        auto& c = dynamic_cast<core::view::Call6DofInteraction&>(call);

        c.SetButtonStates(this->buttonStates & valid);
        c.SetOrientation(this->manipulator.GetOrientation());
        c.SetPosition(this->manipulator.GetPosition());
        c.SetRigidBody(this->stick);

        return true;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteWarn(_T("Call to getState failed: %s "), e.GetMsg());
        return false;
    }
}


/*
 * megamol::tracking::NatNetTracker::onMotion
 */
void megamol::tracking::NatNetTracker::onStickMotion(const sRigidBodyData& data) {
    //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "Position = (%f, %f, %f), "
    //    "Orientation = (%f, %f, %f, %f)\n", data.x, data.y, data.z,
    //    data.qx, data.qy, data.qz, data.qw);

    this->manipulator.SetOrientation(Manipulator::QuaternionType(
        static_cast<graphics::SceneSpaceType>(data.qx),
        static_cast<graphics::SceneSpaceType>(data.qy),
        static_cast<graphics::SceneSpaceType>(data.qz),
        static_cast<graphics::SceneSpaceType>(data.qw)));

    //vislib::graphics::SceneSpaceType angle;
    //vislib::graphics::SceneSpaceVector3D axis;
    //this->manipulator.GetOrientation().AngleAndAxis(angle, axis);
    //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "Axis = (%f, %f, %f), Angle = %f\n", axis.X(), axis.Y(), axis.Z(), angle);

    this->manipulator.SetPosition(Manipulator::PointType(
        static_cast<graphics::SceneSpaceType>(data.x),
        static_cast<graphics::SceneSpaceType>(data.y),
        static_cast<graphics::SceneSpaceType>(data.z)));

    this->manipulator.ApplyTransformations();
}

void megamol::tracking::NatNetTracker::onGlassesMotion(const sRigidBodyData& data) {
    if (!this->paramEnableGlasses.Param<core::param::BoolParam>()->Value()) {
        return;
    }

    // TODO: Transform camera.
}