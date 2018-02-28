#include "stdafx.h"
#include "VrpnTracker.h"

#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/FloatParam.h"

#include "vislib/StringConverter.h"
#include "vislib/tchar.h"
#include "vislib/Trace.h"

#include "vislib/math/Matrix.h"

#include "vislib/sys/Log.h"
#include "vislib/sys/SystemInformation.h"

#include <chrono>


tracking::VrpnTracker::VrpnTracker(void) : Module(),
        motionDevice("motion"), buttonDevice("button"),
        getCamSlot("getcam", "Camera getter."),
        activeNodeSlot("activeNode", "Enables VRPN only on the node with the specified name."),
        connectSlot("connect", "Connects the two devices as specified."),
        disconnectSlot("disconnect", "Disconnects the two devices.") {
    for (auto slot : this->motionDevice.GetParams()) {
        this->MakeSlotAvailable(slot);
    }
    for (auto slot : this->buttonDevice.GetParams()) {
        this->MakeSlotAvailable(slot);
    }
    for (auto slot : this->manipulator .GetParams()) {
        this->MakeSlotAvailable(slot);
    }

    this->connectSlot << new core::param::ButtonParam();
    this->connectSlot.SetUpdateCallback(this, &VrpnTracker::onConnect);
    this->MakeSlotAvailable(&this->connectSlot);

    this->disconnectSlot << new core::param::ButtonParam();
    this->disconnectSlot.SetUpdateCallback(this, &VrpnTracker::onDisconnect);
    this->MakeSlotAvailable(&this->disconnectSlot);

    this->getCamSlot.SetCompatibleCall<core::view::CallCamParamSyncDescription>();
    this->MakeSlotAvailable(&this->getCamSlot);

    this->activeNodeSlot << new core::param::StringParam(_T(""));
    this->MakeSlotAvailable(&this->activeNodeSlot);
}

tracking::VrpnTracker::~VrpnTracker(void) {
    this->Release();
}

bool tracking::VrpnTracker::create(void) {
    return true;
}

void tracking::VrpnTracker::release(void) {
    this->stop();
}

void tracking::VrpnTracker::mainLoopBody(void) {
    VLTRACE(vislib::Trace::LEVEL_VL_INFO, "VRPN main loop.\n");

    while (this->isRunning) {
        //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "Motion main loop.\n");
        this->motionDevice.MainLoop();
        //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "Button main loop.\n");
        this->buttonDevice.MainLoop();
        std::this_thread::yield();
    }
}

void VRPN_CALLBACK tracking::VrpnTracker::onTrackerChanged(void *userData, const vrpn_TRACKERCB vrpnData) {
    //VLTRACE
    //(
    //    vislib::Trace::LEVEL_VL_INFO,
    //    "OnMotion: Position = (%g, %g, %g), Rotation = (%g, %g, %g, %g)\n",

    //    vrpnData. pos[0], vrpnData. pos[1], vrpnData. pos[2],
    //    vrpnData.quat[0], vrpnData.quat[1], vrpnData.quat[2], vrpnData.quat[3]
    //);

    VrpnTracker *instance = (VrpnTracker *) userData;

    // Save data
    instance->manipulator.SetOrientation(Manipulator::QuaternionType(
        static_cast<graphics::SceneSpaceType>(vrpnData.quat[0]),
        static_cast<graphics::SceneSpaceType>(vrpnData.quat[1]),
        static_cast<graphics::SceneSpaceType>(vrpnData.quat[2]),
        static_cast<graphics::SceneSpaceType>(vrpnData.quat[3])));

    instance->manipulator.SetPosition(Manipulator::PointType(
        static_cast<graphics::SceneSpaceType>(vrpnData.pos[0]),
        static_cast<graphics::SceneSpaceType>(vrpnData.pos[1]),
        static_cast<graphics::SceneSpaceType>(vrpnData.pos[2])));

    //// TODO: Zero out values other than max
    //float speed = instance->translationSpeedSlot.Param<core::param::FloatParam>()->Value();
    //instance->translation = (instance->curPos - instance->prePos) * speed;

    instance->manipulator.ApplyTransformations();
}

void VRPN_CALLBACK tracking::VrpnTracker::onButtonChanged(void *userData, const vrpn_BUTTONCB vrpnData) {
    //VLTRACE
    //(
    //    vislib::Trace::LEVEL_VL_INFO,
    //    "OnButtonPressed: Button = %d, State = %d\n",

    //    vrpnData.button,
    //    vrpnData.state
    //);

    VrpnTracker *instance = (VrpnTracker *) userData;
    instance->manipulator.OnButtonChanged(vrpnData.button, (vrpnData.state != 0));
}


bool tracking::VrpnTracker::onConnect(core::param::ParamSlot& slot) {
    this->stop();

    core::view::CallCamParamSync *call = this->getCamSlot.CallAs<core::view::CallCamParamSync>();

    if (call == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(_T("No call for retrieving the camera that should be manipulated by VRPN is registered."));
        return false; 
    }

    vislib::sys::Log::DefaultLog.WriteInfo(_T("Retrieving camera ..."));

    (*call)(core::view::CallCamParamSync::IDX_GET_CAM_PARAMS);
    auto camera = call->PeekCamParams();
    this->manipulator.SetCamParams(camera);

    vislib::sys::Log::DefaultLog.WriteInfo(_T("Registering VRPN trackers ..."));

    this->motionDevice.Connect();
    this->motionDevice.Register<vrpn_TRACKERCHANGEHANDLER>(&VrpnTracker::onTrackerChanged, this);

    this->buttonDevice.Connect();
    this->buttonDevice.Register<vrpn_BUTTONCHANGEHANDLER >(&VrpnTracker::onButtonChanged, this);

    this->start();

    return true;
}

bool tracking::VrpnTracker::onDisconnect(core::param::ParamSlot& slot) {
    this->stop();

    this->motionDevice.Disconnect();
    this->buttonDevice.Disconnect();

    return true;
}

void tracking::VrpnTracker::stop(void) {
    vislib::sys::Log::DefaultLog.WriteInfo(_T("Stopping VRPN trackers ..."));

    this->isRunning = false;

    if (this->mainLoopThread.get() != nullptr && this->mainLoopThread.get()->joinable()) {
        this->mainLoopThread.get()->join();
    }
}

void tracking::VrpnTracker::start(void) {
    vislib::TString computerName;
    vislib::sys::SystemInformation::ComputerName(computerName);
    vislib::TString activeNode = this->activeNodeSlot.Param<core::param::StringParam>()->Value();

    if (activeNode.IsEmpty() || computerName.Equals(activeNode, false)) {
        vislib::sys::Log::DefaultLog.WriteInfo(_T("Starting VRPN trackers ..."));

        this->isRunning = true;
        this->mainLoopThread = std::make_shared<std::thread>(&VrpnTracker::mainLoopBody, this);
    }
}