/*
 * VrpnDevice.h
 *
 * Copyright (C) 2014 by Tilo Pfannkuch
 * Copyright (C) 2008-2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLVRPN_VRPNDEVICE_H_INCLUDED
#define MEGAMOLVRPN_VRPNDEVICE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

//#define MEGAMOLVRPN_VRPNDEVICE_WRITE_PLAYBACKLOG

#include "param/ParamSlot.h"
#include "vrpn/vrpn_Tracker.h"
#include "vislib/sys/Log.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "param/StringParam.h"
#include "param/IntParam.h"
#include "param/EnumParam.h"
#include <string>
#include <vector>
#include <memory>

using namespace megamol;
using namespace vislib;

namespace megamol {
namespace vrpnModule {

    template <class R> class VrpnDevice {

    public:

        static class Protocol { public: enum Enum { Udp, Tcp, Count }; };
        static const std::string ProtocolToString[Protocol::Count];

        VrpnDevice(const std::string& prefix);

        template <typename H> void Register(H handler, void *userData);
        std::vector<core::param::ParamSlot *> &GetParams(void);
        void MainLoop(void);
        void Connect(void);
        void Disconnect(void);

    private:

        std::shared_ptr<R> remote;
        core::param::ParamSlot nameSlot;
        core::param::ParamSlot serverSlot;
        core::param::ParamSlot portSlot;
        core::param::ParamSlot protocolSlot;
        std::vector<core::param::ParamSlot *> paramSlots;
        TString url;

    };

} /* end namespace vrpnModule */
} /* end namespace megamol */

template <class R>
const std::string vrpnModule::VrpnDevice<R>::ProtocolToString[Protocol::Count] = { "udp", "tcp" };

template <class R>
vrpnModule::VrpnDevice<R>::VrpnDevice(const std::string& prefix) :
nameSlot((prefix + "::name").c_str(), "The name of the device."),
serverSlot((prefix + "::server").c_str(), "The sever hosting the device."),
portSlot((prefix + "::port").c_str(), "The port used for connecting to the device."),
protocolSlot((prefix + "::protocol").c_str(), "The protocol used for connecting to the device.") {
    core::param::EnumParam *enumParam = new core::param::EnumParam(Protocol::Tcp);
    enumParam->SetTypePair(Protocol::Tcp, "TCP");
    enumParam->SetTypePair(Protocol::Udp, "UDP");

    this->nameSlot << new core::param::StringParam("Stick"); this->paramSlots.push_back(&this->nameSlot);
    this->serverSlot << new core::param::StringParam("mini"); this->paramSlots.push_back(&this->serverSlot);
    this->portSlot << new core::param::IntParam(3883); this->paramSlots.push_back(&this->portSlot);
    this->protocolSlot << (enumParam); this->paramSlots.push_back(&this->protocolSlot);

}

template <class R>
void vrpnModule::VrpnDevice<R>::Connect(void) {
    Protocol::Enum protocol = static_cast<Protocol::Enum>(this->protocolSlot.Param<core::param::EnumParam  >()->Value());
    std::string    name = T2A(this->nameSlot.Param<core::param::StringParam>()->Value());
    std::string    server = T2A(this->serverSlot.Param<core::param::StringParam>()->Value());
    int            port = (this->portSlot.Param<core::param::IntParam   >()->Value());

    this->url.Format
        (
        _T("%hs@%hs://%hs:%d"),

        name.c_str(),
        VrpnDevice::ProtocolToString[protocol].c_str(),
        server.c_str(),
        port
        );

    vislib::sys::Log::DefaultLog.WriteInfo
        (
        _T("Connecting to VRPN tracker %s ..."),

        this->url.PeekBuffer()
        );

    vrpn_Connection *connection = vrpn_get_connection_by_name
        (
        T2A(this->url)

#ifdef MEGAMOLVRPN_VRPNDEVICE_WRITE_PLAYBACKLOG
        ,(name +  "_local_in.log").c_str(), (name +  "_local_out.log").c_str(),
        (name + "_remote_in.log").c_str(), (name + "_remote_out.log").c_str()
#endif
        );

    this->remote = std::make_shared<R>(T2A(this->url), connection);
    this->remote->shutup = true;
}

template <class R>
void vrpnModule::VrpnDevice<R>::Disconnect(void) {
    // TODO: Is this the correct way of disconnecting?
    this->remote = nullptr;
}

template <class R> template <typename H>
void vrpnModule::VrpnDevice<R>::Register(H handler, void *userData) {
    this->remote->register_change_handler(userData, handler);
}

template <class R>
std::vector<core::param::ParamSlot *> &vrpnModule::VrpnDevice<R>::GetParams(void) {
    return this->paramSlots;
}

template <class R>
void vrpnModule::VrpnDevice<R>::MainLoop(void) {
	if (this->remote != nullptr) {
		this->remote->mainloop();
	}
}

#endif /* MEGAMOLVRPN_VRPNDEVICE_H_INCLUDED */
