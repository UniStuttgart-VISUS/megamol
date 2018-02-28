/*
 * NatNetDevicePool.cpp
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "NatNetDevicePool.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"

#include "vislib/assert.h"
#include "vislib/Trace.h"

#include "vislib/sys/Log.h"


/*
 * megamol::tracking::NatNetDevicePool::NatNetDevicePool
 */
megamol::tracking::NatNetDevicePool::NatNetDevicePool(void) :

    client(nullptr),

    paramClient        ("motion::client"     , "Specifies the local IP address used for NatNet."),
    paramCommandPort   ("motion::commandPort", "Specifies the port used for NatNet commands."   ),
    paramConnectionType("motion::type"       , "Specifies the type of NatNet connection."       ),
    paramDataPort      ("motion::dataPort"   , "Specifies the port used for NatNet data."       ),
    //paramRigidBody     ("motion::rigidBody"  , "The name of the rigid body to track."           ),
    paramServer        ("motion::server"     , "Specifies the NatNet server."                   ),
    paramVerbosity     ("motion::verbosity"  , "Specifies the verbosity of the NatNet client."  )

{
    this->paramClient << new core::param::StringParam("");
    this->params.push_back(&this->paramClient);

    this->paramCommandPort << new core::param::IntParam(0, 0, 1 << 16);
    this->params.push_back(&this->paramCommandPort);

    this->paramDataPort << new core::param::IntParam(0, 0, 1 << 16);
    this->params.push_back(&this->paramDataPort);

    auto enumParam = new core::param::EnumParam(ConnectionType_Unicast);
    enumParam->SetTypePair(ConnectionType_Unicast, "Unicast");
    enumParam->SetTypePair(ConnectionType_Multicast, "Multicast");
    this->paramConnectionType << enumParam;
    this->params.push_back(&this->paramConnectionType);

    enumParam = new core::param::EnumParam(Verbosity_Error);
    enumParam->SetTypePair(Verbosity_None, "None");
    enumParam->SetTypePair(Verbosity_Error, "Error");
    enumParam->SetTypePair(Verbosity_Warning, "Warning");
    enumParam->SetTypePair(Verbosity_Info, "Info");
    enumParam->SetTypePair(Verbosity_Debug, "Debug");
    this->paramVerbosity << enumParam;
    this->params.push_back(&this->paramVerbosity);

    this->paramServer << new core::param::StringParam("");
    this->params.push_back(&this->paramServer);

    //this->paramRigidBody << new core::param::StringParam("");
    //this->params.push_back(&this->paramRigidBody);
}


/*
 * megamol::tracking::NatNetDevicePool::~NatNetDevicePool
 */
megamol::tracking::NatNetDevicePool::~NatNetDevicePool(void)
{
    this->Disconnect();
}


/*
 * megamol::tracking::NatNetDevicePool::Connect
 */
void megamol::tracking::NatNetDevicePool::Connect(void)
{
    using namespace megamol::core::param;
    using vislib::sys::Log;

    /* Terminate previous connection. */
    this->Disconnect();

    char localAddress[128];
    char remoteAddress[128];
    sDataDescriptions *dataDesc = nullptr;
    int ec = ErrorCode_OK;
    sServerDescription serverDesc;
    unsigned char version[4];
    void *response = nullptr;
    int cntResponse = 0;

    auto ip = this->paramClient.Param<StringParam>()->Value();
    auto cmdPort = this->paramCommandPort.Param<IntParam>()->Value();
    auto dataPort = this->paramDataPort.Param<IntParam>()->Value();
    //auto rigidBody = this->paramRigidBody.Param<StringParam>()->Value();
    auto server = this->paramServer.Param<StringParam>()->Value();
    auto type = this->paramConnectionType.Param<EnumParam>()->Value();
    auto verbosity = this->paramVerbosity.Param<EnumParam>()->Value();

    /* Move server address to non-constant buffer. */
    ::strcpy_s(localAddress, T2A(ip));
    ::strcpy_s(remoteAddress, T2A(server));

    /* Create the client. */
    this->client = new NatNetClient(type);

    // [optional] use old multicast group
    //theClient->SetMulticastAddress("224.0.0.1");

    /* Log NatNet version. */
    this->client->NatNetVersion(version);
    Log::DefaultLog.WriteInfo(_T("Using NatNet version %d.%d.%d.%d"),
        version[0], version[1], version[2], version[3]);

    /* Register callback handlers. */
    this->client->SetVerbosityLevel(verbosity);
    this->client->SetMessageCallback(NatNetDevicePool::onMessage);
    this->client->SetDataCallback(NatNetDevicePool::onData, const_cast<NatNetDevicePool *>(this));
    
    /* Connect to the server. */
    Log::DefaultLog.WriteInfo(_T("Connecting to NatNet server ..."));

    if ((cmdPort != 0) && (dataPort != 0))
    {
        ec = this->client->Initialize(localAddress, remoteAddress, cmdPort, dataPort);
    }
    else
    {
        ec = this->client->Initialize(localAddress, remoteAddress);
    }

    /* Check whether the connection was successful. */
    if (ec == ErrorCode_OK)
    {
        ::ZeroMemory(&serverDesc, sizeof(serverDesc));
        this->client->GetServerDescription(&serverDesc);

        if (!serverDesc.HostPresent)
        {
            throw vislib::Exception("No NatNet host is present.", __FILE__, __LINE__);
        }

        Log::DefaultLog.WriteInfo(_T("NatNet host application: %hs %d.%d.%d.%d"),
            serverDesc.szHostApp, serverDesc.HostAppVersion[0], 
            serverDesc.HostAppVersion[1], serverDesc.HostAppVersion[2],
            serverDesc.HostAppVersion[3]);

        Log::DefaultLog.WriteInfo(_T("Server side NatNet version: %d.%d.%d.%d"),
            serverDesc.NatNetVersion[0], serverDesc.NatNetVersion[1], 
            serverDesc.NatNetVersion[2], serverDesc.NatNetVersion[3]);

        Log::DefaultLog.WriteInfo(_T("NatNet server: %hs"),
            serverDesc.szHostComputerName);
    }
    else
    {
        throw vislib::Exception(_T("Unable to connect to NatNet server."), __FILE__, __LINE__);
    }

    //ec = this->client->SendMessageAndWait("TestRequest", &response, &cntResponse);
    //if (ec != ErrorCode_OK) {
    //    throw vislib::Exception(_T("Unable to process NatNet test request."),
    //        __FILE__, __LINE__);
    //}

    //ec = this->client.SendMessageAndWait("FrameRate", &response, &cntResponse);


    /* Look up rigid body IDs */

    Log::DefaultLog.WriteInfo(_T("Looking up rigid body IDs ..."));
    this->client->GetDataDescriptions(&dataDesc);

    if (dataDesc == nullptr)
    {
        throw vislib::Exception(_T("Unable to retrieve tracking data descriptions."), __FILE__, __LINE__);
    }

    for (int i = 0; i < dataDesc->nDataDescriptions; ++i)
    {
        if (dataDesc->arrDataDescriptions[i].type == Descriptor_RigidBody)
        {
            auto *rb = dataDesc->arrDataDescriptions[i].Data.RigidBodyDescription;

            std::string name(rb->szName);
            int id = rb->ID;

            this->ids[name] = id;
			Log::DefaultLog.WriteInfo("Ridgid body \"%s\" found at %d.\n", name.c_str(), id);
        }
    }
}


/*
 * megamol::tracking::NatNetDevicePool::Disconnect
 */
void megamol::tracking::NatNetDevicePool::Disconnect(void)
{
    if (this->client != nullptr)
    {
        this->client->Uninitialize();
        this->client = nullptr;
    }
}


/*
 * megamol::tracking::NatNetDevicePool::Register
 */
void megamol::tracking::NatNetDevicePool::Register(std::string rigidBodyName, CallbackType callback)
{
	auto it = this->ids.find(rigidBodyName);
	if (it != this->ids.end()) {
		this->callbacks[it->second].push_back(callback);
	}
}

/*
 * megamol::tracking::NatNetDevicePool::ClearCallbacks
 */
void megamol::tracking::NatNetDevicePool::ClearCallbacks(void)
{
    for (auto pair : this->callbacks)
    {
        pair.second.clear();
    }
}


/*
 * megamol::tracking::NatNetDevicePool::onData
 */
void __cdecl megamol::tracking::NatNetDevicePool::onData(sFrameOfMocapData *data, void *pUserData)
{
	auto that = static_cast<NatNetDevicePool *>(pUserData);

    ASSERT(data != nullptr);
    ASSERT(that != nullptr);

    for (int i = 0; i < data->nRigidBodies; ++i)
    {
        int id = data->RigidBodies[i].ID;
        int n = that->callbacks[id].size();

        if (n > 0)
        {
            // All zero seems to be an indicator that the rigid body is not
            // visible atm. Do not report this to the listener.
            bool isValid =
            (
                   (data->RigidBodies[i].qx != 0.0f)
                || (data->RigidBodies[i].qy != 0.0f)
                || (data->RigidBodies[i].qz != 0.0f)
                || (data->RigidBodies[i].qw != 0.0f)
                || (data->RigidBodies[i].x  != 0.0f)
                || (data->RigidBodies[i].y  != 0.0f)
                || (data->RigidBodies[i].z  != 0.0f)
            );

            //VLTRACE(vislib::Trace::LEVEL_VL_INFO, "ID = %d; MeanError = %f; "
            //    "Params = %d; Orientation = %f, %f, %f, %f; "
            //    "Position = %f, %f, %f; (isValid = %d)\n",
            //    data->RigidBodies[i].ID, data->RigidBodies[i].MeanError,
            //    data->RigidBodies[i].params, data->RigidBodies[i].qx,
            //    data->RigidBodies[i].qy, data->RigidBodies[i].qz,
            //    data->RigidBodies[i].qw, data->RigidBodies[i].x,
            //    data->RigidBodies[i].y, data->RigidBodies[i].z, isValid);

            if (isValid)
            {
                for (auto callback : that->callbacks[id])
                {
                    callback(data->RigidBodies[i]);
                }
            }
        }
    }
}


/*
 * megamol::tracking::NatNetDevicePool::onMessage
 */
void __cdecl megamol::tracking::NatNetDevicePool::onMessage(int msgType, char *msg)
{
    vislib::sys::Log::DefaultLog.WriteInfo(_T("NatNet: %hs"), msg);
}
