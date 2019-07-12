/*
 * AbstractTileView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/cluster/AbstractClusterView.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/cluster/InfoIconRenderer.h"
#include "mmcore/cluster/NetMessages.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/view/AbstractView.h"
#include "vislib/RawStorageSerialiser.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/net/IPCommEndPoint.h"
#include "vislib/net/IPEndPoint.h"
#include "vislib/net/NetworkInformation.h"
#include "vislib/net/TcpCommChannel.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/sys/sysfunctions.h"


using namespace megamol::core;


/*
 * cluster::AbstractClusterView::InitCameraHookHandler::InitCameraHookHandler
 */
cluster::AbstractClusterView::InitCameraHookHandler::InitCameraHookHandler(cluster::CommChannel* channel)
    : view::AbstractView::Hooks(), channel(channel), frameCnt(0) {}


/*
 * cluster::AbstractClusterView::InitCameraHookHandler::~InitCameraHookHandler
 */
cluster::AbstractClusterView::InitCameraHookHandler::~InitCameraHookHandler(void) { this->channel = NULL; }


/*
 * cluster::AbstractClusterView::InitCameraHookHandler::BeforeRender
 */
void cluster::AbstractClusterView::InitCameraHookHandler::BeforeRender(view::AbstractView* view) {
    this->frameCnt++;
    if (this->frameCnt > 3) {
        vislib::net::SimpleMessage outMsg;

        view->UnregisterHook(this);

        outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_REQUEST_CAMERAVALUES);
        outMsg.GetHeader().SetBodySize(0);
        outMsg.AssertBodySize();
        if (this->channel != NULL) {
            this->channel->SendMessage(outMsg);
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_INFO, "Camera initialization request sent.\n");

        } else {
            vislib::sys::Log::DefaultLog.WriteMsg(
                vislib::sys::Log::LEVEL_ERROR, "Failed to send camera initialization request.\n");
        }

        delete this;
    }
}


/*
 * cluster::AbstractClusterView::AbstractClusterView
 */
cluster::AbstractClusterView::AbstractClusterView(void)
    : view::AbstractTileView()
    , ClusterControllerClient::Listener()
    , CommChannel::Listener()
    , ccc()
    , ctrlChannel()
    , lastPingTime(0)
    , serverAddressSlot("serverAddress", "The TCP/IP address of the server including the port")
    , setupState(SETUP_UNKNOWN)
    , graphInitData(NULL) {

    this->ccc.AddListener(this);
    this->MakeSlotAvailable(&this->ccc.RegisterSlot());
    this->ctrlChannel.AddListener(this);

    this->serverAddressSlot << new param::StringParam("");
    this->serverAddressSlot.SetUpdateCallback(&AbstractClusterView::onServerAddressChanged);
    this->MakeSlotAvailable(&this->serverAddressSlot);
}


/*
 * cluster::AbstractClusterView::~AbstractClusterView
 */
cluster::AbstractClusterView::~AbstractClusterView(void) {
    try {
        this->ccc.RemoveListener(this);
        this->ctrlChannel.RemoveListener(this);
        if (this->ctrlChannel.IsOpen()) {
            this->ctrlChannel.Close();
        }
    } catch (...) {
    }

    vislib::net::SimpleMessage* m = this->graphInitData;
    this->graphInitData = NULL;
    delete m;
}


/*
 * cluster::AbstractClusterView::ResetView
 */
void cluster::AbstractClusterView::ResetView(void) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::initClusterViewParameters
 */
void cluster::AbstractClusterView::initClusterViewParameters(void) {
    const utility::Configuration& cfg = this->GetCoreInstance()->Configuration();
    if (cfg.IsConfigValueSet("cmvshost")) {
        this->serverAddressSlot.Param<param::StringParam>()->SetValue(cfg.ConfigValue("cmvshost"));
    }
}


/*
 * cluster::AbstractClusterView::commPing
 */
void cluster::AbstractClusterView::commPing(void) {
    unsigned int ping = vislib::sys::GetTicksOfDay() / 1000;
    if (ping == this->lastPingTime) return;
    this->lastPingTime = ping;

    if (!this->ctrlChannel.IsOpen()) {
        this->ccc.SendUserMsg(ClusterControllerClient::USRMSG_QUERYHEAD, NULL, 0);
    }
}


/*
 * cluster::AbstractClusterView::renderFallbackView
 */
void cluster::AbstractClusterView::renderFallbackView(void) {

    ::glViewport(0, 0, this->getViewportWidth(), this->getViewportHeight());
    ::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    ::glClear(GL_COLOR_BUFFER_BIT);

    vislib::net::AbstractSimpleMessage* initmsg = this->graphInitData;
    if (initmsg != NULL) {
        this->graphInitData = NULL;
        this->GetCoreInstance()->SetupGraphFromNetwork(static_cast<void*>(initmsg));
        delete initmsg;
        this->continueSetup();
        return;
    }

    if ((this->getViewportHeight() <= 1) || (this->getViewportWidth() <= 1)) return;

    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    float aspect = static_cast<float>(this->getViewportWidth()) / static_cast<float>(this->getViewportHeight());
    if ((this->getProjType() == vislib::graphics::CameraParameters::MONO_PERSPECTIVE) ||
        (this->getProjType() == vislib::graphics::CameraParameters::MONO_ORTHOGRAPHIC)) {
        if (aspect > 1.0f) {
            ::glScalef(2.0f / aspect, -2.0f, 1.0f);
        } else {
            ::glScalef(2.0f, -2.0f * aspect, 1.0f);
        }
        ::glTranslatef(-0.5f, -0.5f, 0.0f);
    } else {
        if (this->getEye() == vislib::graphics::CameraParameters::RIGHT_EYE) {
            ::glTranslatef(0.5f, 0.0f, 0.0f);
        } else {
            ::glTranslatef(-0.5f, 0.0f, 0.0f);
        }
        if (aspect > 2.0f) {
            ::glScalef(2.0f / aspect, -2.0f, 1.0f);
        } else {
            ::glScalef(1.0f, -1.0f * aspect, 1.0f);
        }
        ::glTranslatef(-0.5f, -0.5f, 0.0f);
    }
    const float border = 0.2f;
    ::glTranslatef(border, border, 0.0f);
    ::glScalef(1.0f - 2.0f * border, 1.0f - 2.0f * border, 0.0f);

    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();

    InfoIconRenderer::IconState icon = InfoIconRenderer::ICONSTATE_UNKNOWN;
    vislib::TString msg;
    this->getFallbackMessageInfo(msg, icon);
    InfoIconRenderer::RenderInfoIcon(icon, msg);
}


/*
 * cluster::AbstractClusterView::getFallbackMessageInfo
 */
void cluster::AbstractClusterView::getFallbackMessageInfo(
    vislib::TString& outMsg, InfoIconRenderer::IconState& outState) {
    outState = InfoIconRenderer::ICONSTATE_UNKNOWN;
    outMsg = _T("State unknown");
}


/*
 * cluster::AbstractClusterView::OnClusterUserMessage
 */
void cluster::AbstractClusterView::OnClusterUserMessage(cluster::ClusterControllerClient& sender,
    const cluster::ClusterController::PeerHandle& hPeer, bool isClusterMember, const UINT32 msgType,
    const BYTE* msgBody) {

    switch (msgType) {
    case ClusterControllerClient::USRMSG_HEADHERE:
        this->serverAddressSlot.Param<param::StringParam>()->SetValue(reinterpret_cast<const char*>(msgBody));
        break;
    case ClusterControllerClient::USRMSG_SHUTDOWN:
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_INFO, "Cluster Shutdown message received. Terminating application.\n");
        this->GetCoreInstance()->Shutdown();
        break;
    }
}


/*
 * cluster::AbstractClusterView::OnCommChannelConnect
 */
void cluster::AbstractClusterView::OnCommChannelConnect(cluster::CommChannel& sender) {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Connected to head node\n");

    this->continueSetup(SETUP_TIME);
}


/*
 * cluster::AbstractClusterView::OnCommChannelDisconnect
 */
void cluster::AbstractClusterView::OnCommChannelDisconnect(cluster::CommChannel& sender) {
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Disconnected from head node\n");
    this->serverAddressSlot.Param<param::StringParam>()->SetValue("", false);
}


/*
 * cluster::AbstractClusterView::OnCommChannelMessage
 */
void cluster::AbstractClusterView::OnCommChannelMessage(
    cluster::CommChannel& sender, const vislib::net::AbstractSimpleMessage& msg) {
    using vislib::sys::Log;
    vislib::net::SimpleMessage outMsg;

    switch (msg.GetHeader().GetMessageID()) {
    case cluster::netmessages::MSG_SHUTDOWN:
        this->GetCoreInstance()->Shutdown();
        break;

    case cluster::netmessages::MSG_PING_TIMESYNC:
        ASSERT(msg.GetHeader().GetBodySize() == sizeof(cluster::netmessages::TimeSyncData));
        outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_PING_TIMESYNC);
        outMsg.GetHeader().SetBodySize(sizeof(cluster::netmessages::TimeSyncData));
        outMsg.AssertBodySize();
        ::memcpy(outMsg.GetBody(), msg.GetBody(), sizeof(cluster::netmessages::TimeSyncData));
        outMsg.GetBodyAs<cluster::netmessages::TimeSyncData>()
            ->clntTimes[outMsg.GetBodyAs<cluster::netmessages::TimeSyncData>()->trip] =
            this->GetCoreInstance()->GetCoreInstanceTime();
        sender.SendMessage(outMsg);
        break;

    case cluster::netmessages::MSG_DONE_TIMESYNC: {
        if (this->setupState != SETUP_TIME) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                "Setup step order problem: TimeSync-Message received during setup step #%d\n",
                static_cast<int>(this->setupState));
            break; // setup step order screwed up
        }

        ASSERT(msg.GetHeader().GetBodySize() == sizeof(cluster::netmessages::TimeSyncData));
        const cluster::netmessages::TimeSyncData& dat = *msg.GetBodyAs<cluster::netmessages::TimeSyncData>();
        double offsets[cluster::netmessages::MAX_TIME_SYNC_PING];

        vislib::StringA msg("TimeSync Done:\n");
        vislib::StringA line;
        for (unsigned int i = 0; i < cluster::netmessages::MAX_TIME_SYNC_PING; i++) {
            line.Format("    %f (%f)=> %f (%f)\n", dat.srvrTimes[i], (dat.srvrTimes[i + 1] + dat.srvrTimes[i]) * 0.5,
                dat.clntTimes[i], ((dat.srvrTimes[i] + dat.srvrTimes[i + 1]) * 0.5) - dat.clntTimes[i]);
            msg += line;
        }
        line.Format("    %f\n", dat.srvrTimes[cluster::netmessages::MAX_TIME_SYNC_PING]);
        msg += line;
        for (unsigned int i = 0; i < cluster::netmessages::MAX_TIME_SYNC_PING; i++) {
            offsets[i] = ((dat.srvrTimes[i] + dat.srvrTimes[i + 1]) * 0.5) - dat.clntTimes[i];
            line.Format("Offset %d: %f\n", i, offsets[i]);
            msg += line;
        }
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 500, msg);

        double offset = 0.0;
        for (unsigned int i = 0; i < cluster::netmessages::MAX_TIME_SYNC_PING; i++) {
            offset += offsets[i];
        }
        offset /= static_cast<double>(cluster::netmessages::MAX_TIME_SYNC_PING);

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Time sync finished with offset %f", offset);
        this->GetCoreInstance()->OffsetInstanceTime(offset);

        this->continueSetup();

    } break;

    case cluster::netmessages::MSG_TIME_SANITYCHECK:
        outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_TIME_SANITYCHECK);
        outMsg.GetHeader().SetBodySize(sizeof(double));
        outMsg.AssertBodySize();
        *outMsg.GetBodyAs<double>() = this->GetCoreInstance()->GetCoreInstanceTime();
        sender.SendMessage(outMsg);
        break;

    case cluster::netmessages::MSG_WHATSYOURNAME: {
        vislib::StringA myname;
        vislib::sys::SystemInformation::ComputerName(myname);
        outMsg.GetHeader().SetMessageID(cluster::netmessages::MSG_MYNAMEIS);
        outMsg.GetHeader().SetBodySize(myname.Length() + 1);
        outMsg.AssertBodySize();
        ::memcpy(outMsg.GetBody(), myname.PeekBuffer(), myname.Length() + 1);
        sender.SendMessage(outMsg);
    } break;

    case cluster::netmessages::MSG_MYNAMEIS:
        ASSERT(msg.GetHeader().GetBodySize() > 0);
        sender.SetCounterpartName(msg.GetBodyAs<char>());
        break;

    case cluster::netmessages::MSG_REQUEST_RESETUP:
        this->continueSetup(SETUP_GRAPH); // restart setup process from graph setup
        break;

    case cluster::netmessages::MSG_GRAPHSETUP:
        if (this->setupState != SETUP_GRAPH) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                "Setup step order problem: Graph-Setup-Message received during setup step #%d\n",
                static_cast<int>(this->setupState));
            break; // setup step order screwed up
        }
        if (this->graphInitData == NULL) {
            {
                vislib::sys::AutoLock lock(this->ModuleGraphLock());
                this->disconnectOutgoingRenderCall(); // this will result in rendering the fallback view at least once
            }
            this->GetCoreInstance()->CleanupModuleGraph();

            this->graphInitData = new vislib::net::SimpleMessage(msg);

        } else {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Failed to setup module graph: still pending init data\n");
        }
        break;

    case cluster::netmessages::MSG_SET_CLUSTERVIEW: {
        factories::CallDescription::ptr desc =
            this->GetCoreInstance()->GetCallDescriptionManager().Find("CallRenderView");
        if (desc != NULL) {
            Call* c = this->GetCoreInstance()->InstantiateCall(
                this->FullName() + "::renderView", vislib::StringA(msg.GetBodyAs<char>()) + "::render", desc);
            if (c == NULL) {
                Log::DefaultLog.WriteMsg(
                    Log::LEVEL_ERROR, "Unable to connect cluster display to view %s\n", msg.GetBodyAs<char>());
            }
        } else {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Internal Error: \"CallRenderView\" is not registered\n");
        }

        view::AbstractView* view = this->getConnectedView();
        if (view != NULL) {
            view->RegisterHook(new InitCameraHookHandler(&sender));
        }

    } break;

    case cluster::netmessages::MSG_SET_CAMERAVALUES: {
        view::AbstractView* av = this->getConnectedView();
        if (av != NULL) {
            vislib::RawStorageSerialiser rss(static_cast<const BYTE*>(msg.GetBody()), msg.GetHeader().GetBodySize());
            av->DeserialiseCamera(rss);
        }
    } break;

    case cluster::netmessages::MSG_SET_PARAMVALUE: {
        vislib::StringA name(msg.GetBodyAs<char>());
        vislib::StringA value(msg.GetBodyAsAt<char>(name.Length() + 1));
        AbstractNamedObject::ptr_type p = this->FindNamedObject(name, true);
        param::ParamSlot* ps = dynamic_cast<param::ParamSlot*>(p.get());
        if (ps != NULL) {
            // printf("SET[%s]=%s\n", name.PeekBuffer(), value.PeekBuffer());
            ps->Parameter()->ParseValue(value);
        }
    } break;

    default:
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Unhandled message received: %u\n",
            static_cast<unsigned int>(msg.GetHeader().GetMessageID()));
        break;
    }
}


/*
 * cluster::AbstractClusterView::onServerAddressChanged
 */
bool cluster::AbstractClusterView::onServerAddressChanged(param::ParamSlot& slot) {
    ASSERT(&slot == &this->serverAddressSlot);
    vislib::StringA address(this->serverAddressSlot.Param<param::StringParam>()->Value());

    if (address.IsEmpty()) {
        try {
            if (this->ctrlChannel.IsOpen()) {
                this->ctrlChannel.Close();
            }
        } catch (...) {
        }
        return true;
    }

    vislib::net::IPEndPoint ep;
    float wildness = vislib::net::NetworkInformation::GuessRemoteEndPoint(ep, address);

    try {
        if (this->ctrlChannel.IsOpen()) {
            this->ctrlChannel.Close();
        }
    } catch (...) {
    }

    if (wildness > 0.8) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Guessed server end point \"%s\" from \"%s\" with too high wildness: %f\n", ep.ToStringA().PeekBuffer(),
            address.PeekBuffer(), wildness);
        this->serverAddressSlot.Param<param::StringParam>()->SetValue("", false);
        return true;
    }

    vislib::sys::Log::DefaultLog.WriteMsg(
        (wildness > 0.3) ? vislib::sys::Log::LEVEL_WARN : vislib::sys::Log::LEVEL_INFO,
        "Starting server on \"%s\" guessed from \"%s\" with wildness: %f\n", ep.ToStringA().PeekBuffer(),
        address.PeekBuffer(), wildness);

    vislib::SmartRef<vislib::net::TcpCommChannel> channel = vislib::net::TcpCommChannel::Create(
        vislib::net::TcpCommChannel::FLAG_NODELAY | vislib::net::TcpCommChannel::FLAG_REUSE_ADDRESS);

    try {
        channel->Connect(vislib::net::IPCommEndPoint::Create(ep));
        this->ctrlChannel.Open(channel.DynamicCast<vislib::net::AbstractCommClientChannel>());
    } catch (vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to connect to server: %s\n", ex.GetMsgA());
        this->serverAddressSlot.Param<param::StringParam>()->SetValue("", false);
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to connect to server: unexpected exception\n");
        this->serverAddressSlot.Param<param::StringParam>()->SetValue("", false);
    }

    return true;
}


/*
 * cluster::AbstractClusterView::continueSetup
 */
void cluster::AbstractClusterView::continueSetup(cluster::AbstractClusterView::SetupState state) {
    if (state == SETUP_UNKNOWN) {
        switch (this->setupState) {
        case SETUP_TIME:
            this->setupState = SETUP_GRAPH;
            break;
        case SETUP_GRAPH:
            this->setupState = SETUP_CAMERA;
            break;
        case SETUP_CAMERA:
            this->setupState = SETUP_COMPLETE;
            break;
        case SETUP_COMPLETE: /* no change */
            break;
        default:
            this->setupState = SETUP_UNKNOWN;
            break;
        }
    } else {
        this->setupState = state;
    }

    if (this->setupState == SETUP_COMPLETE) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Setup complete");
        return;
    }
    if (this->setupState == SETUP_UNKNOWN) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Setup in undefined stated. Restarting.");
        this->setupState = SETUP_TIME;
    }
    vislib::sys::Log::DefaultLog.WriteMsg(
        vislib::sys::Log::LEVEL_INFO + 50, "Entering setup state #%d\n", static_cast<int>(this->setupState));

    vislib::net::SimpleMessage msg;
    switch (this->setupState) {
    case SETUP_TIME:
        msg.GetHeader().SetMessageID(cluster::netmessages::MSG_REQUEST_TIMESYNC);
        msg.GetHeader().SetBodySize(0);
        msg.AssertBodySize();
        this->ctrlChannel.SendMessage(msg);
        break;
    case SETUP_GRAPH:
        msg.GetHeader().SetMessageID(cluster::netmessages::MSG_REQUEST_GRAPHSETUP);
        msg.GetHeader().SetBodySize(0);
        msg.AssertBodySize();
        this->ctrlChannel.SendMessage(msg);
        break;
    case SETUP_CAMERA:
        msg.GetHeader().SetMessageID(cluster::netmessages::MSG_REQUEST_CAMERASETUP);
        msg.GetHeader().SetBodySize(0);
        msg.AssertBodySize();
        this->ctrlChannel.SendMessage(msg);
        break;
    default:
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Setup state #%d not implemented\n", static_cast<int>(this->setupState));
        break;
    }
}
