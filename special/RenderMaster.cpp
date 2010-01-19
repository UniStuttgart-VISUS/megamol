/*
 * RenderMaster.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "RenderMaster.h"
#include <climits>
#include "param/BoolParam.h"
#include "param/ButtonParam.h"
#include "param/IntParam.h"
#include "param/StringParam.h"
#include "special/RenderNetUtil.h"
#include "Call.h"
#include "CallDescription.h"
#include "CallDescriptionManager.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "CoreInstance.h"
#include "Module.h"
#include "ModuleDescription.h"
#include "ModuleDescriptionManager.h"
#include "vislib/Array.h"
#include "vislib/assert.h"
//#include "vislib/AutoLock.h"
//#include "vislib/Exception.h"
#include "vislib/IPEndPoint.h"
#include "vislib/Log.h"
#include "vislib/Map.h"
#include "vislib/NetworkInformation.h"
#include "vislib/RawStorageWriter.h"
#include "vislib/RunnableThread.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/Thread.h"
#include "vislib/UTF8Encoder.h"
//#include "vislib/Trace.h"

using namespace megamol::core;

/****************************************************************************/


/*
 * special::RenderMaster::Connection::Connection
 */
special::RenderMaster::Connection::Connection(void) : name(), owner(NULL),
        receiver(NULL), socket() {
    // intentionally empty
}


/*
 * special::RenderMaster::Connection::Connection
 */
special::RenderMaster::Connection::Connection(
        const special::RenderMaster::Connection& src) : name(src.name),
        owner(src.owner), receiver(src.receiver), socket(src.socket) {
    // intentionally empty
}


/*
 * special::RenderMaster::Connection::Connection
 */
special::RenderMaster::Connection::Connection(special::RenderMaster *owner,
        const vislib::StringA& name, const vislib::net::Socket& socket)
        : name(name), owner(owner), receiver(), socket(socket) {

    this->receiver = new vislib::sys::Thread(
        &special::RenderMaster::Connection::receive);
    this->receiver->Start(static_cast<void*>(new Connection(*this)));
}


/*
 * special::RenderMaster::Connection::~Connection
 */
special::RenderMaster::Connection::~Connection(void) {
    this->owner = NULL; // DO NOT DELETE
    this->receiver = NULL; // DO NOT DELETE It will delete itself
    // do not close this->socket. It might still be in use.
}


/*
 * special::RenderMaster::Connection::Close
 */
void special::RenderMaster::Connection::Close(bool join) {
    using vislib::sys::Log;
    if (this->socket.IsValid()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 250,
            "Closing socket to %s", this->name.PeekBuffer());
        try {
            this->socket.Shutdown();
        } catch(...) {
        }
        try {
            this->socket.Close();
        } catch(...) {
        }
    }

    ASSERT(this->owner != NULL);
    this->owner->connLock.Lock();
    if (this->owner->connections.Contains(*this)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "Connection closed to %s", this->name.PeekBuffer());
        this->owner->connections.RemoveAll(*this);
    }
    this->owner->connLock.Unlock();

    if (join && (this->receiver != NULL)) {
        this->receiver->Join();
    }
}


/*
 * special::RenderMaster::Connection::Send
 */
void special::RenderMaster::Connection::Send(
        const special::RenderNetMsg& msg) {
    RenderNetUtil::SendMessage(this->socket, msg);
}


/*
 * special::RenderMaster::Connection::operator=
 */
special::RenderMaster::Connection&
special::RenderMaster::Connection::operator=(
        const special::RenderMaster::Connection& rhs) {
    this->name = rhs.name;
    this->owner = rhs.owner;
    this->receiver = rhs.receiver;
    this->socket = rhs.socket;
    return *this;
}


/*
 * special::RenderMaster::Connection::operator ==
 */
bool special::RenderMaster::Connection::operator ==(
        const special::RenderMaster::Connection &rhs) const {
    return this->name.Equals(rhs.name)
        && (this->owner == rhs.owner)
        && (this->receiver == rhs.receiver);
    // sockets are not compared because some handles might be invalid, because
    // other have already be closed.
}


/*
 * special::RenderMaster::Connection::receive
 */
DWORD special::RenderMaster::Connection::receive(void *userData) {
    special::RenderMaster::Connection *This
        = static_cast<special::RenderMaster::Connection*>(userData);
    ASSERT(This->owner != NULL);
    RenderNetMsg msg;

    try {

        while (true) {
            RenderNetUtil::ReceiveMessage(This->socket, msg);

            if (!This->owner->HandleMessage(*This, msg)) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                    "Unhandled Message (%u; %u; %u)",
                    static_cast<unsigned int>(msg.GetType()),
                    static_cast<unsigned int>(msg.GetID()),
                    static_cast<unsigned int>(msg.GetDataSize()));
            }
        }

    } catch(...) {
    }
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Leaving receiver loop");

    This->Close(false);
    delete This;
    return 0;
}

/****************************************************************************/


/*
 * special::RenderMaster::RenderMaster
 */
special::RenderMaster::RenderMaster() : job::AbstractJobThread(), Module(),
        serverAdapSlot("serverAdap", "The network adapter to run the server on"),
        serverPortSlot("serverPort", "The network port to run the server on"),
        serverUpSlot("serverUp", "Flag indicating if the server is running or not"),
        masterViewNameSlot("masterView", "The name of the view instance to be used as master for the rendering network"),
        connLock(), masterView(NULL) {

    this->serverAdapSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->serverAdapSlot);

    // Note: Port minval should be 49152, which is the lowest port for the
    //  private dynamic ip port range. However, we do not want to enforce
    //  using only these ports.
    this->serverPortSlot << new param::IntParam(
        RenderNetUtil::DefaultPort, 1, USHRT_MAX);
    this->MakeSlotAvailable(&this->serverPortSlot);

    this->serverUpSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->serverUpSlot);

    this->masterViewNameSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->masterViewNameSlot);

    // TODO: Implement

}


/*
 * special::RenderMaster::~RenderMaster
 */
special::RenderMaster::~RenderMaster() {
    Module::Release();
    ASSERT(this->connections.IsEmpty());

    // TODO: Implement

}


/*
 * special::RenderMaster::OnNewConnection
 */
bool special::RenderMaster::OnNewConnection(vislib::net::Socket& socket,
        const vislib::net::IPEndPoint& addr) throw() {
    using vislib::sys::Log;
    try {
        socket.SetNoDelay(true);

        RenderNetUtil::HandshakeAsServer(socket);
        vislib::StringA clientName = RenderNetUtil::WhoAreYou(socket);

        Connection conn(this, clientName, socket);

        this->connLock.Lock();
        this->connections.Add(conn);
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "Connection established with %s (%s)\n", clientName.PeekBuffer(),
            addr.ToStringA().PeekBuffer());
        this->connLock.Unlock();

        return true;

    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Connection refused from %s: %s\n", addr.ToStringA().PeekBuffer(),
            e.GetMsgA());

    } catch(...) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Connection refused from %s: Unknown error.\n",
            addr.ToStringA().PeekBuffer());

    }

    return false;
}


/*
 * special::RenderMaster::create
 */
bool special::RenderMaster::create(void) {
    vislib::net::Socket::Startup();

    // TODO: Implement

    return true;
}


/*
 * special::RenderMaster::release
 */
void special::RenderMaster::release(void) {

    // TODO: Implement

    this->closeAllConnections();
    ASSERT(this->connections.IsEmpty());
    vislib::net::Socket::Cleanup();
}


/*
 * special::RenderMaster::Run
 */
DWORD special::RenderMaster::Run(void *userData) {
    using vislib::sys::Log;
    vislib::sys::RunnableThread<vislib::net::TcpServer> server;
    vislib::net::IPEndPoint srvEndPoint;

    server.AddListener(this);
    this->serverUpSlot.ForceSetDirty(); // to trigger init

    while (!this->shouldTerminate()) {

        if (this->masterViewNameSlot.IsDirty()) {
            this->masterViewNameSlot.ResetDirty();
            vislib::StringA mvn(this->masterViewNameSlot.Param<param::StringParam>()->Value());

            this->LockModuleGraph(false);
            AbstractNamedObjectContainer *anoc = dynamic_cast<AbstractNamedObjectContainer*>(this->RootModule());
            AbstractNamedObject *ano = anoc->FindChild(mvn);
            ViewInstance *vi = dynamic_cast<ViewInstance *>(ano);
            if (this->masterView != vi) {
                this->masterView = vi;

                this->connLock.Lock();
                vislib::RawStorage data;
                this->makeGraphSetupData(data);
                RenderNetMsg m(RenderNetUtil::MSGTYPE_SETUP_MODGRAPH, 0, data.GetSize(), data);
                vislib::SingleLinkedList<Connection>::Iterator i = this->connections.GetIterator();
                while (i.HasNext()) {
                    i.Next().Send(m);
                }
                this->connLock.Unlock();

            }
            this->UnlockModuleGraph();
        }

        if (this->serverAdapSlot.IsDirty() || this->serverPortSlot.IsDirty()
                || this->serverUpSlot.IsDirty()) {
            this->serverAdapSlot.ResetDirty();
            this->serverPortSlot.ResetDirty();
            this->serverUpSlot.ResetDirty();

            if (server.IsRunning()) {
                server.Terminate(false);
            }
            // note: old connections remain established!

            vislib::TString addr;
            addr.Format(_T("%s:%d"),
                this->serverAdapSlot.Param<param::StringParam>()->Value().PeekBuffer(),
                this->serverPortSlot.Param<param::IntParam>()->Value());

            float wildness = vislib::net::NetworkInformation::GuessLocalEndPoint(srvEndPoint, addr.PeekBuffer());
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "Preparing RenderMaster serve on %s (from %s; wildness %.2f)",
                srvEndPoint.ToStringA().PeekBuffer(),
                vislib::StringA(addr).PeekBuffer(), wildness);

            server.SetFlags(vislib::net::TcpServer::FLAGS_REUSE_ADDRESS);

            if (this->serverUpSlot.Param<param::BoolParam>()->Value()) {
                try {
                    if (!server.Start(&srvEndPoint)) {
                        throw vislib::Exception("server.Start returned false",
                            __FILE__, __LINE__);
                    }

                    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                        "RenderMaster server started\n");

                } catch (vislib::Exception e) {
                    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "Unable to start RenderMaster server on %s: %s\n",
                        srvEndPoint.ToStringA().PeekBuffer(),
                        e.GetMsgA());
                    this->serverUpSlot.Param<param::BoolParam>()->SetValue(false);

                } catch (...) {
                    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "Unable to start RenderMaster server on %s\n",
                        srvEndPoint.ToStringA().PeekBuffer());
                    this->serverUpSlot.Param<param::BoolParam>()->SetValue(false);

                }
            }
        }

        // run management code only once per second
        vislib::sys::Thread::Sleep(1000);
    }

    if (server.IsRunning()) {
        server.Terminate(false);
    }
    this->closeAllConnections();

    return 0;
}


/*
 * special::RenderMaster::closeAllConnections
 */
void special::RenderMaster::closeAllConnections(void) {
    this->connLock.Lock();
    vislib::Array<Connection> cons(this->connections.Count());
    vislib::SingleLinkedList<Connection>::Iterator i = this->connections.GetIterator();
    while (i.HasNext()) {
        cons.Append(i.Next());
    }
    this->connLock.Unlock();

    for (SIZE_T j = 0; j < cons.Count(); j++) {
        cons[j].Close(true);
        // cons[j] is now invalid!
    }
}


/*
 * special::RenderMaster::HandleMessage
 */
bool special::RenderMaster::HandleMessage(
        special::RenderMaster::Connection& con, special::RenderNetMsg& msg) {
    using vislib::sys::Log;

    switch (msg.GetType()) {
        case RenderNetUtil::MSGTYPE_REQUEST_TIMESYNC: {
            special::RenderNetMsg msg(RenderNetUtil::MSGTYPE_TIMESYNC,
                msg.GetID(), sizeof(RenderNetUtil::TimeSyncData));
            RenderNetUtil::TimeSyncData *tsd = msg.DataAs<RenderNetUtil::TimeSyncData>();
            for (unsigned int i = 1; i < RenderNetUtil::TimeSyncTripCnt; i++) {
                tsd->srvrTimes[i] = 0.0;
            }
            tsd->trip = 1;
            tsd->srvrTimes[0] = this->GetCoreInstance()->GetInstanceTime();

            con.Send(msg);
            return true;
        }
        case RenderNetUtil::MSGTYPE_TIMESYNC: {
            if (msg.GetDataSize() != sizeof(RenderNetUtil::TimeSyncData)) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                    "Message data size mismatch. Ignoring message %u.\n", msg.GetID());
                return false;
            }

            RenderNetUtil::TimeSyncData *tsd = msg.DataAs<RenderNetUtil::TimeSyncData>();
            if (tsd->trip < RenderNetUtil::TimeSyncTripCnt) {
                tsd->srvrTimes[tsd->trip++] = this->GetCoreInstance()->GetInstanceTime();
                
                con.Send(msg);

            } else {
                Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                    "Corrupt time sync message. Ignoring message %u.\n", msg.GetID());
                return false;
            }
            return true;
        }
        case RenderNetUtil::MSGTYPE_REQUEST_MODGRAPHSYNC: {
            this->LockModuleGraph(false);

            if (this->masterView != NULL) {
                // check if view is still correct
                bool found = false;
                AbstractNamedObjectContainer *anoc = dynamic_cast<AbstractNamedObjectContainer*>(this->RootModule());
                AbstractNamedObjectContainer::ChildList::Iterator iter = anoc->GetChildIterator();
                while (iter.HasNext()) {
                    if (iter.Next() == this->masterView) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    this->masterView = NULL;
                }
            }

            vislib::RawStorage data;
            this->makeGraphSetupData(data);
            RenderNetMsg m(RenderNetUtil::MSGTYPE_SETUP_MODGRAPH, msg.GetID(), data.GetSize(), data);
            con.Send(m);

            this->UnlockModuleGraph();
            return true;
        }
    }

    // TODO: Implement

    return false;
}


/*
 * special::RenderMaster::makeGraphSetupData
 */
void special::RenderMaster::makeGraphSetupData(vislib::RawStorage& outBuf) {
    if (this->masterView == NULL) {
        outBuf.EnforceSize(0);
        return;
    }


    vislib::SingleLinkedList<Module *> mods;
    vislib::SingleLinkedList<Call *> calls;
    vislib::SingleLinkedList<Call *>::Iterator itercalls;
    vislib::SingleLinkedList<Call *> allcalls; // collect all calls in the graph
    vislib::Stack<AbstractNamedObject *> anoStack;
    vislib::Stack<Module *> stack;
    AbstractNamedObject *ano;
    AbstractNamedObjectContainer *anoc;
    AbstractNamedObjectContainer::ChildList::Iterator anocIter;

    // #1 collect all calls
    anoStack.Push(this->RootModule());
    while (!anoStack.IsEmpty()) {
        ano = anoStack.Pop();
        anoc = dynamic_cast<AbstractNamedObjectContainer*>(ano);
        if (anoc == NULL) continue;
        anocIter = anoc->GetChildIterator();
        while (anocIter.HasNext()) {
            anoStack.Push(anocIter.Next());
        }

        CallerSlot *crs = dynamic_cast<CallerSlot *>(ano);
        if (crs == NULL) continue;

        Call *c = crs->CallAs<Call>();
        if (c != NULL) {
            allcalls.Add(c);
        }
    }


    // #2 collect relevant modules
    Module *mod = dynamic_cast<Module *>(this->masterView->View());
    if (mod != NULL) {
        stack.Push(mod);
    }

    while(!stack.IsEmpty()) {
        mod = stack.Pop();
        if (mod == NULL) continue;
        if (mods.Contains(mod)) continue;
        mods.Add(mod);

        anocIter = mod->GetChildIterator();
        while (anocIter.HasNext()) {
            ano = anocIter.Next();
            CallerSlot *crs = dynamic_cast<CallerSlot *>(ano);
            CalleeSlot *ces = dynamic_cast<CalleeSlot *>(ano);

            if (crs != NULL) {
                Call *c = crs->CallAs<Call>();
                if (c != NULL) {
                    mod = const_cast<Module*>(dynamic_cast<const Module*>(c->PeekCalleeSlot()->Parent()));
                    stack.Push(mod);
                    if (!calls.Contains(c)) {
                        calls.Add(c);
                    }
                }
            } else if (ces != NULL) {
                itercalls = allcalls.GetIterator();
                while (itercalls.HasNext()) {
                    Call *c = itercalls.Next();
                    if (c->PeekCalleeSlot() != ces) continue;
                    mod = const_cast<Module*>(dynamic_cast<const Module*>(c->PeekCallerSlot()->Parent()));
                    stack.Push(mod);
                    if (!calls.Contains(c)) {
                        calls.Add(c);
                    }
                }
            }
        }
    }


    // #3 construct output
    vislib::RawStorageWriter writer(outBuf, 0, sizeof(UINT32));
    UINT32 cnt = static_cast<UINT32>(mods.Count());
    writer << dynamic_cast<Module*>(this->masterView->View())->FullName() << cnt;
    vislib::SingleLinkedList<Module *>::Iterator moditer = mods.GetIterator();
    UINT32 cnter = 0;
    vislib::Map<const Module*, UINT32> modids;
    while (moditer.HasNext()) {
        Module *m = moditer.Next();
        ModuleDescription *md = NULL;
        ModuleDescriptionManager::DescriptionIterator mi = ModuleDescriptionManager::Instance()->GetIterator();
        while (mi.HasNext()) {
            ModuleDescription *d = mi.Next();
            if (d->IsDescribing(m)) {
                md = d;
                break;
            }
        }
        ASSERT(m != NULL);
        ASSERT(md != NULL);
        //printf("  Module: %s (%s)\n", md->ClassName().PeekBuffer(), m->FullName().PeekBuffer());
        writer << md->ClassName() << m->FullName();

        SIZE_T prePos = writer.Position();
        cnt = 0;
        writer << cnt; // dummy
        anocIter = m->GetChildIterator();
        while (anocIter.HasNext()) {
            ano = anocIter.Next();
            param::ParamSlot *ps = dynamic_cast<param::ParamSlot*>(ano);
            if ((ps == NULL) || (ps->Parameter().IsNull())) continue;
            if (ps->Param<param::ButtonParam>() != NULL) continue; // omit buttons
            vislib::StringA value;
            vislib::UTF8Encoder::Encode(value, ps->Parameter()->ValueString());
            writer << ps->Name() << value;
            cnt++;
        }
        SIZE_T postPos = writer.Position();
        writer.SetPosition(prePos);
        writer << cnt; // real count value
        writer.SetPosition(postPos);

        modids[m] = cnter++;
    }
    cnt = static_cast<UINT32>(calls.Count());
    writer << cnt;
    itercalls = calls.GetIterator();
    while (itercalls.HasNext()) {
        Call *c = itercalls.Next();
        CallDescription *cd = NULL;
        CallDescriptionManager::DescriptionIterator ci = CallDescriptionManager::Instance()->GetIterator();
        while (ci.HasNext()) {
            CallDescription *d = ci.Next();
            if (d->IsDescribing(c)) {
                cd = d;
                break;
            }
        }
        const Module *m1 = dynamic_cast<const Module *>(c->PeekCallerSlot()->Parent());
        const Module *m2 = dynamic_cast<const Module *>(c->PeekCalleeSlot()->Parent());
        ASSERT(c != NULL);
        ASSERT(cd != NULL);
        ASSERT(m1 != NULL);
        ASSERT(m2 != NULL);
        //printf("  Call: %s (%s::%s => %s::%s)\n", cd->ClassName(),
        //    m1->FullName().PeekBuffer(), c->PeekCallerSlot()->Name().PeekBuffer(),
        //    m2->FullName().PeekBuffer(), c->PeekCalleeSlot()->Name().PeekBuffer());
        writer << cd->ClassName() << modids[m1] << c->PeekCallerSlot()->Name()
            << modids[m2] << c->PeekCalleeSlot()->Name();
    }


    // #4 truncate buffer to all the real data
    outBuf.EnforceSize(writer.End(), true);

}
