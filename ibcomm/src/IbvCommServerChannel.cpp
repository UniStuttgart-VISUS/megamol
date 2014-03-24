///*
// * IbvCommServerChannel.cpp
// *
// * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
// * Alle Rechte vorbehalten.
// */
//
//#include "vislib/IbvCommServerChannel.h"
//
//#include "vislib/IPCommEndPoint.h"
//#include "the/trace.h"
//#include "the/not_supported_exception.h"
//
//
//vislib::SmartRef<vislib::net::AbstractCommChannel> 
//vislib::net::ib::IbvCommServerChannel::Accept(void) {
//    VLSTACKTRACE("IbvCommServerChannel::Bind", __FILE__, __LINE__);
//
//    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Waiting for IB connection "
//        "request...\n");
//    //this->evtAccept.Wait();
//
//    struct rdma_cm_id *dude;
//
//    int retval = ::rdma_get_request(this->id, &dude);
//
//
//
//    WV_CONNECT_ATTRIBUTES attribs;
//    dude->ep.connect->Query(&attribs);
//    //attribs.Device
//
//    //this->id->ep.connect->Accept();
//    throw 1;
//}
//
//
//void vislib::net::ib::IbvCommServerChannel::Bind(
//        SmartRef<AbstractCommEndPoint> endPoint) {
//    VLSTACKTRACE("IbvCommServerChannel::Bind", __FILE__, __LINE__);
//
//    IPCommEndPoint *cep = endPoint.DynamicPeek<IPCommEndPoint>();
//    IPEndPoint& ep = static_cast<IPEndPoint&>(*cep);
//
//    //THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Binding IB end point to %s...\n",
//    //    ep.ToStringA().PeekBuffer());
//    //int x = ::rdma_bind_addr(this->id, static_cast<sockaddr *>(ep));
//    //int y = ::GetLastError();
//
//    struct ibv_qp_init_attr attr;
//    struct rdma_addrinfo hints, *addrInfo;
//    int retval = 0;
//
//    ::ZeroMemory(&hints, sizeof(hints));
//    hints.ai_flags = RAI_PASSIVE;
//    hints.ai_port_space = RDMA_PS_TCP;
//    StringA str;
//    str.Format("%d", ep.GetPort());
//    retval = ::rdma_getaddrinfo(NULL, (char *) str.PeekBuffer(), &hints, &addrInfo);
//
//
//    ::ZeroMemory(&attr, sizeof(attr));
//    attr.cap.max_recv_wr = 1;
//    attr.cap.max_send_wr = 1;
//    attr.cap.max_recv_sge = 1;
//    attr.cap.max_send_sge = 1;
//    attr.sq_sig_all = 1;
//
//    retval = ::rdma_create_ep(&this->id, addrInfo, NULL, &attr);
//}
//
//
//void vislib::net::ib::IbvCommServerChannel::Close(void) {
//}
//
//
//void vislib::net::ib::IbvCommServerChannel::Connect(
//        SmartRef<AbstractCommEndPoint> endPoint) {
//    throw the::not_supported_exception("IbvCommServerChannel::Connect", 
//        __FILE__, __LINE__);
//}
//
//
//vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
//vislib::net::ib::IbvCommServerChannel::GetLocalEndPoint(void) const {
//    throw 1;
//}
//
//
//vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
//vislib::net::ib::IbvCommServerChannel::GetRemoteEndPoint(void) const {
//    throw the::not_supported_exception("IbvCommServerChannel::"
//        "GetRemoteEndPoint", __FILE__, __LINE__);
//}
//
//
//void vislib::net::ib::IbvCommServerChannel::Listen(const int backlog) {
//    ::rdma_listen(this->id, backlog);
//    //this->evtAccept.Reset();
//    //this->msgThread.Start(this);
//}
//
//
//size_t vislib::net::ib::IbvCommServerChannel::Receive(void *outData, 
//        const size_t cntBytes, const unsigned int timeout, const bool forceReceive) {
//    throw the::not_supported_exception("IbvCommServerChannel::Receive", 
//        __FILE__, __LINE__);
//}
//
//
//size_t vislib::net::ib::IbvCommServerChannel::Send(const void *data, 
//        const size_t cntBytes, const unsigned int timeout, const bool forceSend) {
//    throw the::not_supported_exception("IbvCommServerChannel::Send", __FILE__, 
//        __LINE__);
//}
//
//
///*
// * vislib::net::ib::IbvCommServerChannel::messagePump
// */
//unsigned int vislib::net::ib::IbvCommServerChannel::messagePump(void *userData) {
//    VLSTACKTRACE("IbvCommServerChannel::messagePump", __FILE__, __LINE__);
//    IbvCommServerChannel *channel = static_cast<IbvCommServerChannel *>(
//        userData);
//    int retval = 0;
//
//    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Entering RDMA CM message "
//        "loop...\n");
//    while ((retval = ::rdma_get_cm_event(channel->channel, 
//            &channel->evt)) == 0) {
//        struct rdma_cm_event e;
//        ::memcpy(&e, channel->evt, sizeof(e));
//        ::rdma_ack_cm_event(channel->evt);
//
//        switch (e.event) {
//            case RDMA_CM_EVENT_CONNECT_REQUEST:
//                THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Received "
//                    "RDMA_CM_EVENT_CONNECT_REQUEST.\n");
//                retval = channel->onConnectRequest(e);
//                break;
//
//            default:
//                break;
//        }
//
//        if (retval != 0) {
//            break;
//        }
//    }
//
//    return retval;
//}
//
//
///*
// * vislib::net::ib::IbvCommServerChannel::IbvCommServerChannel
// */
//vislib::net::ib::IbvCommServerChannel::IbvCommServerChannel(void) 
//        : channel(NULL), evt(NULL), id(NULL), msgThread(messagePump) {
//    VLSTACKTRACE("IbvCommServerChannel::IbvCommServerChannel", __FILE__, 
//        __LINE__);
//    // TODO: Implement
//
//    this->channel = ::rdma_create_event_channel();
//
//    //::rdma_create_id(this->channel, &this->id, this, RDMA_PS_TCP);
//
//
//
//}
//
//
///*
// * vislib::net::ib::IbvCommServerChannel::~IbvCommServerChannel
// */
//vislib::net::ib::IbvCommServerChannel::~IbvCommServerChannel(void) {
//    VLSTACKTRACE("IbvCommServerChannel::~IbvCommServerChannel", __FILE__, 
//        __LINE__);
//
//    if (this->id != NULL) {
//        ::rdma_destroy_id(this->id);
//        this->id = NULL;
//    }
//
//    if (this->channel != NULL) {
//        ::rdma_destroy_event_channel(this->channel);
//        this->channel = NULL;
//    }
//}
//
//
///*
// * vislib::net::ib::IbvCommServerChannel::onConnectRequest
// */
//int vislib::net::ib::IbvCommServerChannel::onConnectRequest(
//        struct rdma_cm_event& evt) {
//    VLSTACKTRACE("IbvCommServerChannel::onConnectRequest", __FILE__, __LINE__);
//
//
//    this->evtAccept.Set();
//
//    return 0;
//}
