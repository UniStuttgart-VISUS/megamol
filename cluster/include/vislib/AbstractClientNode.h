/*
 * AbstractClientNode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCLIENTNODE_H_INCLUDED
#define VISLIB_ABSTRACTCLIENTNODE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractClusterNode.h"
#include "vislib/CmdLineParser.h"
#include "vislib/DNS.h"
#include "vislib/Thread.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This class defines additional functionality that a cluster client node
     * must provide.
     *
     * AbstractClientNodes uses virtual inheritance to implement the "delegate
     * to sister" pattern.
     *
     * The Initialise() method parses the specified command line for the server 
     * IP address or name ("--server-name") and the server port 
     * ("--server-port"). If an implementing class does not call its parent
     * Initialise() method, it must set the server address manually using the
     * SetServerAddress() method.
     *
     * The Run() method connects to the server specified by the command line or
     * via SetServerAddress() and starts the message receiver thread for the
     * new connection. Afterwards, it returns.
     */
    class AbstractClientNode : public virtual AbstractClusterNode {

    public:

        /** Dtor. */
        virtual ~AbstractClientNode(void);

        /**
         * Answer the address of the server to connect to.
         *
         * @return The address of the server to connect to.
         */
        inline const IPEndPoint& GetServerAddress(void) const {
            return this->serverAddress;
        }

        /**
         * Initialise the node.
         *
         * Implementing subclasses should build an initialisation chain by 
         * calling their parent class implementation first.
         *
         * @param inOutCmdLine The command line passed to the containing
         *                     application. The method might alter the command
         *                     line and remove consumed options.
         *
         * @throws
         */
        virtual void Initialise(sys::CmdLineProviderA& inOutCmdLine);

        /**
         * Initialise the node.
         *
         * Implementing subclasses should build an initialisation chain by 
         * calling their parent class implementation first.
         *
         * @param inOutCmdLine The command line passed to the containing
         *                     application. The method might alter the command
         *                     line and remove consumed options.
         *
         * @throws
         */
        virtual void Initialise(sys::CmdLineProviderW& inOutCmdLine);

        /**
         * Connect to the server address specified before and start the message
         * receiver thread. Afterwards, the method returns.
         *
         * You should call this Run() method first in subclasses as your own 
         * operations will probably not return immediately.
         *
         * @return 0 in case of success.
         *
         * @throws IllegalStateException If the client node is already 
         *                               connected.
         * @throws SocketException If it was not possible to connect to the 
         *                         server.
         * @throws SystemException If the message receiver thread could not be
         *                         started.
         * @throws std::bad_alloc In case there is insufficient heap memory.
         */
        virtual DWORD Run(void);

        inline void SetReconnectAttempts(const UINT reconnectAttempts) {
            this->reconnectAttempts = reconnectAttempts;
        }

        /**
         * Set the address of the server to connect to. This must be done 
         * before the node connects to the server.
         *
         * @param serverAddress The new server added.
         */
        inline void SetServerAddress(const IPEndPoint& serverAddress) {
            this->serverAddress = serverAddress;
        }

    protected:

        /** Ctor. */
        AbstractClientNode(void);

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        AbstractClientNode(const AbstractClientNode& rhs);

        /**
         * Connect to the server node and start the receiver thread.
         *
         * This method works on the 'socket' and 'receiver' member and has the
         * side effect that 'reconnectAttempts' is decremented by one if it is 
         * not yet 0. The connect attempt is made in any case, even if 
         * 'cntReconnect' is 0.
         *
         * @param rmc If not NULL, the method uses the specified receive
         *            context for the connection.
         *
         * @throws SocketException If it was not possible to connect to the
         *                         server.
         * @throws SystemException If the message receiver thread could not be
         *                         started.
         * @throws std::bad_alloc In case there is insufficient heap memory.
         */
        void connect(PReceiveMessagesCtx rmc);

        /**
         * Disconnect from server.
         *
         * @param isSilent    If true, no exception will be thrown in case of an
         *                    error.
         * @param noReconnect If true, the 'reconnectAttempts' member will be 
         *                    set 0.
         *
         * @throws SocketException In case 'isSilent' is false and a socket 
         *                         error occurred.
         * @throws SystemException In case 'isSilent' is false and the receiver
         *                         thread could not be joined.
         */
        void disconnect(const bool isSilent, const bool noReconnect);

        /**
         * Answer the number of known peer nodes.
         *
         * @return The number of known peer nodes.
         */
        virtual SIZE_T countPeers(void) const;

        /**
         * Call 'func' for each known peer node (socket).
         *
         * On server nodes, the function is usually called for all the client
         * nodes, on client nodes only once (for the server). However, 
         * implementations in subclasses may differ.
         *
         * @param func    The function to be executed for each peer node.
         * @param context This is an additional pointer that is passed to 
         *                'func'.
         *
         * @return The number of sucessful calls to 'func' that have been made.
         *         This is at most 1 for a client node.
         */
        virtual SIZE_T forEachPeer(ForeachPeerFunc func, void *context);

        /**
         * Call 'func' for the peer node that has the specified ID 'peerId'. If
         * such a peer node is not known, nothing should be done.
         *
         * On server nodes, the function should check for a client with the
         * specified ID; on client nodes the implementation should check whether
         * 'peerId' references the server node.
         *
         * @param peerId  The identifier of the node to run 'func' for.
         * @param func    The function to be execured for the specified peer 
         *                node.
         * @param context This is an additional pointer that is passed to 
         *                'func'.
         *
         * @return true if 'func' was executed, false otherwise.
         */
        virtual bool forPeer(const PeerIdentifier& peerId, ForeachPeerFunc func,
            void *context);

        /**
         * The message receiver thread calls this method once it exists.
         *
         * The method tries to reconnect to the server as long as the 
         * 'cntReconnect' member is not 0. Otherwise, the context is released.
         *
         * @param socket The socket that was used from communication with the
         *               peer node.
         * @param rmc    The receive context that was used by the receiver 
         *               thread. The method takes ownership of the context and
         *               should release if not needed any more.
         */
        virtual void onMessageReceiverExiting(Socket& socket,
            PReceiveMessagesCtx rmc);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        AbstractClientNode& operator =(const AbstractClientNode& rhs);

    private:

        /** Superclass typedef. */
        typedef AbstractClusterNode Super;

        /**
         * Character type agnostic initialiser that does the actual work.
         *
         * @param inOutCmdLine The command line passed to the containing
         *                     application. The method might alter the command
         *                     line and remove consumed options.
         *
         * @throws
         */
        template<class T> void initialise(
            sys::CmdLineProvider<T>& inOutCmdLine);

        /** The number of reconnect attempts to make if disconnected. */
        UINT reconnectAttempts;

        /** The receiver thread that generates the message events. */
        sys::Thread receiver;

        /** The address of the server node to connect to. */
        IPEndPoint serverAddress;

        /** The socket for communicating with the server. */
        Socket socket;

    };


    /*
     * vislib::net::cluster::AbstractClientNode::initialise
     */
    template<class T>
    void AbstractClientNode::initialise(sys::CmdLineProvider<T>& inOutCmdLine) {
        typedef vislib::String<T> String;
        typedef vislib::sys::CmdLineParser<T> Parser;
        typedef typename Parser::Argument Argument;
        typedef typename Parser::Option Option;
        typedef typename Option::ValueDesc ValueDesc;

        Parser parser;
        Argument *arg = NULL;

        Option optServer(String(_T("server-node")), 
            String(_T("Specifies the name of the server node.")),
            Option::FLAG_UNIQUE, 
            ValueDesc::ValueList(Option::STRING_VALUE, 
                String(_T("host")), 
                String(_T("The host name or IP address of the server node."))));
        parser.AddOption(&optServer);

        Option optPort(String(_T("server-port")), 
            String(_T("Specifies the post of the server node.")),
            Option::FLAG_UNIQUE, 
            ValueDesc::ValueList(Option::INT_VALUE, 
                String(_T("port")), 
                String(_T("The port the server node is listening on."))));
        parser.AddOption(&optPort);

        if (parser.Parse(inOutCmdLine.ArgC(), inOutCmdLine.ArgV()) >= 0) {
            if ((arg = optServer.GetFirstOccurrence()) != NULL) {
                // TODO: IPv6
                this->serverAddress.SetIPAddress(IPAddress::Create(
                    StringA(arg->GetValueString())));
            } else {
                // TODO: IPv6
                this->serverAddress.SetIPAddress(IPAddress::LOCALHOST);
            }

            if ((arg = optPort.GetFirstOccurrence()) != NULL) {
                this->serverAddress.SetPort(arg->GetValueInt());
            } else {
                this->serverAddress.SetPort(DEFAULT_PORT);
            }
        }
    }

} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCLIENTNODE_H_INCLUDED */
