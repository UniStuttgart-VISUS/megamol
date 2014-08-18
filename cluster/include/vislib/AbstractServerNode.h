/*
 * AbstractServerNode.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTSERVERNODE_H_INCLUDED
#define VISLIB_ABSTRACTSERVERNODE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractClusterNode.h"
#include "vislib/CmdLineParser.h"
#include "vislib/PtrArray.h"
#include "vislib/RunnableThread.h"
#include "vislib/TcpServer.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * This class defines the interface of a server node and implements the
     * basic communication facilities using TCP/IP sockets.
     *
     * This class uses virtual inheritance to implement the "delegate to 
     * sister" pattern.
     *
     * The Initialise() method of AbstractServerNode parses the given command
     * line for the adapter to bind the server to ("--bind-address") and the
     * port to bind the server to ("--bind-port"). If a subclass does not call
     * the parent Initialise(), it can set the bind address using the
     * SetBindAddress() method. If no bind address is set, the server will 
     * bind to an arbitrary network adapter (IPEndPoint::ANY) and use the
     * DEFAULT_PORT.
     *
     * The Run() method of AbstractServerNode starts the TCP server on a
     * separate thread and returns immediately.
     */
    class AbstractServerNode : public virtual AbstractClusterNode,
            public TcpServer::Listener {

    public:

        /** Dtor. */
        virtual ~AbstractServerNode(void);

        /**
         * Answer the socket address the server is binding to.
         *
         * @return The address the server is binding to.
         */
        virtual const IPEndPoint& GetBindAddress(void) const;

        /**
         * Initialise the node.
         *
         * Subclasses should build an initialisation chain by calling their 
         * parent class implementation first, or must provide all 
         * initialisation information that this implementation draws from the
         * command line by other means.
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
         * Subclasses should build an initialisation chain by calling their 
         * parent class implementation first, or must provide all 
         * initialisation information that this implementation draws from the
         * command line by other means.
         *
         * @param inOutCmdLine The command line passed to the containing
         *                     application. The method might alter the command
         *                     line and remove consumed options.
         *
         * @throws
         */
        virtual void Initialise(sys::CmdLineProviderW& inOutCmdLine);

        /**
         * This is the callback method that the TCP server uses to add a new 
         * node. It adds the peer node identified by 'addr' and using the socket
         * 'socket' to the client list and starts a message receiver thread for 
         * it.
         *
         * @aram socket The socket used for communication with the client.
         * @param addr  The address of the peer node.
         */
        virtual bool OnNewConnection(Socket& socket,
            const IPEndPoint& addr) throw();

        /**
         * This method is used by the TCP server to notify the object that no
         * further connections will be accepted.
         */
        virtual void OnServerStopped(void) throw();

        /**
         * Starts the server thread and returns afterwards.
         *
         * You should call this Run() method first in subclasses as your own 
         * operations will probably not return immediately.
         *
         * @return 0 in case of success, an error code otherwise.
         *
         * @throws SystemException If the server thread could not be started.
         */
        virtual DWORD Run(void);

        /**
         * Set a new socket address the server should bind to. 
         * 
         * This has only an effect if the server is not yet running.
         *
         * @param bindAddress The address to bind to.
         */
        virtual void SetBindAddress(const IPEndPoint& bindAddress);

        /**
         * Make the server bind to any adapter, but use the specified port.
         *
         * This has only an effect if the server is not yet running.
         *
         * @param port The port to bind to.
         */
        virtual void SetBindAddress(const unsigned short port);

    protected:

        /** Ctor. */
        AbstractServerNode(void);

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        AbstractServerNode(const AbstractServerNode& rhs);

        /**
         * Answer the number of known peer nodes.
         *
         * @return The number of known peer nodes.
         */
        virtual SIZE_T countPeers(void) const;

        /**
         * Remove the 'idx'th peer node. 
         *
         * This operation includes closing the socket and stopping the receiver
         * thread.
         *
         * It is safe to pass an invalid index, i.e. to disconnect from a 
         * non-connected node. All exceptions regarding communication and thread
         * errors will be caught.
         *
         * @param idx The index of the peer node to be removed.
         */
        void disconnectPeer(const SIZE_T idx);

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
        AbstractServerNode& operator =(const AbstractServerNode& rhs);

    private:

        /**
         * This structure represents a client peer node with its socket for
         * communication and the receiver thread that waits for incoming 
         * messages on this socket.
         */
        typedef struct PeerNode_t {
            vislib::net::Socket Socket;
            sys::Thread *Receiver;
        } PeerNode;

        /**
         * Answer the node with the specified identifier.
         *
         * @param peerId Identifier of the node to search.
         *
         * @return The index if the peer node with the specified identifier.
         *
         * @throws NoSuchElementException If no peer with the specified ID is
         *                                known.
         */
        SIZE_T findPeerNode(const PeerIdentifier& peerId);

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

        /** The address to bind the server to. */
        IPEndPoint bindAddress;

        /** The TCP server waiting for clients. */
        sys::RunnableThread<TcpServer> server;

        /** The client sockets. */
        PtrArray<PeerNode> peers;

        /** Lock for protecting the 'peers' member. */
        mutable sys::CriticalSection peersLock;

    };


    /*
     * vislib::net::cluster::AbstractServerNode::initialise
     */
    template<class T>
    void AbstractServerNode::initialise(sys::CmdLineProvider<T>& inOutCmdLine) {
        typedef vislib::String<T> String;
        typedef vislib::sys::CmdLineParser<T> Parser;
        typedef typename Parser::Argument Argument;
        typedef typename Parser::Option Option;
        typedef typename Option::ValueDesc ValueDesc;

        Parser parser;
        Argument *arg = NULL;

        Option optServer(String(_T("bind-address")), 
            String(_T("Specifies the address (adapter) to bind the server to.")),
            Option::FLAG_UNIQUE, 
            ValueDesc::ValueList(Option::STRING_VALUE, 
                String(_T("adapter")), 
                String(_T("The IP address to bind the server to."))));
        parser.AddOption(&optServer);

        Option optPort(String(_T("bind-port")), 
            String(_T("Specifies the post to bind the server to.")),
            Option::FLAG_UNIQUE, 
            ValueDesc::ValueList(Option::INT_VALUE, 
                String(_T("port")), 
                String(_T("The port the server node will listen on."))));
        parser.AddOption(&optPort);

        if (parser.Parse(inOutCmdLine.ArgC(), inOutCmdLine.ArgV()) >= 0) {
            if ((arg = optServer.GetFirstOccurrence()) != NULL) {
                // TODO: IPv6
                this->bindAddress.SetIPAddress(IPAddress::Create(
                    StringA(arg->GetValueString())));
            } else {
                // TODO: IPv6
                this->bindAddress.SetIPAddress(IPAddress::ANY);
            }

            if ((arg = optPort.GetFirstOccurrence()) != NULL) {
                this->bindAddress.SetPort(arg->GetValueInt());
            } else {
                this->bindAddress.SetPort(DEFAULT_PORT);
            }
        }
    }
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTSERVERNODE_H_INCLUDED */
