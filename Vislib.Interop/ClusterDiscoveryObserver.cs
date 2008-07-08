/*
 * ClusterDiscoveryObserver.cs
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */
using System;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;
using System.Collections.Generic;
using System.Text;


namespace Vislib.Interop {

    /// <summary>
    /// This class implements a passive listener that waits for processes to
    /// join a VISlib ClusterDiscoveryService cluster and leave it. This managed
    /// code version is not able to join the cluster, it can only observe what
    /// is happening.
    /// </summary>
    public class ClusterDiscoveryObserver {

        #region Enumerations, Types and Public Constants from VISlib Native Code

        /// <summary>
        /// This class represents a peer node as it used in the CDS net. This
        /// managed class mimics the native peer node handle.
        /// </summary>
        public class PeerNode {

            /// <summary>
            /// Create a new peer node entry.
            /// </summary>
            /// <param name="address"></param>
            /// <param name="discoveryAddr"></param>
            /// <param name="cntResponseChances"></param>
            public PeerNode(SocketAddress address, IPEndPoint discoveryAddr,
                    uint cntResponseChances) {
                this.address = address;
                this.cntResponseChances = cntResponseChances;
                this.discoveryAddr = discoveryAddr;
            }

            /// <summary>
            /// Gets the communication address (ID) that the node uses.
            /// </summary>
            public SocketAddress Address {
                get {
                    return this.address;
                }
            }

            /// <summary>
            /// Implicit disconnect detector. 
            /// 
            /// TODO: This is currently not used.
            /// </summary>
            internal uint cntResponseChances = 0;

            /// <summary>
            /// Discovery service address.
            /// </summary>
            internal IPEndPoint discoveryAddr = null;

            /// <summary>
            /// User communication address (ID).
            /// </summary>
            private SocketAddress address = null;
        }

        /// <summary>
        /// Possible reasons for an OnNodeLost notification:
        ///
        /// LOST_EXPLICITLY means that the peer node explicitly disconnected by
        /// sending the sayonara message.
        ///
        /// LOST_IMLICITLY means that the peer node was removed because it did
        /// not properly answer a alive request.
        /// </summary>
        public enum NodeLostReason : int {
            LOST_EXPLICITLY = 1,
            LOST_IMLICITLY
        };

        /// <summary>
        /// The maximum length of a cluster name in characters, including the
        /// trailing zero.
        /// </summary>
        public const uint MAX_NAME_LEN = MAX_USER_DATA - ADDRESS_LENGTH;

        /// <summary>
        /// The first message ID that can be used for a user message.
        /// </summary>
        protected const UInt32 MSG_TYPE_USER = 16;

        #endregion

        #region Construction

        /// <summary>
        /// Create a new ClusterDiscoveryObserver.
        /// </summary>
        /// <param name="name">This is the name of the cluster to detect. It is
        /// used to ensure that nodes answering a discovery request want to join
        /// the same cluster. The name must have at most MAX_NAME_LEN 
        /// characters.</param>
        public ClusterDiscoveryObserver(string name) 
                : this(name, DEFAULT_PORT) {
        }

        /// <summary>
        /// Create a new ClusterDiscoveryObserver.
        /// </summary>
        /// <param name="name">This is the name of the cluster to detect. It is
        /// used to ensure that nodes answering a discovery request want to join
        /// the same cluster. The name must have at most MAX_NAME_LEN 
        /// characters.</param>
        /// <param name="bindPort">The port to bind the receiver thread to. All
        /// discovery requests must be directed to this port.</param>
        public ClusterDiscoveryObserver(string name, ushort bindPort)
                : this(name, bindPort, new TimeSpan(0, 0, 10), 1) {
        }

        /// <summary>
        /// Create a new ClusterDiscoveryObserver.
        /// </summary>
        /// <param name="name">This is the name of the cluster to detect. It is
        /// used to ensure that nodes answering a discovery request want to join
        /// the same cluster. The name must have at most MAX_NAME_LEN 
        /// characters.</param>
        /// <param name="bindPort">The port to bind the receiver thread to. All
        /// discovery requests must be directed to this port.</param>
        /// <param name="requestInterval">The interval between two discovery 
        /// requests.</param>
        /// <param name="cntResponseChances">The number of requests that another
        /// node may not answer before being removed from this nodes list of
        /// known peers.</param>
        public ClusterDiscoveryObserver(string name, ushort bindPort,
                TimeSpan requestInterval, uint cntResponseChances) {
            this.name = name;
            this.bindAddr = new IPEndPoint(IPAddress.Any, (int) bindPort);
            this.cntResponseChances = cntResponseChances;
            this.requestInterval = requestInterval;
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Clear all peer nodes that have been found until now.
        /// </summary>
        void ClearPeers() {
            lock (this.peerNodes) {
                this.peerNodes.Clear();
            }
        }

        /// <summary>
        /// Start the listener thread and wait for CDS messages. If the CDS 
        /// listener is already running, nothing happens.
        /// </summary>
        public void Start() {
            // Ensure that thread exists, because we want to lock it.
            if (this.listener == null) {
                this.listener = new Thread(this.listen);
                this.listener.Name = this.GetType().Name;
            }

            // Lock thread to protect access to 'isListening' flag.
            lock (this.listener) {
                if (!this.isListening) {
                    this.isListening = true;
                    this.listener.Start();
                }
            }
        }

        /// <summary>
        /// Stop the listener thread.
        /// </summary>
        public void Stop() {
            // Ensure that thread has been created before going on.
            if (this.listener != null) {

                // Lock thread to protect access to 'isListening' flag.
                lock (this.listener) {
                    this.isListening = false;
                    try {
                        // Closing the socket and setting 'isListening' to 
                        // false will cause the listener thread to exit.
                        this.listenSocket.Close();
                    } catch (SocketException e) {
                        Debug.WriteLine("This is probably no problem: A socket "
                            + "exception was caught while terminating the "
                            + "listener thread: " + e.Message);
                    }
                } /* end lock (this.listener) */
            } /* end if (this.listener != null) */
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Gets the number of known peer nodes
        /// </summary>
        int PeerCount {
            get {
                lock (this.peerNodes) {
                    return this.peerNodes.Count;
                }
            }
        }

        /// <summary>
        /// Gets the address the service is listening on for discovery requests.
        /// </summary>
        IPEndPoint BindAddr {
            get {
                return this.bindAddr;
            }
        }

        /// <summary>
        /// Gets the cluster identifier that is used for discovery.
        /// </summary>
        string Name {
            get {
                return this.name;
            }
        }

        #endregion

        #region Events

        /// <summary>
        /// Event handler delegate for being notified about a new node joining
        /// the cluster.
        /// </summary>
        /// <param name="src">The object that fired the event.</param>
        /// <param name="hPeer">Handle for the new node.</param>
        public delegate void NodeFoundHandler(ClusterDiscoveryObserver src,
            PeerNode hPeer);

        /// <summary>
        /// Event handler delegate for bing notified about a node leaving the
        /// cluster.
        /// </summary>
        /// <param name="src">The object that fired the event.</param>
        /// <param name="hPeer">Handle to the node that was lost.</param>
        /// <param name="reason">The reason why the node was removed from the 
        /// node list of the source object.</param>
        public delegate void NodeLostHandler(ClusterDiscoveryObserver src,
            PeerNode hPeer, NodeLostReason reason);

        /// <summary>
        /// Event handler for receiving user messaged.
        /// </summary>
        /// <param name="src">The object that fired the event.</param>
        /// <param name="hPeer">Handle to the node that sent the message.
        /// </param>
        /// <param name="msgType">The message ID.</param>
        /// <param name="msgBody">The message body.</param>
        public delegate void UserMessageHandler(ClusterDiscoveryObserver src,
            PeerNode hPeer, UInt32 msgType, byte[] msgBody);

        /// <summary>
        /// Event for being notified about new nodes.
        /// </summary>
        public event NodeFoundHandler NodeFound;

        /// <summary>
        /// Event for being notified about nodes being removed.
        /// </summary>
        public event NodeLostHandler NodeLost;

        /// <summary>
        /// Event for being notified about user-defined messages.
        /// </summary>
        public event UserMessageHandler UserMessage;

        #endregion

        #region Magic Numbers from VISlib Native Code

        /// <summary>
        /// Size of the socket addresses used in native code in bytes
        /// (sizeof(struct sockaddr_in)).
        /// </summary>
        protected const int ADDRESS_LENGTH = 4;

        /// <summary>
        /// The default port that is used by the VISlib ClusterDiscoveryService.
        /// </summary>
        protected static readonly UInt16 DEFAULT_PORT = 28181;

        /// <summary>
        /// The length of the message header in bytes.
        /// 
        /// The total message size is always the size of the header plus the 
        /// maximum user message size.
        /// </summary>
        protected const uint HEADER_LENGTH = 2 * sizeof(UInt32);

        /// <summary>
        /// The maximum user message length in bytes.
        /// 
        /// The total message size is always the size of the header plus the 
        /// maximum user message size.
        /// </summary>
        protected const uint MAX_USER_DATA = 256;

        /// <summary>
        /// The magic number at the begin of each message.
        /// </summary>
        protected const UInt32 MAGIC_NUMBER = ((UInt32) ('v') << 0)
            | ((UInt32)('c') << 8) 
            | ((UInt32)('d') << 16) 
            | ((UInt32)('s') << 24);

        /// <summary>
        /// Message type ID of a repeated discovery request.
        /// </summary>
        protected const UInt32 MSG_TYPE_IAMALIVE = 2;

        /// <summary>
        /// Message type ID of an initial discovery request.
        /// </summary>
        protected const UInt32 MSG_TYPE_IAMHERE = 1;

        /// <summary>
        /// Message type ID of the explicit disconnect notification.
        /// </summary>
        protected const UInt32 MSG_TYPE_SAYONARA = 3;

        #endregion

        #region Helper Methods

        /// <summary>
        /// Add a new peer node to <c>this.peerNodes</c>. The 
        /// <c>cntResponseChances</c> of the new node is set to 
        /// <c>this.cntResponseChances</c>. If the node is already known, its
        /// <c>cntResponeChances</c> counter is reset.
        /// 
        /// If the node is new, the <c>NodeFound</c> event is 
        /// fired by this method.
        /// </summary>
        /// <param name="address"></param>
        /// <param name="peerAddr"></param>
        protected void addPeerNode(SocketAddress address,
                EndPoint peerAddr) {
            NodeFoundHandler evt = this.NodeFound;
            PeerNode peerNode = null;

            /* Add to collection. */
            lock (this.peerNodes) {
                peerNode = this.peerNodes.Find(delegate(PeerNode p) {
                    return p.Address.Equals(address);
                });

                if (peerNode != null) {
                    /* Node alredy known, reset response counter. */
                    Debug.WriteLine("Node " + address.ToString() + "("
                        + peerAddr.ToString() + ") is already known.");
                    peerNode.cntResponseChances = this.cntResponseChances;

                } else {
                    /* This is a new node, add it and fire event. */
                    Debug.WriteLine("Node " + address.ToString() + "("
                        + peerAddr.ToString() + ") added.");
                    peerNode = new PeerNode(address, (IPEndPoint) peerAddr,
                        this.cntResponseChances);
                    this.peerNodes.Add(peerNode);

                    if (evt != null) {
                        evt(this, peerNode);
                    }
                } /* end if (peerNode != null) */
            } /* end lock (this.peerNodes) */
        }


        protected void fireUserMessage(EndPoint peerAddr, UInt32 msgType, 
                byte[] msgBody, uint bodyOffset) {
            UserMessageHandler evt = this.UserMessage;
            IPEndPoint addr = peerAddr as IPEndPoint;

            Debug.WriteLine("Firing user message notification ...");

            if ((addr != null) && (evt != null)) {
                PeerNode peerNode = this.peerFromDiscoveryAddr(addr);

                if ((peerNode != null)) {
                    byte[] body = new byte[MAX_USER_DATA];
                    Array.Copy(msgBody, bodyOffset, body, 0, MAX_USER_DATA);
                    evt(this, peerNode, msgType, body);
                }
            }
        }

        /// <summary>
        /// This is the thread function that waits for messages from the CDS
        /// net.
        /// </summary>
        protected void listen() {
            this.listenSocket = new Socket(AddressFamily.InterNetwork,
                SocketType.Dgram, ProtocolType.IP);
            byte[] recvBuffer = new byte[HEADER_LENGTH + MAX_USER_DATA];
            int bytesReceived = 0;
            EndPoint peerAddr = new IPEndPoint(IPAddress.Any, 0);
            SocketAddress peerId = new SocketAddress(
                AddressFamily.InterNetwork, ADDRESS_LENGTH);
            string peerName = "";

            /* Configure the socket and bind it to the discovery address. */
            this.listenSocket.SetSocketOption(SocketOptionLevel.Socket,
                SocketOptionName.Broadcast, true);
            // TODO: The following socket options are possibly a security issue.
            this.listenSocket.SetSocketOption(SocketOptionLevel.Socket,
                SocketOptionName.ExclusiveAddressUse, false);
            this.listenSocket.SetSocketOption(SocketOptionLevel.Socket,
                SocketOptionName.ReuseAddress, true);
            this.listenSocket.Bind(this.bindAddr);

            /* Enter the receiver loop. */
            Debug.WriteLine("ClusterDiscoveryObserver now starts listening on "
                + this.bindAddr.ToString() + " ...");
            while (true) {
                try {
                    bytesReceived = this.listenSocket.ReceiveFrom(recvBuffer,
                        (int) (HEADER_LENGTH + MAX_USER_DATA), SocketFlags.None,
                        ref peerAddr);
                    Debug.WriteLine("Received " + bytesReceived.ToString()
                        + " Bytes from " + peerAddr.ToString());

                    // Message header consists of
                    // - 2 byte magic number
                    // - 2 byte message type ID.
                    if (BitConverter.ToUInt32(recvBuffer, 0) == MAGIC_NUMBER) {
                        UInt32 msgType = BitConverter.ToUInt32(recvBuffer, 2);
                        Debug.WriteLine("Type of message received is "
                            + msgType.ToString());

                        /* Unpack the default message. */
                        switch (msgType) {
                            case MSG_TYPE_IAMALIVE:
                                /* falls through. */
                            case MSG_TYPE_IAMHERE:
                            /* falls through. */
                            case MSG_TYPE_SAYONARA:
                                // The message body consists of
                                // - ADDRESS_LENGTH bytes of an IPv4 address
                                // - MAX_NAME_LEN bytes of the name string.
                                for (int i = 0; i < ADDRESS_LENGTH; i++) {
                                    peerId[i] = recvBuffer[HEADER_LENGTH + i];
                                }
                                peerName = new ASCIIEncoding().GetString(
                                    recvBuffer,
                                    (int) HEADER_LENGTH,
                                    (int) MAX_NAME_LEN);
                                Debug.WriteLine("Peer response address (ID) is "
                                    + peerId.ToString());
                                Debug.WriteLine("Cluster name is " + peerName);
                                break;

                            default:
                                /* Other messages should not be unpacked. */
                                break;
                        }

                        /* Take appropriate action for the message. */
                        switch (msgType) {
                            case MSG_TYPE_IAMALIVE:
                                /* falls through. */
                            case MSG_TYPE_IAMHERE:
                                if (peerName == this.name) {
                                    this.addPeerNode(peerId, peerAddr);
                                }
                                break;

                            case MSG_TYPE_SAYONARA:
                                if (peerName == this.name) {
                                    this.removePeerNode(peerId);
                                }
                                break;

                            default:
                                if (msgType >= MSG_TYPE_USER) {
                                    this.fireUserMessage(peerAddr, msgType,
                                        recvBuffer, HEADER_LENGTH);
                                }
                                break;
                        }
                    } /* end if (BitConverter.ToUInt32(recvBuffer, 0) ... */
                } catch (SocketException e) {
                    Debug.WriteLine("A communication error occurred in the "
                        + "cluster discovery listener thread: " + e.Message);
                    lock (this.listener) {
                        if (!this.isListening) {
                            Debug.WriteLine("The cluster discovery listener "
                                + "thread is leaving, because a stop was "
                                + "requested.");
                            return;
                        }
                    }
                } /* end try */
            } /* end while (true) */
        }

        /// <summary>Answer the index of the peer node that runs its discovery
        /// service on <c>addr</c>. If no such node exists, null is returned.
        /// Only the IP address is taken into account, not the port. It is 
        /// therefore safe to use the UDP sender address for <c>addr</c>.
        /// </summary>
        /// <param name="addr">The discovery address to lookup.</param>
        /// <returns>The peer node if found, null otherwise.</returns>
        PeerNode peerFromDiscoveryAddr(IPEndPoint addr) {
            lock (this.peerNodes) {
                foreach (PeerNode peerNode in this.peerNodes) {
                    if (peerNode.discoveryAddr.Address.Equals(addr.Address)) {
                        return peerNode;
                    }
                }
            }
            /* Nothing found here. */

            return null;
        }

        /// <summary>
        /// Remove the peer node having the user communication address 
        /// <c>address</c>. If no such node exists, nothing will happen.
        /// This method also fires the node lost event sigaling an explicit
        /// remove of the node.
        /// </summary>
        /// <param name="address">The socket address that the peer node reported
        /// as its user communication port.</param>
        void removePeerNode(SocketAddress address) {
            NodeLostHandler evt = this.NodeLost;

            lock (this.peerNodes) {
                PeerNode peerNode = this.peerNodes.Find(delegate(PeerNode p) {
                    return p.Address.Equals(address);
                });

                if (peerNode != null) {
                    this.peerNodes.Remove(peerNode);

                    /* Fire the event. */
                    if (evt != null) {
                        evt(this, peerNode, NodeLostReason.LOST_EXPLICITLY);
                    }
                }
            }
        }

        #endregion

        #region Protected Attributes

        /// <summary>
        /// The address that the listener thread binds to.
        /// </summary>
        protected IPEndPoint bindAddr = new IPEndPoint(IPAddress.Any,
            ClusterDiscoveryObserver.DEFAULT_PORT);

        /// <summary>
        /// The number of chances a node gets to respond before it is 
        /// implicitly disconnected from the cluster.
        /// </summary>
        protected uint cntResponseChances = 0;

        /// <summary>
        /// The name of the cluster this discovery service should form.
        /// </summary>
        protected string name = "";

        /// <summary>
        /// The time between two discovery requests.
        /// </summary>
        protected TimeSpan requestInterval = new TimeSpan(0, 0, 10);

        #endregion

        #region Private Attributes

        /// <summary>
        /// Remembers whether the listener thread is and should continue 
        /// running. This flag is used to terminate the listener. If it is
        /// false and a communication error is provoked, the listener thread
        /// exits. This flag is also used as a restart guard to 
        /// <c>this.listener</c>. Access should be protected by locking 
        /// <c>this.listener</c>.
        /// </summary>
        private bool isListening = false;

        /// <summary>
        /// The list of known peer nodes.
        /// </summary>
        private List<PeerNode> peerNodes = new List<PeerNode>();

        /// <summary>
        /// The listener thread that waits for CDS messages to come in.
        /// </summary>
        private Thread listener = null;

        /// <summary>
        /// 
        /// </summary>
        private Socket listenSocket = null;

        #endregion

    }
}
