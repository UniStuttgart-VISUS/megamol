/*
 * CommServer.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/SingleLinkedList.h"
#include "vislib/StringConverter.h"
#include "vislib/net/AbstractCommServerChannel.h"
#include "vislib/net/CommServerListener.h"
#include "vislib/net/TcpCommChannel.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/Runnable.h"


namespace vislib::net {


/**
 * This class implements a server for VISlib communication channels.
 */
class CommServer : public vislib::sys::Runnable {

public:
    /**
     * This structure contains the server configuration and is passed
     * to the Run() method of the server. The server will copy the data
     * of the configuration to local storage. Therefore the caller can
     * release its reference to the resources in the configuration once
     * the Start() method of the server thread returns. The server will
     * not release any references except for those it added by itself.
     */
    typedef struct Configuration_t {
        inline Configuration_t() {}
        inline Configuration_t(SmartRef<AbstractCommServerChannel> channel, SmartRef<AbstractCommEndPoint> endPoint)
                : Channel(channel)
                , EndPoint(endPoint) {}
        inline Configuration_t(SmartRef<TcpCommChannel> channel, SmartRef<AbstractCommEndPoint> endPoint)
                : Channel(channel.DynamicCast<AbstractCommServerChannel>())
                , EndPoint(endPoint) {}

        /** Channel to be used by the server. */
        SmartRef<AbstractCommServerChannel> Channel;

        /** The end point to bind the server to. */
        SmartRef<AbstractCommEndPoint> EndPoint;
    } Configuration;

    /** Ctor. */
    CommServer();

    /** Dtor. */
    ~CommServer() override;

    /**
     * Add a new CommServerListener to be informed about events of this
     * server.
     *
     * The caller remains owner of the memory designated by 'listener' and
     * must ensure that the object exists as long as the listener is
     * registered.
     *
     * This method is thread-safe.
     *
     * @param listener The listener to register. This must not be NULL.
     */
    void AddListener(CommServerListener* listener);

    /**
     * Answer the configuration of the server. Callers should never
     * manipulate the configuration returned here.
     *
     * This method is not thread-safe!
     *
     * @return The server configuration.
     */
    inline const Configuration& GetConfiguration() const {
        return this->configuration;
    }

    /**
     * Startup callback of the thread. The Thread class will call that
     * before Run().
     *
     * @param config A pointer to the Configuration, which specifies the
     *               settings of the server.
     */
    void OnThreadStarting(void* config) override;

    /**
     * Removes, if registered, 'listener' from the list of objects informed
     * about events events.
     *
     * The caller remains owner of the memory designated by 'listener'.
     *
     * This method is thread-safe.
     *
     * @param listener The listener to be removed. Nothing happens, if the
     *                 listener was not registered.
     */
    void RemoveListener(CommServerListener* listener);

    /**
     * Perform the work of a thread.
     *
     * @param config A pointer to the Configuration, which specifies the
     *               settings of the server.
     *
     * @return The application dependent return code of the thread. This
     *         must not be STILL_ACTIVE (259).
     */
    DWORD Run(void* config) override;

    /**
     * Abort the work of the server by forcefully closing the underlying
     * communication channel.
     *
     * @return true.
     */
    bool Terminate() override;

private:
    /** A thread-safe list for the message listeners. */
    typedef SingleLinkedList<CommServerListener*, sys::CriticalSection> ListenerList;

    /**
     * Inform all registered listener about a connection that was
     * established.
     *
     * This method is thread-safe.
     *
     * @param channel The client communication channel.
     *
     * @return true if the connection was accepted, false otherwise.
     */
    bool fireNewConnection(SmartRef<AbstractCommClientChannel>& channel);

    /**
     * Inform all registered listener about an exception that was caught.
     *
     * This method is thread-safe.
     *
     * @param exception The exception that was caught.
     *
     * @return Accumulated (ANDed) return values of all listeners.
     */
    bool fireServerError(const vislib::Exception& exception);

    /**
     * Inform all registered listener about that the server is exiting.
     *
     * This method is thread-safe.
     */
    void fireServerExited();

    /**
     * Inform all registered listener about that the server is starting.
     *
     * This method is thread-safe.
     */
    void fireServerStarted();

    /** The configuration of the server. */
    Configuration configuration;

    /**
     * Flag indicating that the server should proceed with accepting
     * clients.
     */
    VL_INT32 doServe;

    /** The list of listeners. */
    ListenerList listeners;
};

} // namespace vislib::net

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
