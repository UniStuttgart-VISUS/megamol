/*
 * CallRegisterAtController.h
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLREGISTERATCONTROLLER_H_INCLUDED
#define MEGAMOLCORE_CALLREGISTERATCONTROLLER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include <vislib/String.h>


namespace megamol {
namespace core {
namespace cluster {

/** forward declaration */
class ClusterControllerClient;


/**
 * Call for registering a module at the cluster controller
 */
class CallRegisterAtController : public Call {
public:
    /** Call number to register */
    static const unsigned int CALL_REGISTER = 0;

    /** Call number to unregister */
    static const unsigned int CALL_UNREGISTER = 1;

    /** Call number to query the status */
    static const unsigned int CALL_GETSTATUS = 2;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "CallRegisterAtController";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call for registering a module at the cluster controller";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return 3;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case CALL_REGISTER:
            return "register";
        case CALL_UNREGISTER:
            return "unregister";
        case CALL_GETSTATUS:
            return "getstatus";
        default:
            return NULL;
        }
    }

    /**
     * Ctor.
     */
    CallRegisterAtController(void);

    /**
     * ~Dtor.
     */
    virtual ~CallRegisterAtController(void);

    /**
     * Gets the client to be un-/registered
     *
     * @return The client to be un-/registered
     */
    inline ClusterControllerClient* Client(void) {
        return this->client;
    }

    /**
     * Gets the client to be un-/registered
     *
     * @return The client to be un-/registered
     */
    inline const ClusterControllerClient* Client(void) const {
        return this->client;
    }

    /**
     * Gets the name of the cluster
     *
     * @return The name of the cluster
     */
    inline const vislib::StringA& GetStatusClusterName(void) const {
        return this->statClstrName;
    }

    /**
     * Gets the number of connected peers
     *
     * @return The number of connected peers
     */
    inline unsigned int GetStatusPeerCount(void) const {
        return this->statPeerCnt;
    }

    /**
     * Gets the Flag whether or not the discovery service is running
     *
     * @return The flag whether or not the discovery service is running
     */
    inline bool GetStatusRunning(void) const {
        return this->statRun;
    }

    /**
     * Sets the client to be un-/registered
     *
     * @param c The client to be un-/registered
     */
    inline void SetClient(ClusterControllerClient* c) {
        this->client = c;
    }

    /**
     * Sets the status of the controller
     *
     * @param running Flag whether or not the discovery service is running
     * @param peerCount The number of connected peers
     * @param clstrName The name of the cluster
     */
    inline void SetStatus(bool running, unsigned int peerCount, const vislib::StringA& clstrName) {
        this->statRun = running;
        this->statPeerCnt = peerCount;
        this->statClstrName = clstrName;
        // TODO: more to come
    }

private:
    /** The client to be un-/registered */
    ClusterControllerClient* client;

    /** Flag whether or not the discovery service is running */
    bool statRun;

    /** The number of connected peers */
    unsigned int statPeerCnt;

    /** The name of the cluster */
    vislib::StringA statClstrName;
};


/** Description class typedef */
typedef factories::CallAutoDescription<CallRegisterAtController> CallRegisterAtControllerDescription;


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLREGISTERATCONTROLLER_H_INCLUDED */
