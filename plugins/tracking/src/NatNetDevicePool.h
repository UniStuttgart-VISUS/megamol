/*
 * NatNetDevicePool.h
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "mmcore/param/ParamSlot.h"
#include "vislib/Delegate.h"

#include "NatNetTypes.h"
#include "NatNetClient.h"


namespace megamol {
namespace vrpnModule {

    /**
     * Manages the connection to a NatNet host application.
     */
    class NatNetDevicePool
    {

    public:

        typedef vislib::Delegate<void, const sRigidBodyData&> CallbackType;

        NatNetDevicePool(void);
        ~NatNetDevicePool(void);

        void Connect(void);
        void Disconnect(void);
        void ClearCallbacks(void);

        inline std::vector<core::param::ParamSlot *>& GetParams(void)
        {
            return this->params;
        }

        void Register(std::string rigidBodyName, CallbackType callback);

    private:

        static void __cdecl onData(sFrameOfMocapData *data, void *pUserData);
        static void __cdecl onMessage(int msgType, char *msg);

        NatNetClient *client;
        core::param::ParamSlot paramClient;
        core::param::ParamSlot paramCommandPort;
        core::param::ParamSlot paramConnectionType;
        core::param::ParamSlot paramDataPort;
        //core::param::ParamSlot paramRigidBody;
        std::vector<core::param::ParamSlot *> params;
        core::param::ParamSlot paramServer;
        core::param::ParamSlot paramVerbosity;
        std::unordered_map<std::string, int> ids;
        std::unordered_map<int, std::vector<CallbackType>> callbacks;
        
    };

} /* end namespace vrpnModule */
} /* end namespace megamol */
