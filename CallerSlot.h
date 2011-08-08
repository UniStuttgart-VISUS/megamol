/*
 * CallerSlot.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLERSLOT_H_INCLUDED
#define MEGAMOLCORE_CALLERSLOT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "AbstractSlot.h"
#include "Call.h"
#include "CallDescription.h"
#include "vislib/Array.h"
#include "vislib/String.h"


namespace megamol {
namespace core {

    /** forward declaration */
    class CalleeSlot;


    /**
     * The caller slot connect one calling module with one called module
     * using a Call object.
     */
    class MEGAMOLCORE_API CallerSlot : public AbstractSlot {
    public:

        /**
         * Ctor.
         *
         * @param name The name of this slot.
         * @param desc A human readable description of this slot.
         */
        CallerSlot(const vislib::StringA& name,
            const vislib::StringA& desc);

        /** Dtor. */
        virtual ~CallerSlot(void);

        /**
         * Calls the call function 'func'.
         *
         * @param func The function to be called.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool Call(unsigned int func = 0);

        /**
         * Gets the call dynamic-casted to the specified template type.
         *
         * @return The casted call or 'NULL' in case the slot is not connected
         *         or the call could not be casted.
         */
        template<class T> inline T * CallAs(void) {
            return dynamic_cast<T*>(this->call);
        }

        /**
         * Connects a call to this slot. The caller slot takes the memory
         * ownership of the call object.
         *
         * @param call The call to connect.
         *
         * @return 'true' on success, 'false' on failure
         */
        bool ConnectCall(megamol::core::Call *call) {
            if (this->Parent()) this->Parent()->LockModuleGraph(true);

            if (call == NULL) {
                if (this->call != NULL) {
                    this->SetStatusDisconnected();
                    megamol::core::Call *c = this->call;
                    this->call = NULL;
                    delete c;
                }
                if (this->Parent()) this->Parent()->UnlockModuleGraph(true);
                return true;
            }

            for (unsigned int i = 0; i < this->compDesc.Count(); i++) {
                if (this->compDesc[i]->IsDescribing(call)) {
                    if (this->call != NULL) this->call->caller = NULL;
                    delete this->call;
                    this->call = call;
                    this->call->caller = this;
                    this->SetStatusConnected();
                    if (this->Parent()) this->Parent()->UnlockModuleGraph(true);
                    return true;
                }
            }

            if (this->Parent()) this->Parent()->UnlockModuleGraph(true);
            return false;
        }

        /**
         * Answers whether a given call is compatible with this slot.
         *
         * @param desc The description of the call to test.
         *
         * @return 'true' if the call is compatible, 'false' otherwise.
         */
        inline bool IsCallCompatible(CallDescription* desc) const {
            vislib::StringA dcn;
            if (desc == NULL) return false;
            dcn = desc->ClassName();
            for (unsigned int i = 0; i < this->compDesc.Count(); i++) {
                if (dcn.Equals(this->compDesc[i]->ClassName())) {
                    return true;
                }
            }
            return false;
        }

        /**
         * Registers a call as compatible call with this caller slot. Use the
         * description class of the call as template parameter 'T'. Calls of
         * this class can then be connected to this slot.
         */
        template<class T> inline void SetCompatibleCall(void) {
            T *d = new T();
            for (unsigned int i = 0; i < this->compDesc.Count(); i++) {
                if (vislib::StringA(this->compDesc[i]->ClassName())
                        .Equals(d->ClassName())) {
                    delete d;
                    return;
                }
            }
            this->compDesc.Append(d);
        }

        /**
         * Registers a call as compatible call with this caller slot. Calls of
         * this class can then be connected to this slot.
         *
         * @param desc The description object to be added.
         */
        inline void SetCompatibleCall(const CallDescription& desc) {
            for (unsigned int i = 0; i < this->compDesc.Count(); i++) {
                if (vislib::StringA(this->compDesc[i]->ClassName())
                        .Equals(desc.ClassName())) {
                    return;
                }
            }
            this->compDesc.Append(desc.Clone());
        }

        /**
         * Answers whether this caller is connected to the specified callee.
         *
         * @param target The callee slot
         *
         * @return The connecting call or 'NULL' if 'this' and 'target' are not connected.
         */
        ::megamol::core::Call *IsConnectedTo(::megamol::core::CalleeSlot *target);

        /**
         * Answers whether this caller is connected to the specified callee.
         *
         * @param target The callee slot
         *
         * @return The connecting call or 'NULL' if 'this' and 'target' are not connected.
         */
        const ::megamol::core::Call *IsConnectedTo(const ::megamol::core::CalleeSlot *target) const;

        /**
         * Clears the cleanup mark for this and all dependent objects.
         */
        virtual void ClearCleanupMark(void);

        /**
         * Disconnects calls from all slots which are marked for cleanup.
         */
        virtual void DisconnectCalls(void);

        /**
         * Answers whether the given parameter is relevant for this view.
         *
         * @param searched The already searched objects for cycle detection.
         * @param param The parameter to test.
         *
         * @return 'true' if 'param' is relevant, 'false' otherwise.
         */
        virtual bool IsParamRelevant(
            vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
            const vislib::SmartPtr<param::AbstractParam>& param) const;

    private:

        /** The connecting call object */
        megamol::core::Call *call;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** Array of descriptions of compatible calls. */
        vislib::Array<CallDescription *> compDesc;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLERSLOT_H_INCLUDED */
