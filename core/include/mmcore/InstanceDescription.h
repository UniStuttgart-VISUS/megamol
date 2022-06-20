/*
 * InstanceDescription.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_INSTANCEDESCRIPTION_H_INCLUDED
#define MEGAMOLCORE_INSTANCEDESCRIPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/ParamValueSetRequest.h"
#include "mmcore/factories/CallDescription.h"
#include "mmcore/factories/ModuleDescription.h"
#include "mmcore/factories/ObjectDescription.h"
#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/String.h"


namespace megamol {
namespace core {

/**
 * Abstract base class of job and view descriptions.
 */
class InstanceDescription : public factories::ObjectDescription, public ParamValueSetRequest {
public:
    /** Type of module instantiation requests */
    typedef vislib::Pair<vislib::StringA, factories::ModuleDescription::ptr> ModuleInstanceRequest;

    /** Type of call instantiation requests */
    class CallInstanceRequest {
    public:
        /** Ctor */
        CallInstanceRequest(void) : from(), to(), desc(), doProfiling(false) {
            // intentionally empty
        }

        /**
         * Copy Ctor
         *
         * @param rhs The object to clone from
         */
        CallInstanceRequest(const CallInstanceRequest& rhs)
                : from(rhs.from)
                , to(rhs.to)
                , desc(rhs.desc)
                , doProfiling(rhs.doProfiling) {
            // intentionally empty
        }

        /**
         * Ctor
         *
         * @param caller The id of the caller slot.
         * @param callee The id of the callee slot.
         * @param desc The description of the call to be instantiated.
         * @param doProfiling Flag if this call should be added to profiling
         *            if the profiling set is to 'SELECTED'
         */
        CallInstanceRequest(const vislib::StringA& caller, const vislib::StringA& callee,
            factories::CallDescription::ptr desc, bool doProfiling)
                : from(caller)
                , to(callee)
                , desc(desc)
                , doProfiling(doProfiling) {
            // intentionally empty
        }

        /**
         * Answers the id of the caller slot.
         *
         * @return The id of the caller slot.
         */
        inline const vislib::StringA& From(void) const {
            return this->from;
        }

        /**
         * Answers the id of the callee slot.
         *
         * @return The id of the callee slot.
         */
        inline const vislib::StringA& To(void) const {
            return this->to;
        }

        /**
         * Gets the description of the call to be instantiated.
         *
         * @return The description of the call to be instantiated.
         */
        inline factories::CallDescription::ptr Description(void) const {
            return this->desc;
        }

        /**
         * Gets the flag if this call should be added to profiling
         *
         * @return The flag if this call should be added to profiling
         */
        inline bool DoProfiling(void) const {
            return this->doProfiling;
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return True if 'this' and 'rhs' are equal
         */
        inline bool operator==(const CallInstanceRequest& rhs) const {
            return (this->from == rhs.from) && (this->to == rhs.to) && (this->desc == rhs.desc) &&
                   (this->doProfiling == rhs.doProfiling);
        }

        /**
         * Assignment operator
         *
         * @param rhs The object to copy from
         *
         * @return A reference to this object
         */
        inline CallInstanceRequest& operator=(const CallInstanceRequest& rhs) {
            this->from = rhs.from;
            this->to = rhs.to;
            this->desc = rhs.desc;
            this->doProfiling = rhs.doProfiling;
            return *this;
        }

    private:
        /** caller The id of the caller slot. */
        vislib::StringA from;

        /** callee The id of the callee slot. */
        vislib::StringA to;

        /** The description of the call to be instantiated. */
        factories::CallDescription::ptr desc;

        /** Flag if this call should be added to profiling */
        bool doProfiling;
    };

    /**
     * Add a request for a call instantiation to this instance
     * description.
     *
     * @param desc The description of the call to be instantiated.
     * @param caller The id of the caller slot.
     * @param callee The id of the callee slot.
     * @param doProfiling Flag if this call should be added to profiling
     *            if the profiling set is to 'SELECTED'
     */
    inline void AddCall(factories::CallDescription::ptr desc, const vislib::StringA& caller,
        const vislib::StringA& callee, bool doProfiling = false) {
        this->calls.Add(CallInstanceRequest(caller, callee, desc, doProfiling));
    }

    /**
     * Add a request for a module instantiation to this instance
     * description.
     *
     * @param desc The description of the module to be instantiated.
     * @param id The id for the module instance
     */
    inline void AddModule(factories::ModuleDescription::ptr desc, const vislib::StringA& id) {
        this->modules.Add(ModuleInstanceRequest(id, desc));
    }

    /**
     * Gets the 'idx'-th call instantiation request.
     *
     * @param idx The index of the instantiation request to be returned.
     *
     * @return The requested call instantiation request.
     */
    inline const CallInstanceRequest& Call(unsigned int idx) const {
        return this->calls[idx];
    }

    /**
     * Gets the number of call instantiation requests in this job
     * description.
     *
     * @return The number of module instantiation requests.
     */
    inline unsigned int CallCount(void) const {
        return static_cast<unsigned int>(this->calls.Count());
    }

    /**
     * Answer the name of the job of this description.
     *
     * @return The name of the job of this description.
     */
    virtual const char* ClassName(void) const {
        return this->classname.PeekBuffer();
    }

    /**
     * Removes all call instantiation requests
     */
    void ClearCalls(void) {
        this->calls.Clear();
    }

    /**
     * Removes all module instantiation requests
     */
    void ClearModules(void) {
        this->modules.Clear();
    }

    /**
     * Gets a human readable description of the job.
     *
     * @return A human readable description of the job.
     */
    virtual const char* Description(void) const {
        return this->description.PeekBuffer();
    }

    /**
     * Gets the 'idx'-th module instantiation request.
     *
     * @param idx The index of the instantiation request to be returned.
     *
     * @return The requested module instantiation request.
     */
    inline const ModuleInstanceRequest& Module(unsigned int idx) const {
        return this->modules[idx];
    }

    /**
     * Gets the number of module instantiation requests in this job
     * description.
     *
     * @return The number of module instantiation requests.
     */
    inline unsigned int ModuleCount(void) const {
        return static_cast<unsigned int>(this->modules.Count());
    }

    /**
     * Sets a human readable description for the job.
     *
     * @param desc A human readable description for the job.
     */
    inline void SetDescription(const vislib::StringA& desc) {
        this->description = desc;
    }

protected:
    /**
     * Ctor.
     *
     * @param classname The name of the instance described.
     */
    InstanceDescription(const char* classname);

    /**
     * Dtor.
     */
    virtual ~InstanceDescription(void);

private:
#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The name of the instance described. */
    vislib::StringA classname;

    /** The human readable description of the instance. */
    vislib::StringA description;

    /** The instantiation requests of the modules */
    vislib::Array<ModuleInstanceRequest> modules;

    /** The instantiation requests of the calls */
    vislib::Array<CallInstanceRequest> calls;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */
};

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_INSTANCEDESCRIPTION_H_INCLUDED */
