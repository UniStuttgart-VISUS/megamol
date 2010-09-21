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

#include "api/MegaMolCore.std.h"
#include "ObjectDescription.h"
#include "ModuleDescription.h"
#include "CallDescription.h"
#include "ParamValueSetRequest.h"
#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/String.h"


namespace megamol {
namespace core {

    /**
     * Abstract base class of job and view descriptions.
     */
    class MEGAMOLCORE_API InstanceDescription : public ObjectDescription,
        public ParamValueSetRequest {
    public:

        /** Type of module instantiation requests */
        typedef vislib::Pair<vislib::StringA, ModuleDescription*>
            ModuleInstanceRequest;

        /** Type of call instantiation requests */
        typedef vislib::Pair<vislib::Pair<vislib::StringA, vislib::StringA>,
            CallDescription*> CallInstanceRequest;

        /**
         * Add a request for a call instantiation to this instance
         * description.
         *
         * @param desc The description of the call to be instantiated.
         * @param caller The id of the caller slot.
         * @param callee The id of the callee slot.
         */
        inline void AddCall(CallDescription *desc,
                const vislib::StringA& caller, const vislib::StringA& callee) {
            this->calls.Add(CallInstanceRequest(vislib::Pair<vislib::StringA,
                vislib::StringA>(caller, callee), desc));
        }

        /**
         * Add a request for a module instantiation to this instance
         * description.
         *
         * @param desc The description of the module to be instantiated.
         * @param id The id for the module instance
         */
        inline void AddModule(ModuleDescription *desc,
                const vislib::StringA& id) {
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
        virtual const char *ClassName(void) const {
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
        virtual const char *Description(void) const {
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
        InstanceDescription(const char *classname);

        /**
         * Dtor.
         */
        virtual ~InstanceDescription(void);

    private:

#ifdef _WIN32
#pragma warning (disable: 4251)
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
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_INSTANCEDESCRIPTION_H_INCLUDED */
