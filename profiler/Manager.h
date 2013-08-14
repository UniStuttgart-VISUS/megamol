/*
 * profiler/Manager.h
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PROFILER_MANAGER_H_INCLUDED
#define MEGAMOLCORE_PROFILER_MANAGER_H_INCLUDED
#pragma once

#include "vislib/String.h"
#include "CoreInstance.h"
#include "vislib/Array.h"
#include "profiler/Connection.h"


namespace megamol {
namespace core {
namespace profiler {

    /**
     * Runtime profiling manager
     *
     * Perform call profiling:
     *
     * Add to megamol.cfg:
     *    <set name="profiling" value="all" />
     *    <set name="profiling" value="selected" />
     *    <set name="profiling" value="none" />
     * A value interpretable as boolean 'true' will result in 'selected'.
     * Any other values will result in 'none'.
     *
     * Add to megamol.mmprj:
     *    <call ... profile="true" />
     * The value of 'profile' must be interpretable as boolean 'true' to select this call for profiling.
     */
    class Manager {
    public:

        /**
         * The profiling modus
         */
        typedef enum _Modus_t {
            PROFILE_NONE = 0,     //< profiles no calls at all
            PROFILE_SELECTED = 1, //< profiles selected calls
            PROFILE_ALL = 2       //< profiles all calls
        } Modus;

        /**
         * Answer the only instance of this class
         *
         * @return The only instance of this class
         */
        static Manager& Instance(void);

        /**
         * Sets the core instance object
         * This must be called before any other calls can succeed.
         *
         * @param ci The core instance object
         */
        inline void SetCoreInstance(CoreInstance *ci) {
            this->ci = ci;
        }

        /**
         * Gets the current profiling modus
         *
         * @return The current profiling modus
         */
        inline Modus GetModus(void) const {
            return this->modus;
        }

        /**
         * Sets the profiling modus.
         * Changing to 'selected' from any other modus will not add any calls to the selected calls!
         *
         * @param modus The new profiling modus
         */
        void SetModus(Modus modus);

        /**
         * Unselects all calls from being profiled
         */
        void UnselectAll(void);

        /**
         * Selects a specified call to being profiled
         *
         * @param caller The fully qualified name of the caller slot connected
         *               to the call to be profiled
         */
        void Select(const vislib::StringA& caller);

        /**
         * Adds a connection
         *
         * @param conn The connection to be added
         */
        void AddConnection(Connection::ptr_type conn);

        /**
         * Removes a connection
         *
         * @param conn The connection to be removed
         */
        void RemoveConnection(Connection::ptr_type conn);

    private:

        /** Hidden ctor */
        Manager(void);

        /** Hidden dtor */
        ~Manager(void);

        /** The current profiling modus */
        Modus modus;

        /** The core instance */
        CoreInstance *ci;

        /** The connections to call callbacks */
        vislib::Array<Connection::ptr_type> connections;

    };

} /* end namespace profiler */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PROFILER_MANAGER_H_INCLUDED */
