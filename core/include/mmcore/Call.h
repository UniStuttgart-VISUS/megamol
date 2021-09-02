/*
 * Call.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALL_H_INCLUDED
#define MEGAMOLCORE_CALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <memory>
#include <utility>
#ifdef PROFILING
#include "CallProfiling.h"
#endif

#include "mmcore/api/MegaMolCore.std.h"

namespace megamol {
namespace core {

    /** Forward declaration of description and slots */
    class CalleeSlot;
    class CallerSlot;
    namespace factories {
        class CallDescription;
    }


    /**
     * Base class of rendering graph calls
     */
    class MEGAMOLCORE_API Call : public std::enable_shared_from_this<Call> {
    public:

        /** The description generates the function map */
        friend class ::megamol::core::factories::CallDescription;

        /** Callee slot is allowed to map functions */
        friend class CalleeSlot;

        /** The caller slot registeres itself in the call */
        friend class CallerSlot;

        /** Shared ptr type alias */
        using ptr_type = std::shared_ptr<Call>;

        /** Shared ptr type alias */
        using const_ptr_type = std::shared_ptr<const Call>;

        /** Weak ptr type alias */
        using weak_ptr_type = std::weak_ptr<Call>;

        /** Ctor. */
        Call(void);

        /** Dtor. */
        virtual ~Call(void);

        /**
         * Calls function 'func'.
         *
         * @param func The function to be called.
         *
         * @return The return value of the function.
         */
        bool operator()(unsigned int func = 0);

        /**
         * Answers the callee slot this call is connected to.
         *
         * @return The callee slot this call is connected to.
         */
        inline const CalleeSlot * PeekCalleeSlot(void) const {
            return this->callee;
        }

        CalleeSlot* PeekCalleeSlotNoConst() const { return this->callee; }

        /**
         * Answers the caller slot this call is connected to.
         *
         * @return The caller slot this call is connected to.
         */
        inline const CallerSlot * PeekCallerSlot(void) const {
            return this->caller;
        }

        CallerSlot* PeekCallerSlotNoConst() const { return this->caller; }

        inline void SetClassName(const char *name) {
            this->className = name;
        }

        inline const char * ClassName() const {
            return this->className;
        }

#ifdef PROFILING
        bool UsesGL() { return uses_gl; }
        const CallProfiling& GetProfiling() const { return profiling; }
#endif

    private:

        /** The callee connected by this call */
        CalleeSlot *callee;

        /** The caller connected by this call */
        CallerSlot *caller;

        const char *className;

        /** The function id mapping */
        unsigned int *funcMap;

#ifdef PROFILING
        friend class MegaMolGraph;
        friend class PerformanceQueryManager;

        CallProfiling profiling;

        void setProfilingInfo(std::vector<std::string> names, bool usesGL) {
            uses_gl = usesGL;
            profiling.setProfilingInfo(std::move(names), this);
        }

        bool uses_gl = false;
#endif //PROFILING

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALL_H_INCLUDED */
