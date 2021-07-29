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
#ifdef PROFILING
#include <vector>
#include <array>
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
        inline double GetLastCPUTime(uint32_t func) const {
            if (func < last_cpu_time.size())
                return last_cpu_time[func];
            else
                return -1.0;
        }
        inline double GetAverageCPUTime(uint32_t func) const {
            if (func < avg_cpu_time.size())
                return avg_cpu_time[func];
            else
                return -1.0;
        }
        inline uint32_t GetNumCPUSamples(uint32_t func) const {
            if (func < num_cpu_time_samples.size())
                return num_cpu_time_samples[func];
            else
                return 0;
        }
        inline double GetLastGPUTime(uint32_t func) const {
            if (func < last_gpu_time.size())
                return last_gpu_time[func];
            else
                return -1.0;
        }
        inline double GetAverageGPUTime(uint32_t func) const {
            if (func < avg_gpu_time.size())
                return avg_gpu_time[func];
            else
                return -1.0;
        }
        inline uint32_t GetNumGPUSamples(uint32_t func) const {
            if (func < num_gpu_time_samples.size())
                return num_gpu_time_samples[func];
            else
                return 0;
        }

        inline uint32_t GetFuncCount() const {
            /// XXX assert(last_cpu_time.size() == avg_cpu_time.size() == num_cpu_time_samples.size() == last_gpu_time.size() ==
            ///       avg_gpu_time.size() == num_gpu_time_samples.size());
            return static_cast<uint32_t>(last_cpu_time.size());
        }
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
        std::vector<double> last_cpu_time;
        std::vector<double> avg_cpu_time;
        std::vector<uint32_t> num_cpu_time_samples;

        std::vector<double> last_gpu_time;
        std::vector<double> avg_gpu_time;
        std::vector<uint32_t> num_gpu_time_samples;

        class my_query_id {
        public:
            my_query_id();
            ~my_query_id();
            my_query_id(const my_query_id&);
            uint32_t Get() const {
                return the_id;
            }
            bool Started() const {
                return started;
            }
            void Start() {
                started = true;
            }
        private:
            uint32_t the_id = 0;
            bool started = false;
        };
        std::array<std::vector<my_query_id>, 2> queries;
        uint32_t query_start_buffer = 1;
        uint32_t query_read_buffer = 0;
#endif PROFILING

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALL_H_INCLUDED */
