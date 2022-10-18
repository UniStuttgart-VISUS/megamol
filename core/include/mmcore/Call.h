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

#include "CallCapabilities.h"
#ifdef MEGAMOL_USE_PROFILING
#include "PerformanceManager.h"
#endif

#include <memory>
#include <string>
#include <utility>
#include <vector>

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
class Call : public std::enable_shared_from_this<Call> {
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
     * Calls function 'func'.
     *
     * @param func The function to be called.
     *
     * @return The return value of the function.
     */
    template <typename RET, typename REQ>
    std::shared_ptr<RET> operator()(unsigned int func, REQ const& req) {
        if (callee != nullptr) {
            return callee->InCall(funcMap[func], req);
        }
        return nullptr;
    }

    /**
     * Answers the callee slot this call is connected to.
     *
     * @return The callee slot this call is connected to.
     */
    inline const CalleeSlot* PeekCalleeSlot(void) const {
        return this->callee;
    }

    CalleeSlot* PeekCalleeSlotNoConst() const {
        return this->callee;
    }

    /**
     * Answers the caller slot this call is connected to.
     *
     * @return The caller slot this call is connected to.
     */
    inline const CallerSlot* PeekCallerSlot(void) const {
        return this->caller;
    }

    CallerSlot* PeekCallerSlotNoConst() const {
        return this->caller;
    }

    inline void SetClassName(const char* name) {
        this->className = name;
    }

    inline const char* ClassName() const {
        return this->className;
    }

    const CallCapabilities& GetCapabilities() const {
        return caps;
    }

    void SetCallbackNames(std::vector<std::string> names);

    const std::string& GetCallbackName(uint32_t idx) const;

    std::string GetDescriptiveText() const;

    uint32_t GetCallbackCount() const {
        return static_cast<uint32_t>(callback_names.size());
    }

private:
    /** The callee connected by this call */
    CalleeSlot* callee;

    /** The caller connected by this call */
    CallerSlot* caller;

    const char* className;

    /** The function id mapping */
    unsigned int* funcMap;

    /* Callback names for runtime introspection */
    std::vector<std::string> callback_names;

    inline static std::string err_out_of_bounds = "index out of bounds";

#ifdef MEGAMOL_USE_PROFILING
    // i cant make access to the queries work without making the Profiling_Service a friend class
    // and thereby linking the frontend service headers into the core
    // so make the perf queries public when profiling is active...
public:
    frontend_resources::PerformanceManager* perf_man = nullptr;
    frontend_resources::PerformanceManager::handle_vector cpu_queries, gl_queries;
#endif // MEGAMOL_USE_PROFILING
protected:
    CallCapabilities caps;
};


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALL_H_INCLUDED */
