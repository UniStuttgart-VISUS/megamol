/*
 * Call.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#ifdef MEGAMOL_USE_PROFILING
#include "mmcore/CoreInstance.h"
#endif
#include "mmcore/utility/log/Log.h"

using namespace megamol::core;

/*
 * Call::Call
 */
Call::Call(void) : callee(nullptr), caller(nullptr), className(nullptr), funcMap(nullptr) {}


/*
 * Call::~Call
 */
Call::~Call(void) {
    if (this->caller != nullptr) {
        CallerSlot* cr = this->caller;
        this->caller = nullptr; // DO NOT DELETE
        cr->ConnectCall(nullptr);
    }
    if (this->callee != nullptr) {
        this->callee->ConnectCall(nullptr);
        this->callee = nullptr; // DO NOT DELETE
    }
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("destructed call \"%s\"\n", typeid(*this).name());
    ARY_SAFE_DELETE(this->funcMap);
}


/*
 * Call::operator()
 */
bool Call::operator()(unsigned int func) {
    bool res = false;
    if (this->callee != nullptr) {
#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
        auto f = this->callee->GetCallbackFuncName(func);
        auto parent = callee->Parent().get();
        if (caps.OpenGLRequired()) {
            std::string output = dynamic_cast<core::Module*>(parent)->ClassName();
            output += "::";
            output += f;
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 1234, -1, output.c_str());
            // megamol::core::utility::log::Log::DefaultLog.WriteInfo("called %s::%s", p3->ClassName(), f);
        }
#endif
#ifdef MEGAMOL_USE_PROFILING
        const auto frameID = this->callee->GetCoreInstance()->GetFrameID();
        perf_man->start_timer(cpu_queries[func], frameID);
        if (caps.OpenGLRequired())
            perf_man->start_timer(gl_queries[func], frameID);
#endif
        res = this->callee->InCall(this->funcMap[func], *this);
#ifdef MEGAMOL_USE_PROFILING
        if (caps.OpenGLRequired())
            perf_man->stop_timer(gl_queries[func]);
        perf_man->stop_timer(cpu_queries[func]);
#endif
#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
        if (caps.OpenGLRequired())
            glPopDebugGroup();
#endif
    }
    // megamol::core::utility::log::Log::DefaultLog.WriteInfo("calling %s, idx %i, result %s (%s)", this->ClassName(), func,
    //    res ? "true" : "false", this->callee == nullptr ? "no callee" : "from callee");
    return res;
}

std::string Call::GetDescriptiveText() const {
    if (this->caller != nullptr && this->callee != nullptr) {
        return caller->FullName().PeekBuffer() + std::string("->") + callee->FullName().PeekBuffer();
    }
    return "";
}

void Call::SetCallbackNames(std::vector<std::string> names) {
    callback_names = std::move(names);
}

const std::string& Call::GetCallbackName(uint32_t idx) const {
    if (idx < callback_names.size()) {
        return callback_names[idx];
    } else {
        return err_out_of_bounds;
    }
}
