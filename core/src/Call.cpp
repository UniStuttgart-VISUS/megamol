/*
 * Call.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/utility/log/Log.h"

#ifdef MEGAMOL_USE_TRACY
#include "tracy/Tracy.hpp"
#ifdef MEGAMOL_USE_OPENGL
#include "glad/gl.h"
#include "tracy/TracyOpenGL.hpp"
#endif
#endif
#if defined(MEGAMOL_USE_TRACY) || defined(MEGAMOL_USE_OPENGL_DEBUGGROUPS)
#include "mmcore/Module.h"
#endif

using namespace megamol::core;

/*
 * Call::Call
 */
Call::Call() : callee(nullptr), caller(nullptr), className(nullptr), funcMap(nullptr) {}


/*
 * Call::~Call
 */
Call::~Call() {
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
#if defined(MEGAMOL_USE_TRACY) || defined(MEGAMOL_USE_OPENGL_DEBUGGROUPS)
        auto f = this->callee->GetCallbackFuncName(func);
        auto parent = callee->Parent().get();
        std::string output = dynamic_cast<core::Module*>(parent)->ClassName();
        output += "::";
        output += f;
#endif
#ifdef MEGAMOL_USE_TRACY
        ZoneScoped;
        ZoneName(output.c_str(), output.size());
#ifdef MEGAMOL_USE_OPENGL
        TracyGpuZoneTransient(___tracy_gpu_zone, output.c_str(), caps.OpenGLRequired());
#endif
#endif
#ifdef MEGAMOL_USE_OPENGL_DEBUGGROUPS
        if (caps.OpenGLRequired()) {
            // let some service do it!
            gl_helper->PushDebugGroup(1234, -1, output.c_str());
        }
#endif
#ifdef MEGAMOL_USE_PROFILING
        perf_man->start_timer(cpu_queries[func]);
        if (caps.OpenGLRequired()) {
            perf_man->start_timer(gl_queries[func]);
        }
#endif
        res = this->callee->InCall(this->funcMap[func], *this);
#ifdef MEGAMOL_USE_PROFILING
        if (caps.OpenGLRequired()) {
            perf_man->stop_timer(gl_queries[func]);
        }
        perf_man->stop_timer(cpu_queries[func]);
#endif
#ifdef MEGAMOL_USE_OPENGL_DEBUGGROUPS
        if (caps.OpenGLRequired()) {
            gl_helper->PopDebugGroup();
        }
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
