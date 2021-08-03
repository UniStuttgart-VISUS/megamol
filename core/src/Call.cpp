/*
 * Call.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/RigRendering.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
#    include "mmcore/view/Renderer2DModule.h"
#    include "mmcore/view/Renderer3DModule.h"
#    include "mmcore/view/Renderer3DModuleGL.h"
#    include "vislib/graphics/gl/IncludeAllGL.h"
#endif
#ifdef PROFILING
#    include "mmcore/view/Renderer3DModuleGL.h"
#    include "mmcore/view/Renderer2DModule.h"
#    include <ctime>
#endif
#include "mmcore/utility/log/Log.h"

using namespace megamol::core;

/*
 * Call::Call
 */
Call::Call(void) : callee(nullptr), caller(nullptr), className(nullptr), funcMap(nullptr) {
    // intentionally empty
}


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
    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO + 350, "destructed call \"%s\"\n", typeid(*this).name());
    ARY_SAFE_DELETE(this->funcMap);
}


#ifdef PROFILING
Call::my_query_id::my_query_id() {
    glGenQueries(1, &the_id);
}
Call::my_query_id::~my_query_id() {
    glDeleteQueries(1, &the_id);
}
Call::my_query_id::my_query_id(const my_query_id&) {
    glGenQueries(1, &the_id);
}
#endif


/*
 * Call::operator()
 */
bool Call::operator()(unsigned int func) {
    bool res = false;
    if (this->callee != nullptr) {
#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
        auto f = this->callee->GetCallbackFuncName(func);
        auto parent = callee->Parent().get();
        auto p3 = dynamic_cast<core::view::Renderer3DModule*>(parent);
        auto p3_2 = dynamic_cast<core::view::Renderer3DModuleGL*>(parent);
        auto p2 = dynamic_cast<core::view::Renderer2DModule*>(parent);
        if (p3 || p3_2 || p2) {
            std::string output = dynamic_cast<core::Module*>(parent)->ClassName();
            output += "::";
            output += f;
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 1234, -1, output.c_str());
            // megamol::core::utility::log::Log::DefaultLog.WriteInfo("called %s::%s", p3->ClassName(), f);
        }
#endif
#ifdef PROFILING
        const std::clock_t c_start = std::clock();
        if (uses_gl) {
            // you can only have one query per target in flight. no idea what to do really.
            
            //queries[query_start_buffer].resize(paranoid_size);
            //queries[query_read_buffer].resize(paranoid_size);
            //auto& q = queries[query_start_buffer][func];
            //glBeginQuery(GL_TIME_ELAPSED, q.Get());
            //q.Start();
        }
#endif
        res = this->callee->InCall(this->funcMap[func], *this);
#ifdef PROFILING
        const std::clock_t c_end = std::clock();
        last_cpu_time[func] = 1000.0 * (static_cast<double>(c_end-c_start) / CLOCKS_PER_SEC);
        const auto total = (avg_cpu_time[func] * num_cpu_time_samples[func] + last_cpu_time[func]);
        num_cpu_time_samples[func]++;
        avg_cpu_time[func] = total / static_cast<double>(num_cpu_time_samples[func]);
        if (uses_gl) {
            //GLuint64 time;
            //auto& q = queries[query_read_buffer][func];
            //if (q.Started()) {
            //    int done = 0;
            //    glGetQueryObjectiv(q.Get(), GL_QUERY_RESULT_AVAILABLE, &done);
            //    ASSERT(done);
            //    glGetQueryObjectui64v(q.Get(), GL_QUERY_RESULT, &time);
            //    last_gpu_time[func] = static_cast<double>(time / 1000000.0);
            //    avg_gpu_time[func] = (avg_gpu_time[func] * num_gpu_time_samples[func] + last_gpu_time[func]) / ++num_gpu_time_samples[func];
            //}
            //std::swap(query_start_buffer, query_read_buffer);
        }
#endif
#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
        if (p2 || p3 || p3_2) glPopDebugGroup();
#endif
    }
    // megamol::core::utility::log::Log::DefaultLog.WriteInfo("calling %s, idx %i, result %s (%s)", this->ClassName(), func,
    //    res ? "true" : "false", this->callee == nullptr ? "no callee" : "from callee");
    return res;
}
