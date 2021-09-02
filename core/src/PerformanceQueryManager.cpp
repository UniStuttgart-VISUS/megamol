#ifdef PROFILING
#include <cassert>
#include <ctime>
#include "mmcore/Call.h"
#include "mmcore/PerformanceQueryManager.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol::core;

PerformanceQueryManager::PerformanceQueryManager() {
    glGenQueries(1, &query_infos[0].id);
    glGenQueries(1, &query_infos[1].id);
}
PerformanceQueryManager::~PerformanceQueryManager() {
    glDeleteQueries(1, &query_infos[0].id);
    glDeleteQueries(1, &query_infos[1].id);
}
PerformanceQueryManager::PerformanceQueryManager(const PerformanceQueryManager&) {
    glGenQueries(1, &query_infos[0].id);
    glGenQueries(1, &query_infos[1].id);
}

void PerformanceQueryManager::AddCall(megamol::core::Call* c) {
    all_calls.push_back(c);
    ResetGLProfiling();
}

void PerformanceQueryManager::RemoveCall(megamol::core::Call* c) {
    all_calls.erase(std::remove(all_calls.begin(), all_calls.end(), c));
    ResetGLProfiling();
}

void PerformanceQueryManager::ResetGLProfiling() {
    if (!all_calls.empty()) {
        starting_call = 0;
        starting_func = 0;
    } else {
        starting_call = -1;
        starting_func = -1;
    }
}

void PerformanceQueryManager::AdvanceGLProfiling() {
    starting_func = (starting_func + 1) % all_calls[starting_call]->profiling.GetFuncCount();
    if (starting_func == 0) {
        // we wrapped, advance to next call!
        starting_call = (starting_call + 1) % all_calls.size();
    }
}


bool PerformanceQueryManager::Start(Call* c, uint32_t frameId, int32_t funcIdx) {
    const auto idx = next_query;
    if (c == all_calls[starting_call] && funcIdx == starting_func && !query_infos[idx].started) {
        glBeginQuery(GL_TIME_ELAPSED, query_infos[idx].id);
        query_infos[idx].started = true;
        query_infos[idx].call_idx = starting_call;
        query_infos[idx].func_idx = starting_func;
        running_query = idx;
        next_query = (next_query + 1) % 2;
        //utility::log::Log::DefaultLog.WriteInfo("[PerfQueryMan] frame %u: started GL query for call %u, func %u in query %u",
        //    frameId, starting_call, starting_func, running_query);
    }
    return query_infos[idx].started;
}

void PerformanceQueryManager::Stop(uint32_t frameId) {
    if (running_query > -1 && query_infos[running_query].started) {
        auto& qi = query_infos[running_query];
        glEndQuery(GL_TIME_ELAPSED);
        //utility::log::Log::DefaultLog.WriteInfo("[PerfQueryMan] frame %u: stopped GL query for call %u, func %u in query %u",
        //    frameId, qi.call_idx, qi.func_idx, running_query);
        running_query = -1;
    }
}

void PerformanceQueryManager::Collect() {
    // collect next query because that is what will be used next frame and needs to be freed up!
    const uint32_t the_query = next_query;
    auto& qi = query_infos[the_query];
    if (qi.started) {
        //utility::log::Log::DefaultLog.WriteInfo("[PerfQueryMan] frame %u: collecting GL query for call %u, func %u in query %u",
        //    0, qi.call_idx, qi.func_idx, the_query);
        int done = 0;
        GLuint64 time;
        const auto oid = qi.id;
        do {
            // for some reason, one frame is not enough for the query to come back, so we really need to block.
            glGetQueryObjectiv(oid, GL_QUERY_RESULT_AVAILABLE, &done);
        } while (!done);
        glGetQueryObjectui64v(oid, GL_QUERY_RESULT, &time);
        const auto the_func = qi.func_idx;
        const auto the_call = qi.call_idx;
        qi.started = false;
        auto c = all_calls[the_call];
        c->profiling.gpu_history[the_func].push_value(static_cast<double>(time) / 1000000.0);
    }
    AdvanceGLProfiling(); // important! regardless of whether the last call was actually profiled! we need to advance
                          // through the graph though
}


#endif

