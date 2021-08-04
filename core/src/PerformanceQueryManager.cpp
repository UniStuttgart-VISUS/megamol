#ifdef PROFILING
#include <cassert>
#include <ctime>
#include "mmcore/Call.h"
#include "mmcore/PerformanceQueryManager.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

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
    starting_func = (starting_func + 1) % all_calls[starting_call]->GetFuncCount();
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
        next_query = (next_query + 1) % 2;
    }
    return query_infos[idx].started;
}

void PerformanceQueryManager::Stop(uint32_t frameId) {
    glEndQuery(GL_TIME_ELAPSED);
}

void PerformanceQueryManager::Collect() {
    // collect next query because that is what will be used next frame and needs to be freed up!
    const uint32_t the_query = next_query;
    auto& qi = query_infos[the_query];
    if (qi.started) {
        int done = 0;
        GLuint64 time;
        const auto oid = qi.id;
        glGetQueryObjectiv(oid, GL_QUERY_RESULT_AVAILABLE, &done);
        if (!done) {
            assert(done);
        }
        glGetQueryObjectui64v(oid, GL_QUERY_RESULT, &time);
        const auto the_func = qi.func_idx;
        const auto the_call = qi.call_idx;
        qi.started = false;
        auto c = all_calls[the_call];
        c->last_gpu_time[the_func] = static_cast<double>(time / 1000000.0);
        const auto total = c->avg_gpu_time[the_func] * c->num_gpu_time_samples[the_func] + c->last_gpu_time[the_func];
        c->num_gpu_time_samples[the_func]++;
        c->avg_gpu_time[the_func] = total / c->num_gpu_time_samples[the_func];
    }
    AdvanceGLProfiling(); // important! regardless of whether the last call was actually profiled!
}


#endif

