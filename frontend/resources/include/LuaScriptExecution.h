/*
 * LuaScriptExecution.h
 *
 * Copyright (C) 2022 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "LuaScriptExecution.h"

#include <chrono>
#include <functional>
#include <future>
#include <list>
#include <string>

namespace megamol {
namespace frontend_resources {

static std::string LuaScriptExecution_Req_Name = "LuaScriptExecution";

using LuaExecutionResult = std::tuple<bool, std::string>;
using LuaFutureExecutionResult = std::future<LuaExecutionResult>;
using LuaExecutionResultCallback = std::function<void(LuaExecutionResult const&)>;

using lua_execution_func = std::function<LuaExecutionResult(std::string const&)>;
using lua_deferred_execution_func =
    std::function<LuaFutureExecutionResult(std::string const& /*script*/, std::string const& /*script path*/)>;
using lua_deferred_execution_callback_func = std::function<void(
    std::string const& /*script*/, std::string const& /*script path*/, LuaExecutionResultCallback const&)>;

// one may execute lua script immediately or at the start of the next frame (deferred)
struct LuaScriptExecution {
    // calls the lua interpreter immediately. this is only safe when not inside the rendering of a frame, i.e. inside megamol graph execution.
    lua_execution_func execute_immediate;

    // queues lua script execution for beginning of next frame, executed by lua service at beginning of next frame cycle (but after network lua requests)
    // this is safe to execute from inside megamol graph execution as it will not immediately alter graph state from lua.
    lua_deferred_execution_func execute_deferred;

    // queues lua script for execution, along with a callback function which gets called after execution of the lua script
    lua_deferred_execution_callback_func execute_deferred_callback;
};

// holds a set of futures of LuaFutureExecutionResult, helping you to check the futures for newly available results
struct LuaFutureExecutionResult_Helper {
    std::list<LuaFutureExecutionResult> futures;

    // adds a future to list of pending future results
    void add(LuaFutureExecutionResult& future) {
        futures.push_back(std::move(future));
    }

    // calls 'callback' function providing the LuaExecutionResult of futures containing a result
    // futures which provided a result will be removed from the list of pending future results
    void check_all(LuaExecutionResultCallback const& callback) {
        using namespace std::chrono_literals;

        auto first = futures.begin();
        while (first != futures.end()) {
            auto& future = *first;
            if (future.wait_for(0ms) == std::future_status::ready) {
                callback(future.get());
                auto old = first++;
                futures.erase(old);
            } else {
                first++;
            }
        }
    }

    void clear() {
        futures.clear();
    }
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
