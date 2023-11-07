/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <mutex>
#include <string>

#include "LuaCallbacksCollection.h"
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>
#include "mmcore/MegaMolGraph.h"

struct lua_State; // lua includes should stay in the core

namespace megamol::core {

/**
 * This class holds a Lua state. It can be used to interact with a MegaMol instance.
 * No sandboxing is performed, the environment is complete with the exception of
 * print being redirected to the MegaMol log as information.
 * Lua constants LOGINFO, LOGWARNING, LOGERROR are provided for MegaMol log output.
 */
class LuaAPI {
public:
    LuaAPI();

    ~LuaAPI();

    // TODO forbid copy-contructor? assignment?

    /**
     * Run a script file.
     */
    sol::safe_function_result RunFile(const std::string& fileName);
    /**
     * Run a script file.
     */
    sol::safe_function_result RunFile(const std::wstring& fileName);
    /**
     * Run a script string.
     */
    sol::safe_function_result RunString(const std::string& script, std::string scriptPath = "");

    /**
     * Answers the current project file path
     */
    std::string GetScriptPath();

    /**
     * Sets the current project file path
     */
    void SetScriptPath(std::string const& scriptPath);

    void AddCallbacks(megamol::frontend_resources::LuaCallbacksCollection const& callbacks);
    void RemoveCallbacks(
        megamol::frontend_resources::LuaCallbacksCollection const& callbacks, bool delete_verbatim = true);
    void RemoveCallbacks(std::vector<std::string> const& callback_names);
    void ClearCallbacks();

protected:
    // ** MegaMol API provided for configuration / startup

    /** mmGetBithWidth get bits of executable (integer) */
    static unsigned int GetBitWidth();

    /** mmGetConfiguration: get compile configuration (debug, release) */
    static std::string GetCompileMode();

    /** mmGetOS: get operating system (windows, linux, unknown) */
    static std::string GetOS();

    /** mmGetMachineName: get machine name */
    static std::string GetMachineName();

    /**
     * mmGetEnvValue(string name): get the value of environment variable 'name'
     */
    static std::string GetEnvValue(const std::string& variable);

    /** answer the ProcessID of the running MegaMol */
    static unsigned int GetProcessID();

    int ReadTextFile(lua_State* L);
    int WriteTextFile(lua_State* L);

    /** expose current script path to lua */
    std::string CurrentScriptPath();

private:
    /** all of the Lua startup code */
    void commonInit();

    /** the one Lua state */
    sol::state luaApiInterpreter_;

    std::list<megamol::frontend_resources::LuaCallbacksCollection> verbatim_lambda_callbacks_;
    std::list<std::tuple<std::string, std::function<int(lua_State*)>>> wrapped_lambda_callbacks_;
    void register_callbacks(megamol::frontend_resources::LuaCallbacksCollection& callbacks);

    /** no two threads must interfere with the reentrant L */
    std::mutex stateLock;

    std::string currentScriptPath = "";
};

} // namespace megamol::core
