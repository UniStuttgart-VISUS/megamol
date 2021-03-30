/*
 * LuaAPI.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_LUAAPI_H_INCLUDED
#define MEGAMOLCORE_LUAAPI_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <mutex>
#include <string>
#include "LuaInterpreter.h"
#include "mmcore/MegaMolGraph.h"

#include "LuaCallbacksCollection.h"

struct lua_State; // lua includes should stay in the core

namespace megamol {
namespace core {

/**
 * This class holds a Lua state. It can be used to interact with a MegaMol instance.
 * For sandboxing, a standard environment megamol_env is provided that only
 * allows lua/lib calls that are considered safe and additionally redirects
 * print() to the MegaMol log output. By default only loads base, coroutine,
 * string, table, math, package, and os (see LUA_FULL_ENVIRONMENT define).
 * Lua constants LOGINFO, LOGWARNING, LOGERROR are provided for MegaMol log output.
 */
class MEGAMOLCORE_API LuaAPI {
public:
    static const std::string MEGAMOL_ENV;

    /**
     * @param imperativeOnly choose whether only reply-less commands will be made available
     * to avoid having round-trips across frames/threads etc. Basically config/project scripts
     * are reply-less and the LuaHost can get replies.
     */
    LuaAPI();

    ~LuaAPI();

    // TODO forbid copy-contructor? assignment?

    /**
     * Run a script file, sandboxed in the environment provided.
     */
    bool RunFile(const std::string& envName, const std::string& fileName, std::string& result);
    /**
     * Run a script file, sandboxed in the environment provided.
     */
    bool RunFile(const std::string& envName, const std::wstring& fileName, std::string& result);
    /**
     * Run a script string, sandboxed in the environment provided.
     */
    bool RunString(
        const std::string& envName, const std::string& script, std::string& result, std::string scriptPath = "");

    /**
     * Run a script file, sandboxed in the standard megamol_env.
     */
    bool RunFile(const std::string& fileName, std::string& result);
    /**
     * Run a script file, sandboxed in the standard megamol_env.
     */
    bool RunFile(const std::wstring& fileName, std::string& result);
    /**
     * Run a script string, sandboxed in the standard megamol_env.
     */
    bool RunString(const std::string& script, std::string& result, std::string scriptPath = "");

    /**
     * Answer whether the wrapped lua state is valid
     */
    bool StateOk();

    /**
     * Answers the current project file path
     */
    std::string GetScriptPath(void);

    /**
     * Sets the current project file path
     */
    void SetScriptPath(std::string const& scriptPath);

    void AddCallbacks(megamol::frontend_resources::LuaCallbacksCollection const& callbacks);
    void RemoveCallbacks(megamol::frontend_resources::LuaCallbacksCollection const& callbacks, bool delete_verbatim = true);
    void RemoveCallbacks(std::vector<std::string> const& callback_names);
    void ClearCallbacks();

protected:

    // ** MegaMol API provided for configuration / startup

    /** mmGetBithWidth get bits of executable (integer) */
    int GetBitWidth(lua_State* L);

    /** mmGetConfiguration: get compile configuration (debug, release) */
    int GetCompileMode(lua_State* L);

    /** mmGetOS: get operating system (windows, linux, unknown) */
    int GetOS(lua_State* L);

    /** mmGetMachineName: get machine name */
    int GetMachineName(lua_State* L);

    /**
     * mmGetEnvValue(string name): get the value of environment variable 'name'
     */
    int GetEnvValue(lua_State* L);

    /** answer the ProcessID of the running MegaMol */
    int GetProcessID(lua_State* L);

    int ReadTextFile(lua_State* L);
    int WriteTextFile(lua_State* L);

    int CurrentScriptPath(lua_State* L);

    int Invoke(lua_State* L);

private:

    /** all of the Lua startup code */
    void commonInit();

    /** gets a string from the stack position i. returns false if it's not a string */
    //bool getString(int i, std::string& out);

    /** the one Lua state */
    LuaInterpreter<LuaAPI> luaApiInterpreter_;

    std::list<megamol::frontend_resources::LuaCallbacksCollection> verbatim_lambda_callbacks_;
    std::list<std::tuple<std::string, std::function<int(lua_State*)>>> wrapped_lambda_callbacks_;
    void register_callbacks(megamol::frontend_resources::LuaCallbacksCollection& callbacks);

    /** no two threads must interfere with the reentrant L */
    std::mutex stateLock;

    std::string currentScriptPath = "";
};

} /* namespace core */
} /* namespace megamol */
#endif /* MEGAMOLCORE_LUAAPI_H_INCLUDED */
