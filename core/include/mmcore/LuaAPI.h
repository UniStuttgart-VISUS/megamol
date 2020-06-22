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

struct lua_State; // lua includes should stay in the core

namespace megamol {
namespace core {
namespace utility {
class Configuration;
}


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
    LuaAPI(megamol::core::MegaMolGraph &graph, bool imperativeOnly);

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

    // ************************************************************
    // Lua interface routines, published to Lua as mm<name>
    // ************************************************************

protected:

    // ** MegaMol API provided for configuration / startup

    /** mmGetBithWidth get bits of executable (integer) */
    int GetBitWidth(lua_State* L);

    /** mmGetConfiguration: get compile configuration (debug, release) */
    int GetConfiguration(lua_State* L);

    /** mmGetOS: get operating system (windows, linux, unknown) */
    int GetOS(lua_State* L);

    /** mmGetMachineName: get machine name */
    int GetMachineName(lua_State* L);

    /** mmSetAppDir(string path) */
    int SetAppDir(lua_State* L);

    /**
     * mmAddShaderDir(string path): add path for searching shaders
     * and .btf files.
     */
    int AddShaderDir(lua_State* L);

    /**
     * mmAddResourceDir(string path): add path for searching generic
     * resources.
     */
    int AddResourceDir(lua_State* L);

    /**
     * mmPluginLoaderInfo(string path, string fileglob,
     * string action): action = ('include', 'exclude').
     * Add information about plugins to load: set a search path
     * plus a globbing pattern for choosing relevant libraries
     * to load as plugins.
     */
    int PluginLoaderInfo(lua_State* L);

    /** mmSetLogFile(string path): set path of the log file. */
    int SetLogFile(lua_State* L);

    /**
     * mmSetLogLevel(string level): set log level of the log file.
     * level = ('error', 'warn', 'warning', 'info', 'none', 'null',
     * 'zero', 'all', '*')
     */
    int SetLogLevel(lua_State* L);

    /** mmSetEchoLevel(string level): set level of console output, see SetLogLevel. */
    int SetEchoLevel(lua_State* L);

    /**
     * mmSetConfigValue(string name, string value): set configuration value 'name'
     * to 'value'.
     */
    int SetConfigValue(lua_State* L);

    /**
     * mmGetConfigValue(string name): get the value of configuration value 'name'
     */
    int GetConfigValue(lua_State* L);

    /**
     * mmGetEnvValue(string name): get the value of environment variable 'name'
     */
    int GetEnvValue(lua_State* L);

    // ** MegaMol API provided for runtime manipulation / Configurator live connection

    /** answer the ProcessID of the running MegaMol */
    int GetProcessID(lua_State* L);

    /**
     * mmGetModuleParams(string name): list all parameters of a module
     * along with their description, type and value.
     */
    int GetModuleParams(lua_State* L);

    /**
     * mmGetParamDescription(string name): get the description of a specific parameter.
     */
    int GetParamDescription(lua_State* L);

    /**
     * mmGetParamValue(string name): get the value of a specific parameter.
     */
    int GetParamValue(lua_State* L);

    /**
     * mmSetParamValue(string name, string value):
     * set the value of a specific parameter.
     */
    int SetParamValue(lua_State* L);

    int CreateParamGroup(lua_State* L);
    int SetParamGroupValue(lua_State* L);

    int CreateModule(lua_State* L);
    int DeleteModule(lua_State* L);
    int CreateCall(lua_State* L);
    int CreateChainCall(lua_State* L);
    int DeleteCall(lua_State* L);

    int QueryModuleGraph(lua_State* L);
    int ListCalls(lua_State* L);
    int ListModules(lua_State* L);
    int ListInstatiations(lua_State* L);
    int ListParameters(lua_State* L);

    int Help(lua_State* L);
    int Quit(lua_State* L);

    int ReadTextFile(lua_State* L);

    int Flush(lua_State* L);
    int CurrentScriptPath(lua_State* L);

    int Invoke(lua_State* L);

private:

    /** all of the Lua startup code */
    void commonInit();

    /**
     * shorthand to ask graph for the param slot, returned in 'out'.
     * WARNING: assumes the graph is ALREADY LOCKED!
     */
    bool getParamSlot(const std::string routine, const char* paramName, core::param::ParamSlot** out);

    /** gets a string from the stack position i. returns false if it's not a string */
    //bool getString(int i, std::string& out);

    /** interpret string log levels */
    static UINT parseLevelAttribute(const std::string attr);

    inline static bool iequals(const std::string& one, const std::string& other) {

        return ((one.size() == other.size()) &&
                std::equal(one.begin(), one.end(), other.begin(),
                    [](const char& c1, const char& c2) { return (c1 == c2 || std::toupper(c1) == std::toupper(c2)); }));
    }

    /** the one Lua state */
    LuaInterpreter<LuaAPI> luaApiInterpreter_;

    /** the respective MegaMol graph */
    megamol::core::MegaMolGraph& graph_;

    /** no two threads must interfere with the reentrant L */
    std::mutex stateLock;

    std::string currentScriptPath = "";

    bool imperative_only_;
};

} /* namespace core */
} /* namespace megamol */
#endif /* MEGAMOLCORE_LUAAPI_H_INCLUDED */