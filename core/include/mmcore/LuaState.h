/*
 * LuaState.h
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_LUASTATE_H_INCLUDED
#define MEGAMOLCORE_LUASTATE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <string>
#include "mmcore/param/ParamSlot.h"
#include "mmcore/ViewInstance.h"
#include "mmcore/JobInstance.h"
#include <mutex>

struct lua_State; // lua includes should stay in the core

namespace megamol {
namespace core {
namespace utility {
    class Configuration;
}
    class CoreInstance;

    /**
     * This class holds a Lua state. It can be used at configuration
     * time or with a running MegaMol instance. It provides an appropriate
     * interface in both cases, on the one hand allowing an imperative
     * setting of configuration values based on the same runtime properties
     * as the traditional configuration (bits, os, machine name, ...), and
     * on the other allowing the same runtime settings as the MegaMol frontend
     * and project files.
     * For sandboxing, a standard environment megamol_env is provided that only
     * allows lua/lib calls that are considered safe and additionally redirects
     * print() to the MegaMol log output. By default only loads base, coroutine,
     * string, table, math, package, and os (see LUA_FULL_ENVIRONMENT define).
     * Lua constants LOGINFO, LOGWARNING, LOGERROR are provided for MegaMol log output.
     */
    class MEGAMOLCORE_API LuaState {
    public:

        static const std::string MEGAMOL_ENV;

        /** ctor for runtime */
        LuaState(CoreInstance *inst);

        /** ctor for startup */
        LuaState(utility::Configuration *conf);

        ~LuaState();
        
        // TODO forbid copy-contructor? assignment?

        /**
         * Load an environment from a file. The environment is a table
         * of the form "env = { alias = function, libraryname = { ... }, ...}"
         * that should be used to sandbox scripts that are run afterwards.
         */
        bool LoadEnviromentFile(const std::string& fileName);

        /** Load an environment from a string. See LoadEnvironmentFile. */
        bool LoadEnviromentString(const std::string& envString);

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
        bool RunString(const std::string& envName, const std::string& script, std::string& result, std::string scriptPath = "");
        
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

        /**
         * mmLog(int level, ...): Output to MegaMol log.
         */
        int Log(lua_State *L);

        /** mmLogInfo: alias for mmLog(LOGINFO, ...) needed for sandboxing print */
        int LogInfo(lua_State *L);

        // ** MegaMol API provided for configuration / startup

        /** mmGetBithWidth get bits of executable (integer) */
        int GetBitWidth(lua_State *L);

        /** mmGetConfiguration: get compile configuration (debug, release) */
        int GetConfiguration(lua_State *L);

        /** mmGetOS: get operating system (windows, linux, unknown) */
        int GetOS(lua_State *L);

        /** mmGetMachineName: get machine name */
        int GetMachineName(lua_State *L);

        /** mmSetAppDir(string path) */
        int SetAppDir(lua_State *L);
        
        /**
         * mmAddShaderDir(string path): add path for searching shaders
         * and .btf files.
         */
        int AddShaderDir(lua_State *L);

        /**
         * mmAddResourceDir(string path): add path for searching generic
         * resources.
         */
        int AddResourceDir(lua_State *L);

        /**
         * mmPluginLoaderInfo(string path, string fileglob,
         * string action): action = ('include', 'exclude').
         * Add information about plugins to load: set a search path
         * plus a globbing pattern for choosing relevant libraries
         * to load as plugins.
         */
        int PluginLoaderInfo(lua_State *L);

        /** mmSetLogFile(string path): set path of the log file. */
        int SetLogFile(lua_State *L);

        /**
         * mmSetLogLevel(string level): set log level of the log file.
         * level = ('error', 'warn', 'warning', 'info', 'none', 'null',
         * 'zero', 'all', '*')
         */
        int SetLogLevel(lua_State *L);

        /** mmSetEchoLevel(string level): set level of console output, see SetLogLevel. */
        int SetEchoLevel(lua_State *L);

        /**
         * mmSetConfigValue(string name, string value): set configuration value 'name'
         * to 'value'.
         */
        int SetConfigValue(lua_State *L);

        /**
         * mmGetConfigValue(string name): get the value of configuration value 'name'
         */
        int GetConfigValue(lua_State *L);

        /**
         * mmGetEnvValue(string name): get the value of environment variable 'name'
         */
        int GetEnvValue(lua_State *L);

        // ** MegaMol API provided for runtime manipulation / Configurator live connection

        /** answer the ProcessID of the running MegaMol */
        int GetProcessID(lua_State *L);

        /**
         * mmGetModuleParams(string name): list all parameters of a module
         * along with their description, type and value.
         */
        int GetModuleParams(lua_State *L);

        /**
         * mmGetParamType(string name): get the type of a specific parameter.
         */
        int GetParamType(lua_State *L);

        /**
        * mmGetParamDescription(string name): get the description of a specific parameter.
        */
        int GetParamDescription(lua_State *L);

        /**
         * mmGetParamValue(string name): get the value of a specific parameter.
         */
        int GetParamValue(lua_State *L);

        /**
         * mmSetParamValue(string name, string value):
         * set the value of a specific parameter.
         */
        int SetParamValue(lua_State *L);

        int CreateParamGroup(lua_State *L);
        int SetParamGroupValue(lua_State* L);

        int CreateModule(lua_State *L);
        int DeleteModule(lua_State *L);
        int CreateCall(lua_State *L);
        int CreateChainCall(lua_State* L);
        int DeleteCall(lua_State *L);
        int CreateJob(lua_State *L);
        int DeleteJob(lua_State *L);
        int CreateView(lua_State *L);
        int DeleteView(lua_State *L);

        int QueryModuleGraph(lua_State *L);
        int ListCalls(lua_State *L);
        int ListModules(lua_State *L);
        int ListInstatiations(lua_State *L);
        int ListParameters(lua_State *L);

        int Help(lua_State *L);
        int Quit(lua_State *L);

        int ReadTextFile(lua_State* L);

        int Flush(lua_State* L);
        int CurrentScriptPath(lua_State* L);

    private:

        /** error handler */
        void consumeError(int error, char const* file, int line) const;

        /** print table on the stack somewhat */
        void printTable(lua_State *L, std::stringstream& out);

        /** answer whether LuaState is instanced for configuration */
        bool checkConfiguring(const std::string where);

        /** answer whether LuaState is instanced for runtime */
        bool checkRunning(const std::string where);

        /** all of the Lua startup code */
        void commonInit();

        /**
         * shorthand to ask graph for the param slot, returned in 'out'.
         * WARNING: assumes the graph is ALREADY LOCKED!
         */
        bool getParamSlot(const std::string routine, const char *paramName,
            core::param::ParamSlot **out);

        // TODO: possibly not needed
        bool getView(const std::string routine, const char *viewName,
            core::ViewInstance **out);

        // TODO: possibly not needed
        bool getJob(const std::string routine, const char *jobName,
            core::JobInstance **out);

        /** gets a string from the stack position i. returns false if it's not a string */
        bool getString(int i, std::string& out);

        /** print the stack somewhat */
        void printStack();

        /** interpret string log levels */
        static UINT parseLevelAttribute(const std::string attr);

        /** the one Lua state */
        lua_State *L;

        /** the respective core instance, if runtime */
        CoreInstance *coreInst;

        /** the respective configuration, if startup */
        utility::Configuration *conf;

        /** no two threads must interfere with the reentrant L */
        std::mutex stateLock;

        std::string currentScriptPath = "";
    };

} /* namespace core */
} /* namespace megamol */
#endif /* MEGAMOLCORE_LUASTATE_H_INCLUDED */