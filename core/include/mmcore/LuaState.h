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
     * (PLANNED - TODO) and project files (PLANNED - TODO).
     * For sandboxing, a standard environment megamol_env is provided that only
     * allows lua/lib calls that are considered safe and additionally redirects
     * print() to the MegaMol log output. By default only loads base, coroutine,
     * string, table, math, and os (see LUA_FULL_ENVIRONMENT define).
     * Lua constants LOGINFO, LOGWARNING, LOGERROR are provided for MegaMol log output.
     */
    class LuaState {
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
        bool RunFile(const std::string& envName, const std::string& fileName);
        /**
         * Run a script file, sandboxed in the environment provided.
         */
        bool RunFile(const std::string& envName, const std::wstring& fileName);
        /**
         * Run a script string, sandboxed in the environment provided.
         */
        bool RunString(const std::string& envName, const std::string& script);
        
        /**
         * Run a script file, sandboxed in the standard megamol_env.
         */
        bool RunFile(const std::string& fileName);
        /**
         * Run a script file, sandboxed in the standard megamol_env.
         */
        bool RunFile(const std::wstring& fileName);
        /**
         * Run a script string, sandboxed in the standard megamol_env.
         */
        bool RunString(const std::string& script);

        /**
         * Answer whether the wrapped lua state is valid
         */
        bool StateOk();

        // ************************************************************
        // Lua interface routines, published to Lua as mm<name>
        // ************************************************************

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
         *mmSetConfigValue(string name, string value): set configuration value 'name'
         * to 'value'.
         */
        int SetConfigValue(lua_State *L);

    private:

        /** error handler */
        void consumeError(int error, char *file, int line);

        /** print table on the stack somewhat */
        void printTable(lua_State *L, std::stringstream& out);

        /** answer whether LuaState is instanced for configuration */
        bool checkConfiguring(const std::string where);

        /** answer whether LuaState is instanced for runtime */
        bool checkRunning(const std::string where);

        /** all of the Lua startup code */
        void commonInit();

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
    };

} /* namespace core */
} /* namespace megamol */
#endif /* MEGAMOLCORE_LUASTATE_H_INCLUDED */