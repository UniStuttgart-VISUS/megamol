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


    class LuaState {
    public:

        static const std::string MEGAMOL_ENV;

        LuaState(CoreInstance *inst);
        LuaState(utility::Configuration *conf);
        ~LuaState();
        
        // forbid copy-contructor? assignment?

        // load env
        bool LoadEnviromentFile(const std::string& fileName);
        bool LoadEnviromentString(const std::string& envString);
        // run in some env
        bool RunFile(const std::string& envName, const std::string& fileName);
        bool RunFile(const std::string& envName, const std::wstring& fileName);
        bool RunString(const std::string& envName, const std::string& script);
        // run in megamol_env
        bool RunFile(const std::string& fileName);
        bool RunFile(const std::wstring& fileName);
        bool RunString(const std::string& script);

        bool StateOk();

        int GetBitWidth(lua_State *L);
        int GetConfiguration(lua_State *L);
        int GetOS(lua_State *L);
        int GetMachineName(lua_State *L);
        int Log(lua_State *L);
        int LogInfo(lua_State *L);

        int SetAppDir(lua_State *L);
        
        int AddShaderDir(lua_State *L);
        int AddResourceDir(lua_State *L);

        //// path, fileglob, inc=true?
        int PluginLoaderInfo(lua_State *L);

        int SetLogFile(lua_State *L);
        int SetLogLevel(lua_State *L);
        int SetEchoLevel(lua_State *L);

        int SetConfigValue(lua_State *L);

    private:
        void consumeError(int error, char *file, int line);
        void printTable(lua_State *L, std::stringstream& out);
        bool checkConfiguring(const std::string where);
        bool checkRunning(const std::string where);
        void commonInit();
        void printStack();
        static UINT parseLevelAttribute(const std::string attr);

        lua_State *L;
        CoreInstance *coreInst;
        utility::Configuration *conf;
    };

} /* namespace core */
} /* namespace megamol */
#endif /* MEGAMOLCORE_LUASTATE_H_INCLUDED */