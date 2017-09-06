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

    class CoreInstance;

    class LuaState {
    public:

        static const std::string MEGAMOL_ENV;

        LuaState(CoreInstance *inst);
        ~LuaState();
        
        // forbid copy-contructor? assignment?

        // load env
        bool LoadEnviromentFile(const std::string& fileName);
        bool LoadEnviromentString(const std::string& envString);
        // run in some env
        bool RunFile(const std::string& envName, const std::string& fileName);
        bool RunString(const std::string& envName, const std::string& script);
        // run in megamol_env
        bool RunFile(const std::string& fileName);
        bool RunString(const std::string& script);

        bool StateOk();

        int GetBitWidth(lua_State *L);
        int GetConfiguration(lua_State *L);
        int GetOS(lua_State *L);
        int GetMachineName(lua_State *L);
        int Log(lua_State *L);
        int LogInfo(lua_State *L);

    private:
        void consumeError(int error, int line);
        void printTable(lua_State *L, std::stringstream& out);
        void printStack();

        lua_State *L;
        CoreInstance *coreInst;
    };

} /* namespace core */
} /* namespace megamol */
#endif /* MEGAMOLCORE_LUASTATE_H_INCLUDED */