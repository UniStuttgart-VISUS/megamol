#ifndef MEGAMOLCORE_LUAINTERPRETER_H_INCLUDED
#define MEGAMOLCORE_LUAINTERPRETER_H_INCLUDED

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "vislib/sys/Log.h"

extern "C" {
#include "lauxlib.h"
#include "lua.h"
#include "lualib.h"
}

namespace megamol {
namespace core {

template <class C> using luaCallbackFunc = int (C::*)(lua_State* L);

template <class C, luaCallbackFunc<C> func> int dispatch(lua_State* L) {
    C* ptr = *static_cast<C**>(lua_getextraspace(L));
    return (ptr->*func)(L);
}


/** Simple Lua interpreter class */
template <class T> class LuaInterpreter {
public:
    static std::string const DEFAULT_ENV;

    static std::string const BASE_ENV;

    void init(std::string const& env);

    explicit LuaInterpreter(T* t);

    LuaInterpreter(LuaInterpreter const& rhs) = delete;

    LuaInterpreter(LuaInterpreter&& rhs) = delete;

    LuaInterpreter& operator=(LuaInterpreter const& rhs) = delete;

    LuaInterpreter& operator=(LuaInterpreter&& rhs) = delete;

    ~LuaInterpreter(void);

    /**
     * Load an environment from a file. The environment is a table
     * of the form "env = { alias = function, libraryname = { ... }, ...}"
     * that should be used to sandbox scripts that are run afterwards.
     */
    bool LoadEnviromentFile(const std::string& fileName);

    /** Load an environment from a string. See LoadEnvironmentFile. */
    bool LoadEnviromentString(const std::string& envString);

    bool FinalizeEnvironment();

    /**
     * Run a script string, sandboxed in the environment provided.
     */
    bool RunString(const std::string& envName, const std::string& script, std::string& result);

    /**
     * Run a script string, sandboxed in the standard megamol_env.
     */
    bool RunString(const std::string& script, std::string& result);

    /**
     * Register callback function to lua state
     */
    template <class C, luaCallbackFunc<C> func> void RegisterCallback(std::string const& name) {
        this->env += name + "=" + name + ",";
        lua_register(L, name.c_str(), &(dispatch<C, func>));
    }

private:
    /** error handler */
    void consumeError(int error, char* file, int line);

    /** gets a string from the stack position i. returns false if it's not a string */
    bool getString(int i, std::string& out);

    /**
     * Register callback function to lua state
     */
    template <class C, luaCallbackFunc<C> func> void registerCallback(std::string const& name) {
        lua_register(L, name.c_str(), &(dispatch<C, func>));
    }

    // ************************************************************
    // Lua interface routines, published to Lua as mmli<name>
    // ************************************************************

    int debugPrint(lua_State* L);

    int logInfo(lua_State* L);

    lua_State* L;

    T* that;

    std::string env;

    // std::vector<std::function<int(lua_State*)>> registry;
}; /* end class LuaInterpreter */


#define USES_CHECK_LUA                                                                                                 \
    int __luaErr;                                                                                                      \
    __luaErr = LUA_OK;
#define CHECK_LUA(call)                                                                                                \
    __luaErr = call;                                                                                                   \
    consumeError(__luaErr, __FILE__, __LINE__);


template <class T> void megamol::core::LuaInterpreter<T>::consumeError(int error, char* file, int line) {
    if (error != LUA_OK) {
        const char* err = lua_tostring(L, -1); // get error from top of stack...
        vislib::sys::Log::DefaultLog.WriteError("Lua Error: %s at %s:%i\n", err, file, line);
        lua_pop(L, 1); // and remove it.
    }
}


template <class T>
std::string const megamol::core::LuaInterpreter<T>::DEFAULT_ENV =
    "default_env = {"
    "  print = mmliLog,"
    "  mmliLog = mmliLog,"
    "  mmliDebugPrint = mmliDebugPrint,"
    "  error = error,"
    "  ipairs = ipairs,"
    "  next = next,"
    "  pairs = pairs,"
    "  pcall = pcall,"
    "  tonumber = tonumber,"
    "  tostring = tostring,"
    "  type = type,"
    "  unpack = unpack,"
    "  coroutine = { create = coroutine.create, resume = coroutine.resume, "
    "      running = coroutine.running, status = coroutine.status, "
    "      wrap = coroutine.wrap },"
    "  string = { byte = string.byte, char = string.char, find = string.find, "
    "      format = string.format, gmatch = string.gmatch, gsub = string.gsub, "
    "      len = string.len, lower = string.lower, match = string.match, "
    "      rep = string.rep, reverse = string.reverse, sub = string.sub, "
    "      upper = string.upper },"
    "  table = { insert = table.insert, maxn = table.maxn, remove = table.remove, "
    "      sort = table.sort },"
    "  math = { abs = math.abs, acos = math.acos, asin = math.asin, "
    "      atan = math.atan, atan2 = math.atan2, ceil = math.ceil, cos = math.cos, "
    "      cosh = math.cosh, deg = math.deg, exp = math.exp, floor = math.floor, "
    "      fmod = math.fmod, frexp = math.frexp, huge = math.huge, "
    "      ldexp = math.ldexp, log = math.log, log10 = math.log10, max = math.max, "
    "      min = math.min, modf = math.modf, pi = math.pi, pow = math.pow, "
    "      rad = math.rad, random = math.random, sin = math.sin, sinh = math.sinh, "
    "      sqrt = math.sqrt, tan = math.tan, tanh = math.tanh },"
    "  os = { clock = os.clock, difftime = os.difftime, time = os.time },"
    "}";


template <class T>
std::string const megamol::core::LuaInterpreter<T>::BASE_ENV =
    "default_env = {"
    "  print = mmliLog,"
    "  mmliLog = mmliLog,"
    "  mmliDebugPrint = mmliDebugPrint,"
    "  error = error,"
    "  ipairs = ipairs,"
    "  next = next,"
    "  pairs = pairs,"
    "  pcall = pcall,"
    "  tonumber = tonumber,"
    "  tostring = tostring,"
    "  type = type,"
    "  unpack = unpack,"
    "  coroutine = { create = coroutine.create, resume = coroutine.resume, "
    "      running = coroutine.running, status = coroutine.status, "
    "      wrap = coroutine.wrap },"
    "  string = { byte = string.byte, char = string.char, find = string.find, "
    "      format = string.format, gmatch = string.gmatch, gsub = string.gsub, "
    "      len = string.len, lower = string.lower, match = string.match, "
    "      rep = string.rep, reverse = string.reverse, sub = string.sub, "
    "      upper = string.upper },"
    "  table = { insert = table.insert, maxn = table.maxn, remove = table.remove, "
    "      sort = table.sort },"
    "  math = { abs = math.abs, acos = math.acos, asin = math.asin, "
    "      atan = math.atan, atan2 = math.atan2, ceil = math.ceil, cos = math.cos, "
    "      cosh = math.cosh, deg = math.deg, exp = math.exp, floor = math.floor, "
    "      fmod = math.fmod, frexp = math.frexp, huge = math.huge, "
    "      ldexp = math.ldexp, log = math.log, log10 = math.log10, max = math.max, "
    "      min = math.min, modf = math.modf, pi = math.pi, pow = math.pow, "
    "      rad = math.rad, random = math.random, sin = math.sin, sinh = math.sinh, "
    "      sqrt = math.sqrt, tan = math.tan, tanh = math.tanh },"
    "  os = { clock = os.clock, difftime = os.difftime, time = os.time },";


template <class T> void LuaInterpreter<T>::init(std::string const& env) {
    // load parts of the environment
    luaL_requiref(L, "_G", luaopen_base, 1);
    lua_pop(L, 1);
    luaL_requiref(L, LUA_COLIBNAME, luaopen_coroutine, 1);
    lua_pop(L, 1);
    luaL_requiref(L, LUA_STRLIBNAME, luaopen_string, 1);
    lua_pop(L, 1);
    luaL_requiref(L, LUA_TABLIBNAME, luaopen_table, 1);
    lua_pop(L, 1);
    luaL_requiref(L, LUA_MATHLIBNAME, luaopen_math, 1);
    lua_pop(L, 1);
    luaL_requiref(L, LUA_OSLIBNAME, luaopen_os, 1);
    lua_pop(L, 1);

    LoadEnviromentString(env);

    auto typ = lua_getglobal(L, "default_env");
    lua_pushstring(L, "LOGINFO");
    lua_pushinteger(L, vislib::sys::Log::LEVEL_INFO);
    lua_rawset(L, -3);
    lua_pushstring(L, "LOGWARNING");
    lua_pushinteger(L, vislib::sys::Log::LEVEL_WARN);
    lua_rawset(L, -3);
    lua_pushstring(L, "LOGERROR");
    lua_pushnumber(L, vislib::sys::Log::LEVEL_ERROR);
    lua_rawset(L, -3);
    lua_pop(L, 1);
}

template <class T> megamol::core::LuaInterpreter<T>::LuaInterpreter(T* t) : L{luaL_newstate()}, that{t} {
    *static_cast<T**>(lua_getextraspace(L)) = that;

    this->env = BASE_ENV;

    registerCallback<LuaInterpreter, &LuaInterpreter::debugPrint>("mmliDebugPrint");
    registerCallback<LuaInterpreter, &LuaInterpreter::logInfo>("mmliLog");
}


// template<class T>
// megamol::core::LuaInterpreter<T>::LuaInterpreter(LuaInterpreter&& rhs) {
//    if (L != nullptr) {
//        lua_close(L);
//    }
//    L = nullptr;
//    std::swap(this->L, rhs.L);
//}


// template<class T>
// megamol::core::LuaInterpreter<T>& megamol::core::LuaInterpreter<T>::operator=(LuaInterpreter&& rhs) {
//    if (L != nullptr) {
//        lua_close(L);
//    }
//    L = nullptr;
//    std::swap(this->L, rhs.L);
//
//    return *this;
//}


template <class T> megamol::core::LuaInterpreter<T>::~LuaInterpreter(void) {
    if (L != nullptr) {
        lua_close(L);
    }
}


template <class T> bool megamol::core::LuaInterpreter<T>::LoadEnviromentFile(const std::string& fileName) {
    std::ifstream input(fileName, std::ios::in);
    std::stringstream buffer;
    buffer << input.rdbuf();
    return LoadEnviromentString(buffer.str());
}


template <class T> bool megamol::core::LuaInterpreter<T>::LoadEnviromentString(const std::string& envString) {
    if (L != nullptr) {
        USES_CHECK_LUA;
        auto n = envString.find('=');
        std::string envName = envString.substr(0, n);
        CHECK_LUA(luaL_loadbuffer(L, envString.c_str(), envString.length(), envName.c_str()));
        CHECK_LUA(lua_pcall(L, 0, LUA_MULTRET, 0));
        return true;
    } else {
        return false;
    }
}


template <class T> bool LuaInterpreter<T>::FinalizeEnvironment() {
    this->env += "}";

    this->init(this->env);

    return true;
}


template <class T> bool megamol::core::LuaInterpreter<T>::RunString(const std::string& script, std::string& result) {
    return RunString("default_env", script, result);
}


template <class T>
bool megamol::core::LuaInterpreter<T>::RunString(
    const std::string& envName, const std::string& script, std::string& result) {
    if (L != nullptr) {
        luaL_loadbuffer(L, script.c_str(), script.length(), "LuaState::RunString");
        lua_getglobal(L, envName.c_str());
        lua_setupvalue(
            L, -2, 1); // replace the environment with the one loaded from env.lua, disallowing some functions
        int ret = lua_pcall(L, 0, LUA_MULTRET, 0);
        if (ret != LUA_OK) {
            const char* err = lua_tostring(
                L, -1); // get error from top of stack...
                        // vislib::sys::Log::DefaultLog.WriteError("Lua Error: %s at %s:%i\n", err, file, line);
            result = std::string(err);
            lua_pop(L, 1); // and remove it.
            return false;
        } else {
            bool good = true;
            // as a result, we still expect a string, if anything
            int n = lua_gettop(L);
            if (n > 0) {
                if (n > 2) {
                    vislib::sys::Log::DefaultLog.WriteError("Lua execution returned more than one value");
                    good = false;
                } else {
                    std::string res;
                    // we are not in a Lua callback, so making lua throw (luaL_checkstring) is not a good idea!
                    if (getString(1, res)) {
                        result = res;
                    } else {
                        result = "Result is a non-string";
                        vislib::sys::Log::DefaultLog.WriteError("Lua execution returned non-string");
                        good = false;
                    }
                }
                // clean up stack!
                for (int i = 1; i <= n; i++) lua_pop(L, 1);
                return good;
            }
            return true;
        }
    } else {
        return false;
    }
}


template <class T> bool megamol::core::LuaInterpreter<T>::getString(int i, std::string& out) {
    int t = lua_type(L, i);
    if (t == LUA_TSTRING) {
        auto* res = lua_tostring(L, i);
        out = std::string(res);
        return true;
    }
    return false;
}


// ************************************************************
// Lua interface routines, published to Lua as mmli<name>
// ************************************************************


template <class T> int megamol::core::LuaInterpreter<T>::debugPrint(lua_State* L) {
    auto level = luaL_checkinteger(L, 1);
    int n = lua_gettop(L); // get number of  arguments
    std::stringstream out;
    for (int x = 2; x <= n; x++) {
        int t = lua_type(L, x);
        switch (t) {
        case LUA_TSTRING: /* strings */
            out << lua_tostring(L, x);
            break;

        case LUA_TBOOLEAN: /* booleans */
            out << (lua_toboolean(L, x) ? "true" : "false");
            break;

        case LUA_TNUMBER: /* numbers */
            out << lua_tonumber(L, x);
            break;

        default: /* other values */
            out << "cannot print a " << lua_typename(L, t);
            break;
        }
    }
    vislib::sys::Log::DefaultLog.WriteMsg(static_cast<UINT>(level), "%s", out.str().c_str());
    return 0;
}


template <class T> int megamol::core::LuaInterpreter<T>::logInfo(lua_State* L) {
    USES_CHECK_LUA;
    lua_pushinteger(L, vislib::sys::Log::LEVEL_INFO);
    lua_insert(L, 1); // prepend info level to arguments
    lua_getglobal(L, "mmliDebugPrint");
    lua_insert(L, 1);                                 // prepend mmLog function to all arguments
    CHECK_LUA(lua_pcall(L, lua_gettop(L) - 1, 0, 0)); // run
    return 0;
}


} /* end namespace core */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOLCORE_LUAINTERPRETER_H_INCLUDED */