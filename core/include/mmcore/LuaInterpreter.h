#ifndef MEGAMOLCORE_LUAINTERPRETER_H_INCLUDED
#define MEGAMOLCORE_LUAINTERPRETER_H_INCLUDED

#include <fstream>
#include <functional>
#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <string>

#include "mmcore/utility/log/Log.h"

#include "lua.hpp"

//#define ATTACH_LUA_DEBUGGER
//#ifdef ATTACH_LUA_DEBUGGER
//// you need lua53.dll in your path to actually attach the debugger!
//luaL_requiref(L, LUA_IOLIBNAME, luaopen_io, 1);
//lua_pop(L, 1);
//luaL_requiref(L, LUA_DBLIBNAME, luaopen_debug, 1);
//lua_pop(L, 1);
//#endif
//void attachDebugger() {
//    static bool first = true;
//    if (first) {
//        luaL_dostring(L, "\npackage.cpath = [[T:/Utilities/zerobrane/x64/clibs53/?.dll;]] .. package.cpath\n"
//                         "package.path = package.path.. "
//                         "[[;T:/Utilities/zerobrane/lualibs/mobdebug/?.lua;T:/Utilities/zerobrane/lualibs/?.lua]]\n"
//                         "require('mobdebug').start()\n"
//                         "print(debug.getinfo(1,'S').source)\n"
//                         "require('mobdebug').off()\n");
//        first = false;
//    } else {
//        luaL_dostring(L, "require('mobdebug').on()\n"
//                         "print(debug.getinfo(1,'S').source)\n"
//                         "require('mobdebug').off()\n");
//    }
//}

namespace megamol {
namespace core {

template <class C> using luaCallbackFunc = int (C::*)(lua_State* L);

template <class C, luaCallbackFunc<C> func> int dispatch(lua_State* L) {
    C* ptr = *static_cast<C**>(lua_getextraspace(L));
    return (ptr->*func)(L);
}

static int invoke_lua_std_function(lua_State* L){
    //const int argc = lua_gettop(L);

    const auto index = lua_upvalueindex(1);
    //if (lua_islightuserdata(L, index))
    //{
        const void* ptr = lua_touserdata(L, index);
        const auto func_ptr = reinterpret_cast<std::function<int(lua_State *)> const*>(ptr);
        return (*func_ptr)(L);

    //} else {
    //    std::cout << "PANIC: NO USER DATA" << std::endl;
    //}
    //return 0;
};

// clang-format off
#define MMC_LUA_MMLOG "mmLog"
#define MMC_LUA_MMLOGINFO "mmLogInfo"
#define MMC_LUA_MMDEBUGPRINT "mmDebugPrint"
#define MMC_LUA_MMHELP "mmHelp"
//clang-format on

/** Simple Lua interpreter class */
template <class T> class LuaInterpreter {
public:

    explicit LuaInterpreter(T* t, std::string const& env = DEFAULT_ENV);

    LuaInterpreter(LuaInterpreter const& rhs) = delete;

    LuaInterpreter(LuaInterpreter&& rhs) = delete;

    LuaInterpreter& operator=(LuaInterpreter const& rhs) = delete;

    LuaInterpreter& operator=(LuaInterpreter&& rhs) = delete;

    ~LuaInterpreter(void);

    /**
     * Load an environment from a file. The environment is a table
     * of the form "env = { alias = function, libraryname = { ... }, ...}"
     * that should be used to sandbox scripts that are run afterwards.
     * 
     * @return the environment name
     */
    bool LoadEnviromentFile(const std::string& fileName, std::string& envName);

    /** Load an environment from a string. See LoadEnvironmentFile. */
    bool LoadEnviromentString(const std::string& envString, std::string& envName);

    /**
     * Run a script string, sandboxed in the environment provided.
     */
    bool RunString(const std::string& envName, const std::string& script, std::string& result);

    /**
     * Run a script string, sandboxed in the standard default_env.
     */
    bool RunString(const std::string& script, std::string& result);

    /**
     * Register callback function in lua and all environments known to date
     */
    template <class C, luaCallbackFunc<C> func> void RegisterCallback(std::string const& name, std::string const& help) {
        //this->theCallbacks += name + "=" + name + ",";
        this->theHelp.push_back(std::make_pair(name,help));
        lua_register(L, name.c_str(), &(dispatch<C, func>));
        for(auto &x: theEnvironments) {
            luaL_dostring(L, (x + "." + name + "="+name).c_str());
        }
    }

    /**
     * Register callback function in lua and all environments known to date
     */
    void RegisterCallback(std::string const& name, std::string const& help, std::function<int(lua_State*)>& func) {
        //this->theCallbacks += name + "=" + name + ",";
        this->theHelp.push_back(std::make_pair(name,help));
        lua_pushlightuserdata(L, &func); // push ptr to func as user data to be used by invoke_lua_std_function onto stack
        lua_pushcclosure(L, &invoke_lua_std_function, 1); // register invoke_lua_std_function as closure that gets ptr to func
        lua_setglobal(L, name.c_str()); // set pushed closure as global under name "name"
        for(auto &x: theEnvironments) {
            luaL_dostring(L, (x + "." + name + "="+name).c_str());
        }
    }

    /**
     * Register an alias to a callback function in lua and all environments known to date.
     * Only works if the function has already been registered.
     */
    void RegisterAlias(std::string const& name, std::string const&alias) {
        //this->theCallbacks += name + "=" + alias + ",";
        for(auto &x: theEnvironments) {
            luaL_dostring(L, (x + "." + alias + "="+name).c_str());
        }
    }

    /**
     * Register a constant in lua and all environments known to date.
     */
    void RegisterConstant(std::string const& name, uint32_t value) {
        //this->theConstants[name] = value;
        lua_pushinteger(L, value);
        lua_setglobal(L, name.c_str());
        for(auto &x: theEnvironments) {
            luaL_dostring(L, (x + "." + name + "="+name).c_str());
        }
    }

    void UnregisterCallback(std::string const& name) {
        auto remove_end = std::remove_if(this->theHelp.begin(), this->theHelp.end(), [name](std::pair<std::string, std::string> const& p){return p.first == name;});
        this->theHelp.erase(remove_end, this->theHelp.end());

        luaL_dostring(L, (name + "= nil").c_str());
        for(auto &x: theEnvironments) {
            luaL_dostring(L, (x + "." + name + "= nil").c_str());
        }
    }

    void UnregisterAlias(std::string const& alias) {
        for(auto &x: theEnvironments) {
            luaL_dostring(L, (x + "." + alias + "= nil").c_str());
        }
    }

    void UnregisterConstant(std::string const& name) {
        lua_pushnil(L);
        lua_setglobal(L, name.c_str());
        for(auto &x: theEnvironments) {
            luaL_dostring(L, (x + "." + name + "= nil").c_str());
        }
    }

    /** print the stack somewhat */
    void printStack();

    void ThrowError(std::string err) {
        lua_pushstring(L, err.c_str());
        lua_error(L);
    }

private:

    static std::string const DEFAULT_ENV;

    /** error handler */
    void consumeError(int error, const char* file, int line);

    /** gets a string from the stack position i. returns false if it's not a string */
    bool getString(int i, std::string& out);

    bool getDouble(int i, double& out);

    /** print table on the stack somewhat */
    void printTable(std::stringstream& out);

    /**
     * Register callback function to lua state
     */
    template <class C, luaCallbackFunc<C> func> void registerCallback(std::string const& name) {
        lua_register(L, name.c_str(), &(dispatch<C, func>));
    }

    // ************************************************************
    // Lua interface routines, published to Lua as mm<name>
    // ************************************************************

    int log(lua_State* L);

    int logInfo(lua_State* L);

    int help(lua_State* L);

    lua_State* L;

    T* that;

    std::vector<std::pair<std::string, std::string>> theHelp;

    std::vector<std::string> theEnvironments;

    // std::vector<std::function<int(lua_State*)>> registry;
}; /* end class LuaInterpreter */


#define USES_CHECK_LUA                                                                                                 \
    int __luaErr;                                                                                                      \
    __luaErr = LUA_OK;
#define CHECK_LUA(call)                                                                                                \
    __luaErr = call;                                                                                                   \
    consumeError(__luaErr, __FILE__, __LINE__);


template <class T> void megamol::core::LuaInterpreter<T>::consumeError(int error, const char* file, int line) {
    if (error != LUA_OK) {
        const char* err = lua_tostring(L, -1); // get error from top of stack...
        megamol::core::utility::log::Log::DefaultLog.WriteError("Lua Error: %s at %s:%i\n", err, file, line);
        lua_pop(L, 1); // and remove it.
    }
}

template <class T>
std::string const megamol::core::LuaInterpreter<T>::DEFAULT_ENV =
    "default_env = {"
    "  error = error,"
    "  ipairs = ipairs,"
    "  load = load,"
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

template <class T> megamol::core::LuaInterpreter<T>::LuaInterpreter(T* t, std::string const& env) : L{luaL_newstate()}, that{t} {
    *static_cast<T**>(lua_getextraspace(L)) = that;

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
    luaL_requiref(L, LUA_LOADLIBNAME, luaopen_package, 1);
    lua_pop(L, 1);

    std::string envName;
    LoadEnviromentString(env, envName);

    RegisterCallback<LuaInterpreter, &LuaInterpreter::log>(MMC_LUA_MMLOG, "(int level, ...)\n\tLog to MegaMol console. Level constants are LOGINFO, LOGWARNING, LOGERROR.");
    RegisterCallback<LuaInterpreter, &LuaInterpreter::logInfo>(MMC_LUA_MMLOGINFO, "(...)\n\tLog to MegaMol console with LOGINFO level.");
    RegisterAlias(MMC_LUA_MMLOGINFO, "print");
    RegisterCallback<LuaInterpreter, &LuaInterpreter::logInfo>(MMC_LUA_MMDEBUGPRINT, "(...)\n\tLog to MegaMol console with LOGINFO level.");
    RegisterCallback<LuaInterpreter, &LuaInterpreter::help>(MMC_LUA_MMHELP, "()\n\tShow this help.");

    RegisterConstant("LOGINFO", megamol::core::utility::log::Log::LEVEL_INFO);
    RegisterConstant("LOGWARNING", megamol::core::utility::log::Log::LEVEL_WARN);
    RegisterConstant("LOGERROR", megamol::core::utility::log::Log::LEVEL_ERROR);
}


template <class T> megamol::core::LuaInterpreter<T>::~LuaInterpreter(void) {
    if (L != nullptr) {
        lua_close(L);
    }
}


template <class T> bool megamol::core::LuaInterpreter<T>::LoadEnviromentFile(const std::string& fileName, std::string& envName) {
    std::ifstream input(fileName, std::ios::in);
    std::stringstream buffer;
    buffer << input.rdbuf();
    return LoadEnviromentString(buffer.str(), envName);
}


template <class T> bool megamol::core::LuaInterpreter<T>::LoadEnviromentString(const std::string& envString, std::string& envName) {
    if (L != nullptr) {
        USES_CHECK_LUA;
        auto n = envString.find('=');
        envName = envString.substr(0, n);
        const std::string whitespace = " \t\n";
        const auto strBegin = envName.find_first_not_of(whitespace);
        if (strBegin == std::string::npos)
            return false;
        const auto strEnd = envName.find_last_not_of(whitespace);
        const auto strRange = strEnd - strBegin + 1;
        theEnvironments.push_back(envName.substr(strBegin, strRange));
        CHECK_LUA(luaL_loadbuffer(L, envString.c_str(), envString.length(), envName.c_str()));
        CHECK_LUA(lua_pcall(L, 0, LUA_MULTRET, 0));
        return true;
    } else {
        return false;
    }
}


template <class T> bool megamol::core::LuaInterpreter<T>::RunString(const std::string& script, std::string& result) {
    return RunString("default_env", script, result);
}


template <class T>
bool megamol::core::LuaInterpreter<T>::RunString(
    const std::string& envName, const std::string& script, std::string& result) {
    if (L != nullptr) {
        luaL_loadbuffer(L, script.c_str(), script.length(), "LuaInterpreter::RunString");
        lua_getglobal(L, envName.c_str());
        lua_setupvalue(
            L, -2, 1); // replace the environment with the one loaded from env.lua, disallowing some functions
        int ret = lua_pcall(L, 0, LUA_MULTRET, 0);
        if (ret != LUA_OK) {
            const char* err = lua_tostring(
                L, -1); // get error from top of stack...
                        // megamol::core::utility::log::Log::DefaultLog.WriteError("Lua Error: %s at %s:%i\n", err, file, line);
            result = std::string(err);
            lua_pop(L, 1); // and remove it.
            return false;
        } else {
            bool good = true;
            // as a result, we still expect a string, if anything
            int n = lua_gettop(L);
            if (n > 0) {
                if (n > 2) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("Lua execution returned more than one value");
                    good = false;
                } else {
                    std::string res;
                    // we are not in a Lua callback, so making lua throw (luaL_checkstring) is not a good idea!
                    if (getString(1, res)) {
                        result = res;
                    } else {
                        double r;
                        if (getDouble(1, r)) {
                            result = std::to_string(r);
                        } else {
                            result = "Result is a complex type.";
                            megamol::core::utility::log::Log::DefaultLog.WriteError("Lua execution returned complex type");
                            good = false;
                        }
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

template <class T> bool megamol::core::LuaInterpreter<T>::getDouble(int i, double& out) {
    int t = lua_type(L, i);
    if (t == LUA_TNUMBER) {
        out = lua_tonumber(L, i);
        return true;
    }
    return false;
}

template <class T> void megamol::core::LuaInterpreter<T>::printTable(std::stringstream& out) {
    bool isOpen = false;
    if ((lua_type(L, -2) == LUA_TSTRING)) {
        out << "      " << lua_tostring(L, -2) << " {" << std::endl;
        isOpen = true;
    }

    lua_pushnil(L);
    while (lua_next(L, -2) != 0) {
        if (lua_isstring(L, -1))
            out << "      " << lua_tostring(L, -2) << "=" << lua_tostring(L, -1) << std::endl;
        else if (lua_isnumber(L, -1))
            out << "      " << lua_tostring(L, -2) << "=" << lua_tonumber(L, -1) << std::endl;
        else if (lua_istable(L, -1))
            printTable(out);
        else if (lua_isfunction(L, -1))
            out << "      unknown function" << std::endl;
        else
            out << "      unknown type " << lua_type(L, -1) << std::endl;
        lua_pop(L, 1);
    }
    if (isOpen)
        out << "      "
            << "}" << std::endl;
}

template <class T> void megamol::core::LuaInterpreter<T>::printStack() {
    int n = lua_gettop(L); // get stack height
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Lua Stack:");
    for (int x = n; x >= 1; x--) {
        int t = lua_type(L, x);
        switch (t) {
        case LUA_TSTRING: /* strings */
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("%02i: string %s", x, lua_tostring(L, x));
            break;

        case LUA_TBOOLEAN: /* booleans */
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("%02i: bool %s", x, (lua_toboolean(L, x) ? "true" : "false"));
            break;

        case LUA_TNUMBER: /* numbers */
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("%02i: number %f", x, lua_tonumber(L, x));
            break;

        case LUA_TTABLE: {
            std::stringstream out;
            printTable(out);
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("%02i: table:\n%s", x, out.str().c_str());
        } break;

        default: /* other values */
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("%02i: unprintable %s", x, lua_typename(L, t));
            break;
        }
    }
}


// ************************************************************
// Lua interface routines, published to Lua as mm<name>
// ************************************************************


template <class T> int megamol::core::LuaInterpreter<T>::log(lua_State* L) {
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
    megamol::core::utility::log::Log::DefaultLog.WriteMsg(static_cast<unsigned int>(level), "%s", out.str().c_str());
    return 0;
}


template <class T> int megamol::core::LuaInterpreter<T>::logInfo(lua_State* L) {
    USES_CHECK_LUA;
    lua_pushinteger(L, megamol::core::utility::log::Log::LEVEL_INFO);
    lua_insert(L, 1); // prepend info level to arguments
    lua_getglobal(L, "mmLog");
    lua_insert(L, 1);                                 // prepend mmLog function to all arguments
    CHECK_LUA(lua_pcall(L, lua_gettop(L) - 1, 0, 0)); // run
    return 0;
}

template <class T> int megamol::core::LuaInterpreter<T>::help(lua_State *L) {
    std::stringstream out;
    out << "MegaMol Lua Help:" << std::endl;
    std::string helpString;
    for (const auto &s : theHelp)
        helpString += s.first + s.second + "\n";
    out << helpString;
    lua_pushstring(L, out.str().c_str());
    return 1;
}

} /* end namespace core */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOLCORE_LUAINTERPRETER_H_INCLUDED */
