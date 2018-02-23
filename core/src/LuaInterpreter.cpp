//#include "stdafx.h"
//#include "mmcore/LuaInterpreter.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "vislib/sys/Log.h"

extern "C" {
#include "lauxlib.h"
#include "lualib.h"
}


#define USES_CHECK_LUA                                                                                                 \
    int __luaErr;                                                                                                      \
    __luaErr = LUA_OK;
#define CHECK_LUA(call)                                                                                                \
    __luaErr = call;                                                                                                   \
    consumeError(__luaErr, __FILE__, __LINE__);


void megamol::core::LuaInterpreter::consumeError(int error, char* file, int line) {
    if (error != LUA_OK) {
        const char* err = lua_tostring(L, -1); // get error from top of stack...
        vislib::sys::Log::DefaultLog.WriteError("Lua Error: %s at %s:%i\n", err, file, line);
        lua_pop(L, 1); // and remove it.
    }
}


std::string const megamol::core::LuaInterpreter::DEFAULT_ENV = "default_env = {"
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


megamol::core::LuaInterpreter::LuaInterpreter(void) 
    : L{luaL_newstate()} {
    RegisterCallback("mmliDebugPrint", *this, &LuaInterpreter::debugPrint);
    RegisterCallback("mmliLog", *this, &LuaInterpreter::logInfo);

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

    LoadEnviromentString(DEFAULT_ENV);

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


megamol::core::LuaInterpreter::LuaInterpreter(LuaInterpreter&& rhs) {
    if (L != nullptr) {
        lua_close(L);
    }
    L = nullptr;
    std::swap(this->L, rhs.L);
}


megamol::core::LuaInterpreter& megamol::core::LuaInterpreter::operator=(LuaInterpreter&& rhs) {
    if (L != nullptr) {
        lua_close(L);
    }
    L = nullptr;
    std::swap(this->L, rhs.L);

    return *this;
}


megamol::core::LuaInterpreter::~LuaInterpreter(void) {
    if (L != nullptr) {
        lua_close(L);
    }
}


bool megamol::core::LuaInterpreter::LoadEnviromentFile(const std::string& fileName) {
    std::ifstream input(fileName, std::ios::in);
    std::stringstream buffer;
    buffer << input.rdbuf();
    return LoadEnviromentString(buffer.str());
}


bool megamol::core::LuaInterpreter::LoadEnviromentString(const std::string& envString) {
    if (L != nullptr) {
        USES_CHECK_LUA;
        auto n = envString.find("=");
        std::string envName = envString.substr(0, n);
        CHECK_LUA(luaL_loadbuffer(L, envString.c_str(), envString.length(), envName.c_str()));
        CHECK_LUA(lua_pcall(L, 0, LUA_MULTRET, 0));
        return true;
    } else {
        return false;
    }
}


bool megamol::core::LuaInterpreter::RunString(const std::string& script, std::string& result) {
    return RunString("default_env", script, result);
}


bool megamol::core::LuaInterpreter::RunString(
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


//bool megamol::core::LuaInterpreter::RegisterCallback(std::string const& name, luaCallbackFunc func) {
//    lua_register(L, name.c_str(), func);
//}


bool megamol::core::LuaInterpreter::getString(int i, std::string& out) {
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


int megamol::core::LuaInterpreter::debugPrint(lua_State* L) {
    auto level = luaL_checkinteger(L, 1);
    int n = lua_gettop(L); // get number of  arguments
    std::stringstream out;
    for (int x = 2; x <= n; x++) {
        int t = lua_type(L, x);
        switch (t) {
        case LUA_TSTRING:  /* strings */
            out << lua_tostring(L, x);
            break;

        case LUA_TBOOLEAN:  /* booleans */
            out << (lua_toboolean(L, x) ? "true" : "false");
            break;

        case LUA_TNUMBER:  /* numbers */
            out << lua_tonumber(L, x);
            break;

        default:  /* other values */
            out << "cannot print a " << lua_typename(L, t);
            break;

        }
    }
    vislib::sys::Log::DefaultLog.WriteMsg(static_cast<UINT>(level), "%s", out.str().c_str());
    return 0;
}


int megamol::core::LuaInterpreter::logInfo(lua_State *L) {
    USES_CHECK_LUA;
    lua_pushinteger(L, vislib::sys::Log::LEVEL_INFO);
    lua_insert(L, 1); // prepend info level to arguments
    lua_getglobal(L, "mmliDebugPrint");
    lua_insert(L, 1); // prepend mmLog function to all arguments
    CHECK_LUA(lua_pcall(L, lua_gettop(L) - 1, 0, 0)); // run
    return 0;
}