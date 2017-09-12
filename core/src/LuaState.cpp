/*
* LuaState.cpp
*
* Copyright (C) 2017 by Universitaet Stuttgart (VIS).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#if (_MSC_VER > 1000)
#pragma warning(disable: 4996)
#endif /* (_MSC_VER > 1000) */
#if (_MSC_VER > 1000)
#pragma warning(default: 4996)
#endif /* (_MSC_VER > 1000) */

#include "mmcore/LuaState.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/utility/Configuration.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/Path.h"
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "vislib/sys/sysfunctions.h"

extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}

//#define LUA_FULL_ENVIRONMENT

/*****************************************************************************/

bool iequals(const std::string& one, const std::string& other) {
    size_t size = one.size();
    if (other.size() != size) {
        return false;
    }
    for (unsigned int i = 0; i < size; ++i) {
        if (tolower(one[i]) != tolower(other[i])) {
            return false;
        }
    }
    return true;
}

const std::string megamol::core::LuaState::MEGAMOL_ENV = "megamol_env = {"
"  print = mmLogInfo,"
"  mmLog = mmLog,"
"  mmLogInfo = mmLogInfo,"
"  mmGetBitWidth = mmGetBitWidth,"
"  mmGetConfiguration = mmGetConfiguration,"
"  mmGetOS = mmGetOS,"
"  mmGetMachineName = mmGetMachineName,"
"  mmSetAppDir = mmSetAppDir,"
"  mmAddShaderDir = mmAddShaderDir,"
"  mmAddResourceDir = mmAddResourceDir,"
"  mmPluginLoaderInfo = mmPluginLoaderInfo,"
"  mmSetLogFile = mmSetLogFile,"
"  mmSetLogLevel = mmSetLogLevel,"
"  mmSetEchoLevel = mmSetEchoLevel,"
"  mmSetConfigValue = mmSetConfigValue,"
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

typedef int (megamol::core::LuaState::*memberFunc)(lua_State * L);
// This template wraps a member function into a C-style "free" function compatible with lua.
template <memberFunc func>
int dispatch(lua_State * L) {
    megamol::core::LuaState *ptr = *static_cast<megamol::core::LuaState**>(lua_getextraspace(L));
    return ((*ptr).*func)(L);
}

#define USES_CHECK_LUA int __luaErr; __luaErr = LUA_OK;
#define CHECK_LUA(call) __luaErr = call;\
    consumeError(__luaErr, __FILE__, __LINE__);

void megamol::core::LuaState::consumeError(int error, char *file, int line) {
    if (error != LUA_OK) {
        const char *err = lua_tostring(L, -1); // get error from top of stack...
        vislib::sys::Log::DefaultLog.WriteError("Lua Error: %s at %s:%i\n", err, file, line);
        lua_pop(L, 1); // and remove it.
    }
}

void megamol::core::LuaState::printTable(lua_State *L, std::stringstream& out) {
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
            printTable(L, out);
        else if (lua_isfunction(L, -1))
            out << "      unknown function" << std::endl;
        else
            out << "      unknown type " << lua_type(L, -1) << std::endl;
        lua_pop(L, 1);
    }
    if (isOpen)
        out << "      " << "}" << std::endl;
}

void megamol::core::LuaState::printStack() {
    int n = lua_gettop(L); // get stack height
    vislib::sys::Log::DefaultLog.WriteInfo("Lua Stack:");
    for (int x = n; x >= 1; x--) {
        int t = lua_type(L, x);
        switch (t) {
            case LUA_TSTRING:  /* strings */
                vislib::sys::Log::DefaultLog.WriteInfo("%02i: string %s", x, lua_tostring(L, x));
                break;

            case LUA_TBOOLEAN:  /* booleans */
                vislib::sys::Log::DefaultLog.WriteInfo("%02i: bool %s", x, 
                    (lua_toboolean(L, x) ? "true" : "false"));
                break;

            case LUA_TNUMBER:  /* numbers */
                vislib::sys::Log::DefaultLog.WriteInfo("%02i: number %f", x, lua_tonumber(L, x));
                break;

            case LUA_TTABLE:
                {
                    std::stringstream out;
                    printTable(L, out);
                    vislib::sys::Log::DefaultLog.WriteInfo("%02i: table:\n%s", x, out.str().c_str());
                }
                break;

            default:  /* other values */
                vislib::sys::Log::DefaultLog.WriteInfo("%02i: unprintable %s", x, 
                    lua_typename(L, t));
                break;

        }
    }
}


bool megamol::core::LuaState::checkConfiguring(const std::string where) {
    if (this->conf != nullptr) {
        return true;
    } else {
        std::string err = where + "is only legal when reading the configuration";
        lua_pushstring(L, err.c_str());
        lua_error(L);
        return false;
    }
}


bool megamol::core::LuaState::checkRunning(const std::string where) {
    if (this->coreInst != nullptr) {
        return true;
    } else {
        std::string err = where + "is only legal when MegaMol is running";
        lua_pushstring(L, err.c_str());
        lua_error(L);
        return false;
    }
}


void megamol::core::LuaState::commonInit() {
    if (L != nullptr) {

        // push API
        //TODO
        *static_cast<LuaState**>(lua_getextraspace(L)) = this;

        lua_register(L, "mmLog", &dispatch<&LuaState::Log>);
        lua_register(L, "mmLogInfo", &dispatch<&LuaState::LogInfo>);

        lua_register(L, "mmGetOS", &dispatch<&LuaState::GetOS>);
        lua_register(L, "mmGetBitWidth", &dispatch<&LuaState::GetBitWidth>);
        lua_register(L, "mmGetConfiguration", &dispatch<&LuaState::GetConfiguration>);
        lua_register(L, "mmGetMachineName", &dispatch<&LuaState::GetMachineName>);

        lua_register(L, "mmSetAppDir", &dispatch<&LuaState::SetAppDir>);
        lua_register(L, "mmAddShaderDir", &dispatch<&LuaState::AddShaderDir>);
        lua_register(L, "mmAddResourceDir", &dispatch<&LuaState::AddResourceDir>);
        lua_register(L, "mmPluginLoaderInfo", &dispatch<&LuaState::PluginLoaderInfo>);

        lua_register(L, "mmSetLogFile", &dispatch<&LuaState::SetLogFile>);
        lua_register(L, "mmSetLogLevel", &dispatch<&LuaState::SetLogLevel>);
        lua_register(L, "mmSetEchoLevel", &dispatch<&LuaState::SetEchoLevel>);

        lua_register(L, "mmSetConfigValue", &dispatch<&LuaState::SetConfigValue>);

#ifdef LUA_FULL_ENVIRONMENT
        // load all environment
        //luaL_openlibs(L);
#else
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
#endif

        LoadEnviromentString(MEGAMOL_ENV);

        auto typ = lua_getglobal(L, "megamol_env");
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
}


/*
* megamol::core::LuaState::LuaState
*/
megamol::core::LuaState::LuaState(CoreInstance *inst) : L(luaL_newstate()),
        coreInst(inst), conf(nullptr) {
    this->commonInit();
}


megamol::core::LuaState::LuaState(utility::Configuration *conf) : L(luaL_newstate()), 
    coreInst(nullptr), conf(conf) {
    this->commonInit();
}


/*
* megamol::core::LuaState::~LuaState
*/
megamol::core::LuaState::~LuaState() {
    if (L != nullptr) {
        lua_close(L);
    }
}


bool megamol::core::LuaState::StateOk() {
    return L != nullptr;
}


bool megamol::core::LuaState::LoadEnviromentFile(const std::string& fileName) {
    std::ifstream input(fileName, std::ios::in);
    std::stringstream buffer;
    buffer << input.rdbuf();
    return LoadEnviromentString(buffer.str());
}


bool megamol::core::LuaState::LoadEnviromentString(const std::string& envString) {
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


bool megamol::core::LuaState::RunFile(const std::string& envName, const std::string& fileName) {
    std::ifstream input(fileName, std::ios::in);
    if (!input.fail()) {
        std::stringstream buffer;
        buffer << input.rdbuf();
        return RunString(envName, buffer.str());
    } else {
        return false;
    }
}


bool megamol::core::LuaState::RunFile(const std::string& envName, const std::wstring& fileName) {
    vislib::sys::File input;
    if (input.Open(fileName.c_str(), vislib::sys::File::AccessMode::READ_ONLY,
        vislib::sys::File::ShareMode::SHARE_READ, vislib::sys::File::CreationMode::OPEN_ONLY)) {
        vislib::StringA contents;
        vislib::sys::ReadTextFile(contents, input);
        input.Close();
        return RunString(envName, std::string(contents));
    } else {
        return false;
    }
}


bool megamol::core::LuaState::RunString(const std::string& envName, const std::string& script) {
    if (L != nullptr) {
        USES_CHECK_LUA;
        luaL_loadbuffer(L, script.c_str(), script.length(), "LuaState::RunString");
        lua_getglobal(L, envName.c_str());
        lua_setupvalue(L, -2, 1); // replace the environment with the one loaded from env.lua, disallowing some functions
        CHECK_LUA(lua_pcall(L, 0, LUA_MULTRET, 0));
        return true;
    } else {
        return false;
    }
}


bool megamol::core::LuaState::RunFile(const std::string& fileName) {
    return RunFile("megamol_env", fileName);
}


bool megamol::core::LuaState::RunFile(const std::wstring& fileName) {
    return RunFile("megamol_env", fileName);
}


bool megamol::core::LuaState::RunString(const std::string& script) {
    return RunString("megamol_env", script);
}


int megamol::core::LuaState::GetBitWidth(lua_State *L) {
    lua_pushinteger(L, vislib::sys::SystemInformation::SelfWordSize());
    return 1;
}


int megamol::core::LuaState::GetConfiguration(lua_State *L) {
#ifdef _DEBUG
    lua_pushstring(L, "debug");
#else
    lua_pushstring(L, "release");
#endif
    return 1;
}


int megamol::core::LuaState::GetOS(lua_State *L) {
    switch (vislib::sys::SystemInformation::SystemType()) {
        case vislib::sys::SystemInformation::OSTYPE_WINDOWS:
            lua_pushstring(L, "windows");
            break;
        case vislib::sys::SystemInformation::OSTYPE_LINUX:
            lua_pushstring(L, "linux");
            break;
        case vislib::sys::SystemInformation::OSTYPE_UNKNOWN:
            lua_pushstring(L, "unknown");
            break;
    }
    return 1;
}


int megamol::core::LuaState::GetMachineName(lua_State *L) {
    lua_pushstring(L, vislib::sys::SystemInformation::ComputerNameA());
    return 1;
}


int megamol::core::LuaState::Log(lua_State *L) {
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


int megamol::core::LuaState::LogInfo(lua_State *L) {
    USES_CHECK_LUA;
    lua_pushinteger(L, vislib::sys::Log::LEVEL_INFO);
    lua_insert(L, 1); // prepend info level to arguments
    lua_getglobal(L, "mmLog");
    lua_insert(L, 1); // prepend mmLog function to all arguments
    CHECK_LUA(lua_pcall(L, lua_gettop(L) - 1, 0, 0)); // run
    return 0;
}


int megamol::core::LuaState::SetAppDir(lua_State *L) {
    if (this->checkConfiguring("mmSetAppDir")) {
        // TODO do we need to make an OS-dependent path here?
        auto p = luaL_checkstring(L, 1);
        this->conf->appDir = vislib::StringW(p);
    }
    return 0;
}


int megamol::core::LuaState::AddShaderDir(lua_State *L) {
    if (this->checkConfiguring("mmAddShaderDir")) {
        // TODO do we need to make an OS-dependent path here?
        auto p = luaL_checkstring(L, 1);
        this->conf->AddShaderDirectory(p);
    }
    return 0;
}


int megamol::core::LuaState::AddResourceDir(lua_State *L) {
    if (this->checkConfiguring("mmAddResourceDir")) {
        // TODO do we need to make an OS-dependent path here?
        auto p = luaL_checkstring(L, 1);
        this->conf->AddResourceDirectory(p);
    }
    return 0;
}


int megamol::core::LuaState::PluginLoaderInfo(lua_State *L) {
    if (this->checkConfiguring("mmPluginLoaderInfo")) {
        // TODO do we need to make an OS-dependent path here?
        auto p = luaL_checkstring(L, 1);
        auto f = luaL_checkstring(L, 2);
        std::string a = luaL_checkstring(L, 3);
        bool inc = true;
        if (iequals(a, "include")) {
            inc = true;
        } else if (iequals(a, "exclude")) {
            inc = false;
        } else {
            lua_pushstring(L, "the third parameter of mmPluginLoaderInfo must be"
                " 'include' or 'exclude'.");
            lua_error(L);
        }
        this->conf->AddPluginLoadInfo(vislib::TString(p), vislib::TString(f), inc);
    }
    return 0;
}


int megamol::core::LuaState::SetLogFile(lua_State *L) {
    if (this->checkConfiguring("mmSetLogFile")) {
        // TODO do we need to make an OS-dependent path here?
        auto p = luaL_checkstring(L, 1);
        if (!megamol::core::utility::Configuration::logFilenameLocked) {
            if (this->conf->instanceLog != nullptr) {
            this->conf->instanceLog->SetLogFileName(vislib::sys::Path::Resolve(p),
                USE_LOG_SUFFIX);
            }
        }
    }
    return 0;
}


int megamol::core::LuaState::SetLogLevel(lua_State *L) {
    if (this->checkConfiguring("mmSetLogLevel")) {
        auto l = luaL_checkstring(L, 1);
        if (!megamol::core::utility::Configuration::logLevelLocked) {
            if (this->conf->instanceLog != nullptr) {
                this->conf->instanceLog->SetLevel(parseLevelAttribute(l));
            }
        }
    }
    return 0;
}


int megamol::core::LuaState::SetEchoLevel(lua_State *L) {
    if (this->checkConfiguring("mmSetEchoLevel")) {
        auto l = luaL_checkstring(L, 1);
        if (!megamol::core::utility::Configuration::logEchoLevelLocked) {
            if (this->conf->instanceLog != nullptr) {
                this->conf->instanceLog->SetEchoLevel(parseLevelAttribute(l));
            }
        }
    }
    return 0;
}


int megamol::core::LuaState::SetConfigValue(lua_State *L) {
    if (this->checkConfiguring("mmSetConfigValue")) {
        auto name = luaL_checkstring(L, 1);
        auto value = luaL_checkstring(L, 2);
        this->conf->setConfigValue(name, value);
    }
    return 0;
}


UINT megamol::core::LuaState::parseLevelAttribute(const std::string attr) {
    UINT retval = vislib::sys::Log::LEVEL_ERROR;
    if (iequals(attr, "error")) {
        retval = vislib::sys::Log::LEVEL_ERROR;
    } else if (iequals(attr, "warn")) {
        retval = vislib::sys::Log::LEVEL_WARN;
    } else if (iequals(attr, "warning")) {
        retval = vislib::sys::Log::LEVEL_WARN;
    } else if (iequals(attr, "info")) {
        retval = vislib::sys::Log::LEVEL_INFO;
    } else if (iequals(attr, "none")) {
        retval = vislib::sys::Log::LEVEL_NONE;
    } else if (iequals(attr, "null")) {
        retval = vislib::sys::Log::LEVEL_NONE;
    } else if (iequals(attr, "zero")) {
        retval = vislib::sys::Log::LEVEL_NONE;
    } else if (iequals(attr, "all")) {
        retval = vislib::sys::Log::LEVEL_ALL;
    } else if (iequals(attr, "*")) {
        retval = vislib::sys::Log::LEVEL_ALL;
    } else {
        try {
            retval = std::stoi(attr);
        } catch (...) {
            retval = vislib::sys::Log::LEVEL_ERROR;
        }
    }
    return retval;
}
