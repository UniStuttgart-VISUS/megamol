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
#include "vislib/sys/AutoLock.h"
#include "vislib/UTF8Encoder.h"
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "vislib/sys/sysfunctions.h"
#include "vislib/sys/Process.h"

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

#define MMC_LUA_MMLOG "mmLog"
#define MMC_LUA_MMLOGINFO "mmLogInfo"
#define MMC_LUA_MMGETBITHWIDTH "mmGetBitWidth"
#define MMC_LUA_MMGETCONFIGURATION "mmGetConfiguration"
#define MMC_LUA_MMGETOS "mmGetOS"
#define MMC_LUA_MMGETMACHINENAME "mmGetMachineName"
#define MMC_LUA_MMSETAPPDIR "mmSetAppDir"
#define MMC_LUA_MMADDSHADERDIR "mmAddShaderDir"
#define MMC_LUA_MMADDRESOURCEDIR "mmAddResourceDir"
#define MMC_LUA_MMPLUGINLOADERINFO "mmPluginLoaderInfo"
#define MMC_LUA_MMGETMODULEPARAMS "mmGetModuleParams"
#define MMC_LUA_MMSETLOGFILE "mmSetLogFile"
#define MMC_LUA_MMSETLOGLEVEL "mmSetLogLevel"
#define MMC_LUA_MMSETECHOLEVEL "mmSetEchoLevel"
#define MMC_LUA_MMSETCONFIGVALUE "mmSetConfigValue"
#define MMC_LUA_MMGETPROCESSID "mmGetProcessID"
#define MMC_LUA_MMGETPARAMTYPE "mmGetParamType"
#define MMC_LUA_MMGETPARAMDESCRIPTION "mmGetParamDescription"
#define MMC_LUA_MMGETPARAMVALUE "mmGetParamValue"
#define MMC_LUA_MMSETPARAMVALUE "mmSetParamValue"
#define MMC_LUA_MMCREATEMODULE "mmCreateModule"
#define MMC_LUA_MMDELETEMODULE "mmDeleteModule"
#define MMC_LUA_MMCREATECALL "mmCreateCall"
#define MMC_LUA_MMDELETECALL "mmDeleteCall"
#define MMC_LUA_MMQUERYMODULES "mmQueryModules"
#define MMC_LUA_MMHELP "mmHelp"


const std::unordered_map<std::string, std::string> MM_LUA_HELP = {
    { MMC_LUA_MMLOG, MMC_LUA_MMLOG"(int level, ...)\n\tLog to MegaMol console. Level constants are LOGINFO, LOGWARNING, LOGERROR." },
    { MMC_LUA_MMLOGINFO, MMC_LUA_MMLOGINFO"(...)\n\tLog to MegaMol console with LOGINFO level." },
    { MMC_LUA_MMGETBITHWIDTH, MMC_LUA_MMGETBITHWIDTH"()\n\tReturns the bit width of the compiled executable." },
    { MMC_LUA_MMGETCONFIGURATION, MMC_LUA_MMGETCONFIGURATION"()\n\tReturns the configuration ('debug' or 'release')." },
    { MMC_LUA_MMGETOS, MMC_LUA_MMGETOS"()\n\tReturns the operating system ('windows', 'linux', or 'unknown')."},
    { MMC_LUA_MMGETMACHINENAME, MMC_LUA_MMGETMACHINENAME"" },
    { MMC_LUA_MMSETAPPDIR, MMC_LUA_MMSETAPPDIR"" },
    { MMC_LUA_MMADDSHADERDIR, MMC_LUA_MMADDSHADERDIR"" },
    { MMC_LUA_MMADDRESOURCEDIR, MMC_LUA_MMADDRESOURCEDIR"" },
    { MMC_LUA_MMPLUGINLOADERINFO, MMC_LUA_MMPLUGINLOADERINFO"" },
    { MMC_LUA_MMGETMODULEPARAMS, MMC_LUA_MMGETMODULEPARAMS"" },
    { MMC_LUA_MMSETLOGFILE, MMC_LUA_MMSETLOGFILE"" },
    { MMC_LUA_MMSETLOGLEVEL, MMC_LUA_MMSETLOGLEVEL"" },
    { MMC_LUA_MMSETECHOLEVEL, MMC_LUA_MMSETECHOLEVEL"" },
    { MMC_LUA_MMSETCONFIGVALUE, MMC_LUA_MMSETCONFIGVALUE"" },
    { MMC_LUA_MMGETPROCESSID, MMC_LUA_MMGETPROCESSID"" },
    { MMC_LUA_MMGETPARAMTYPE, MMC_LUA_MMGETPARAMTYPE"" },
    { MMC_LUA_MMGETPARAMDESCRIPTION, MMC_LUA_MMGETPARAMDESCRIPTION"" },
    { MMC_LUA_MMGETPARAMVALUE, MMC_LUA_MMGETPARAMVALUE"" },
    { MMC_LUA_MMSETPARAMVALUE, MMC_LUA_MMSETPARAMVALUE"" },
    { MMC_LUA_MMCREATEMODULE, MMC_LUA_MMCREATEMODULE"" },
    { MMC_LUA_MMDELETEMODULE, MMC_LUA_MMDELETEMODULE"" },
    { MMC_LUA_MMCREATECALL, MMC_LUA_MMCREATECALL"" },
    { MMC_LUA_MMDELETECALL, MMC_LUA_MMDELETECALL"" },
    { MMC_LUA_MMQUERYMODULES, MMC_LUA_MMQUERYMODULES"" },
    { MMC_LUA_MMHELP, MMC_LUA_MMHELP"" }
};

const std::string megamol::core::LuaState::MEGAMOL_ENV = "megamol_env = {"
"  print = " MMC_LUA_MMLOGINFO ","
"  error = error,"
MMC_LUA_MMLOG "=" MMC_LUA_MMLOG ","
MMC_LUA_MMLOGINFO "=" MMC_LUA_MMLOGINFO ","
MMC_LUA_MMGETBITHWIDTH "=" MMC_LUA_MMGETBITHWIDTH ","
MMC_LUA_MMGETCONFIGURATION "=" MMC_LUA_MMGETCONFIGURATION ","
MMC_LUA_MMGETOS "=" MMC_LUA_MMGETOS ","
MMC_LUA_MMGETMACHINENAME "=" MMC_LUA_MMGETMACHINENAME ","
MMC_LUA_MMSETAPPDIR "=" MMC_LUA_MMSETAPPDIR ","
MMC_LUA_MMADDSHADERDIR "=" MMC_LUA_MMADDSHADERDIR ","
MMC_LUA_MMADDRESOURCEDIR "=" MMC_LUA_MMADDRESOURCEDIR ","
MMC_LUA_MMPLUGINLOADERINFO "=" MMC_LUA_MMPLUGINLOADERINFO ","
MMC_LUA_MMGETMODULEPARAMS "=" MMC_LUA_MMGETMODULEPARAMS ","
MMC_LUA_MMSETLOGFILE "=" MMC_LUA_MMSETLOGFILE ","
MMC_LUA_MMSETLOGLEVEL "=" MMC_LUA_MMSETLOGLEVEL ","
MMC_LUA_MMSETECHOLEVEL "=" MMC_LUA_MMSETECHOLEVEL ","
MMC_LUA_MMSETCONFIGVALUE "=" MMC_LUA_MMSETCONFIGVALUE ","
MMC_LUA_MMGETPROCESSID "=" MMC_LUA_MMGETPROCESSID ","
MMC_LUA_MMGETPARAMTYPE "=" MMC_LUA_MMGETPARAMTYPE ","
MMC_LUA_MMGETPARAMDESCRIPTION "=" MMC_LUA_MMGETPARAMDESCRIPTION ","
MMC_LUA_MMGETPARAMVALUE "=" MMC_LUA_MMGETPARAMVALUE ","
MMC_LUA_MMSETPARAMVALUE "=" MMC_LUA_MMSETPARAMVALUE ","
MMC_LUA_MMCREATEMODULE "=" MMC_LUA_MMCREATEMODULE ","
MMC_LUA_MMDELETEMODULE "=" MMC_LUA_MMDELETEMODULE ","
MMC_LUA_MMCREATECALL "=" MMC_LUA_MMCREATECALL ","
MMC_LUA_MMDELETECALL "=" MMC_LUA_MMDELETECALL ","
MMC_LUA_MMQUERYMODULES "=" MMC_LUA_MMQUERYMODULES ","
MMC_LUA_MMHELP "=" MMC_LUA_MMHELP ","
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

        lua_register(L, MMC_LUA_MMLOG, &dispatch<&LuaState::Log>);
        lua_register(L, MMC_LUA_MMLOGINFO, &dispatch<&LuaState::LogInfo>);

        lua_register(L, MMC_LUA_MMGETOS, &dispatch<&LuaState::GetOS>);
        lua_register(L, MMC_LUA_MMGETBITHWIDTH, &dispatch<&LuaState::GetBitWidth>);
        lua_register(L, MMC_LUA_MMGETCONFIGURATION, &dispatch<&LuaState::GetConfiguration>);
        lua_register(L, MMC_LUA_MMGETMACHINENAME, &dispatch<&LuaState::GetMachineName>);

        lua_register(L, MMC_LUA_MMSETAPPDIR, &dispatch<&LuaState::SetAppDir>);
        lua_register(L, MMC_LUA_MMADDSHADERDIR, &dispatch<&LuaState::AddShaderDir>);
        lua_register(L, MMC_LUA_MMADDRESOURCEDIR, &dispatch<&LuaState::AddResourceDir>);
        lua_register(L, MMC_LUA_MMPLUGINLOADERINFO, &dispatch<&LuaState::PluginLoaderInfo>);

        lua_register(L, MMC_LUA_MMSETLOGFILE, &dispatch<&LuaState::SetLogFile>);
        lua_register(L, MMC_LUA_MMSETLOGLEVEL, &dispatch<&LuaState::SetLogLevel>);
        lua_register(L, MMC_LUA_MMSETECHOLEVEL, &dispatch<&LuaState::SetEchoLevel>);

        lua_register(L, MMC_LUA_MMSETCONFIGVALUE, &dispatch<&LuaState::SetConfigValue>);

        lua_register(L, MMC_LUA_MMGETPROCESSID, &dispatch<&LuaState::GetProcessID>);
        lua_register(L, MMC_LUA_MMGETMODULEPARAMS, &dispatch<&LuaState::GetModuleParams>);
        lua_register(L, MMC_LUA_MMGETPARAMTYPE, &dispatch<&LuaState::GetParamType>);
        lua_register(L, MMC_LUA_MMGETPARAMDESCRIPTION, &dispatch<&LuaState::GetParamDescription>);
        lua_register(L, MMC_LUA_MMGETPARAMVALUE, &dispatch<&LuaState::GetParamValue>);
        lua_register(L, MMC_LUA_MMSETPARAMVALUE, &dispatch<&LuaState::SetParamValue>);

        lua_register(L, MMC_LUA_MMCREATEMODULE, &dispatch<&LuaState::CreateModule>);
        lua_register(L, MMC_LUA_MMDELETEMODULE, &dispatch<&LuaState::DeleteModule>);
        lua_register(L, MMC_LUA_MMCREATECALL, &dispatch<&LuaState::CreateCall>);
        lua_register(L, MMC_LUA_MMDELETECALL, &dispatch<&LuaState::DeleteCall>);

        lua_register(L, MMC_LUA_MMQUERYMODULES, &dispatch<&LuaState::QueryModules>);

        lua_register(L, MMC_LUA_MMHELP, &dispatch<&LuaState::Help>);

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


bool megamol::core::LuaState::getString(int i, std::string& out) {
    int t = lua_type(L, i);
    if (t == LUA_TSTRING) {
        auto *res = lua_tostring(L, i);
        out = std::string(res);
        return true;
    }
    return false;
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


bool megamol::core::LuaState::RunFile(const std::string& envName, const std::string& fileName, std::string& result) {
    std::ifstream input(fileName, std::ios::in);
    if (!input.fail()) {
        std::stringstream buffer;
        buffer << input.rdbuf();
        return RunString(envName, buffer.str(), result);
    } else {
        return false;
    }
}


bool megamol::core::LuaState::RunFile(const std::string& envName, const std::wstring& fileName, std::string& result) {
    vislib::sys::File input;
    if (input.Open(fileName.c_str(), vislib::sys::File::AccessMode::READ_ONLY,
        vislib::sys::File::ShareMode::SHARE_READ, vislib::sys::File::CreationMode::OPEN_ONLY)) {
        vislib::StringA contents;
        vislib::sys::ReadTextFile(contents, input);
        input.Close();
        return RunString(envName, std::string(contents), result);
    } else {
        return false;
    }
}


bool megamol::core::LuaState::RunString(const std::string& envName, const std::string& script, std::string& result) {
    if (L != nullptr) {
        luaL_loadbuffer(L, script.c_str(), script.length(), "LuaState::RunString");
        lua_getglobal(L, envName.c_str());
        lua_setupvalue(L, -2, 1); // replace the environment with the one loaded from env.lua, disallowing some functions
        int old_n = lua_gettop(L);
        int ret = lua_pcall(L, 0, LUA_MULTRET, 0);
        if (ret != LUA_OK) {
            const char *err = lua_tostring(L, -1); // get error from top of stack...
            //vislib::sys::Log::DefaultLog.WriteError("Lua Error: %s at %s:%i\n", err, file, line);
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
                for (int i = 1; i <= n; i++)
                    lua_pop(L, 1);
                return good;
            }
            return true;
        }
    } else {
        return false;
    }
}


bool megamol::core::LuaState::RunFile(const std::string& fileName, std::string& result) {
    return RunFile("megamol_env", fileName, result);
}


bool megamol::core::LuaState::RunFile(const std::wstring& fileName, std::string& result) {
    return RunFile("megamol_env", fileName, result);
}


bool megamol::core::LuaState::RunString(const std::string& script, std::string& result) {
    return RunString("megamol_env", script, result);
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
    if (this->checkConfiguring(MMC_LUA_MMSETAPPDIR)) {
        // TODO do we need to make an OS-dependent path here?
        auto p = luaL_checkstring(L, 1);
        this->conf->appDir = vislib::StringW(p);
    }
    return 0;
}


int megamol::core::LuaState::AddShaderDir(lua_State *L) {
    if (this->checkConfiguring(MMC_LUA_MMADDSHADERDIR)) {
        // TODO do we need to make an OS-dependent path here?
        auto p = luaL_checkstring(L, 1);
        this->conf->AddShaderDirectory(p);
    }
    return 0;
}


int megamol::core::LuaState::AddResourceDir(lua_State *L) {
    if (this->checkConfiguring(MMC_LUA_MMADDRESOURCEDIR)) {
        // TODO do we need to make an OS-dependent path here?
        auto p = luaL_checkstring(L, 1);
        this->conf->AddResourceDirectory(p);
    }
    return 0;
}


int megamol::core::LuaState::PluginLoaderInfo(lua_State *L) {
    if (this->checkConfiguring(MMC_LUA_MMPLUGINLOADERINFO)) {
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
    if (this->checkConfiguring(MMC_LUA_MMSETLOGFILE)) {
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
    if (this->checkConfiguring(MMC_LUA_MMSETLOGLEVEL)) {
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
    if (this->checkConfiguring(MMC_LUA_MMSETECHOLEVEL)) {
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
    if (this->checkConfiguring(MMC_LUA_MMSETCONFIGVALUE)) {
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


int megamol::core::LuaState::GetProcessID(lua_State *L) {
//    if (this->checkRunning("mmGetProcessID")) {
        vislib::StringA str;
        unsigned int id = vislib::sys::Process::CurrentID();
        str.Format("%u", id);
        lua_pushstring(L, str.PeekBuffer());
        return 1;
//    }
}


int megamol::core::LuaState::GetModuleParams(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMGETMODULEPARAMS)) {
        auto paramName = luaL_checkstring(L, 1);

        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());

        AbstractNamedObject::const_ptr_type ano = this->coreInst->ModuleGraphRoot();
        AbstractNamedObjectContainer::const_ptr_type anoc = std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
        if (!anoc) {
            lua_pushstring(L, MMC_LUA_MMGETMODULEPARAMS": no root");
            lua_error(L);
            return 0;
        }
        // TODO honestly!
        Module::const_ptr_type mod = Module::dynamic_pointer_cast(const_cast<AbstractNamedObjectContainer*>(anoc.get())->FindNamedObject(paramName));
        if (!mod) {
            lua_pushstring(L, MMC_LUA_MMGETMODULEPARAMS": module not found");
            lua_error(L);
            return 0;
        }

        std::stringstream answer;
        vislib::StringA name(mod->FullName());
        answer << name << "\1";
        AbstractNamedObjectContainer::child_list_type::const_iterator si, se;
        se = mod->ChildList_End();
        for (si = mod->ChildList_Begin(); si != se; ++si) {
            const param::ParamSlot *slot = dynamic_cast<const param::ParamSlot*>((*si).get());
            if (slot != NULL) {
                //name.Append("::");
                //name.Append(slot->Name());

                answer << slot->Name() << "\1";

                vislib::StringA descUTF8;
                vislib::UTF8Encoder::Encode(descUTF8, slot->Description());
                answer << descUTF8 << "\1";

                auto psp = slot->Parameter();
                if (psp.IsNull()) {
                    std::ostringstream err;
                    err << MMC_LUA_MMGETMODULEPARAMS": ParamSlot " << slot->FullName() << " does seem to hold no parameter";
                    lua_pushstring(L, err.str().c_str());
                    lua_error(L);
                }

                vislib::RawStorage pspdef;
                psp->Definition(pspdef);
                // not nice, but we make HEX (base64 would be better, but I don't care)
                std::string answer2(pspdef.GetSize() * 2, ' ');
                for (SIZE_T i = 0; i < pspdef.GetSize(); ++i) {
                    uint8_t b = *pspdef.AsAt<uint8_t>(i);
                    uint8_t bh[2] = { static_cast<uint8_t>(b / 16), static_cast<uint8_t>(b % 16) };
                    for (unsigned int j = 0; j < 2; ++j) answer2[i * 2 + j] = (bh[j] < 10u) ? ('0' + bh[j]) : ('A' + (bh[j] - 10u));
                }
                answer << answer2 << "\1";

                vislib::StringA valUTF8;
                vislib::UTF8Encoder::Encode(valUTF8, psp->ValueString());

                answer << valUTF8 << "\1";
            }
        }
        lua_pushstring(L, answer.str().c_str());
        return 1;
    }
    return 0;
}


bool megamol::core::LuaState::getParamSlot(const std::string routine, const char *paramName, core::param::ParamSlot **out) {

    AbstractNamedObjectContainer::const_ptr_type root = std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(this->coreInst->ModuleGraphRoot());
    if (!root) {
        std::string err = routine + ": no root";
        lua_pushstring(L, err.c_str());
        lua_error(L);
        return false;
    }
    // TODO honestly!
    AbstractNamedObject::ptr_type obj = const_cast<AbstractNamedObjectContainer*>(root.get())->FindNamedObject(paramName);
    if (!obj) {
        std::string err = routine + ": parameter \"" + paramName + "\" not found";
        lua_pushstring(L, err.c_str());
        lua_error(L);
        return false;
    }
    *out = dynamic_cast<core::param::ParamSlot*>(obj.get());
    if (*out == nullptr) {
        std::string err = routine + ": parameter name \"" + paramName + "\" did not refer to a ParamSlot";
        lua_pushstring(L, err.c_str());
        lua_error(L);
        return false;
    }
    return true;
}


bool megamol::core::LuaState::getView(const std::string routine, const char *viewName,
    core::ViewInstance **out) {

    //AbstractNamedObjectContainer::ptr_type anoc = AbstractNamedObjectContainer::dynamic_pointer_cast(this->RootModule());
    //AbstractNamedObject::ptr_type ano = anoc->FindChild(mvn);
    //ViewInstance *vi = dynamic_cast<ViewInstance *>(ano.get());

    AbstractNamedObjectContainer::const_ptr_type root = AbstractNamedObjectContainer::dynamic_pointer_cast(this->coreInst->ModuleGraphRoot());
    if (!root) {
        std::string err = routine + ": no root";
        lua_pushstring(L, err.c_str());
        lua_error(L);
        return false;
    }
    AbstractNamedObject::ptr_type obj = const_cast<AbstractNamedObjectContainer*>(root.get())->FindNamedObject(viewName);
    if (!obj) {
        std::string err = routine + ": view \"" + std::string(viewName) + "\" not found";
        lua_pushstring(L, err.c_str());
        lua_error(L);
        return false;
    }
    *out = dynamic_cast<ViewInstance *>(obj.get());
    return true;
}


bool megamol::core::LuaState::getJob(const std::string routine, const char *jobName,
    core::JobInstance **out) {
    AbstractNamedObjectContainer::const_ptr_type root = AbstractNamedObjectContainer::dynamic_pointer_cast(this->coreInst->ModuleGraphRoot());
    if (!root) {
        std::string err = routine + ": no root";
        lua_pushstring(L, err.c_str());
        lua_error(L);
        return false;
    }
    AbstractNamedObject::ptr_type obj = const_cast<AbstractNamedObjectContainer*>(root.get())->FindNamedObject(jobName);
    if (!obj) {
        std::string err = routine + ": job \"" + std::string(jobName) + "\" not found";
        lua_pushstring(L, err.c_str());
        lua_error(L);
        return false;
    }
    *out = dynamic_cast<JobInstance *>(obj.get());
    return true;
}


int megamol::core::LuaState::GetParamType(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMGETPARAMTYPE)) {
        auto paramName = luaL_checkstring(L, 1);

        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());
        core::param::ParamSlot *ps = nullptr;
        if (getParamSlot(MMC_LUA_MMGETPARAMTYPE, paramName, &ps)) {

            auto psp = ps->Parameter();
            if (psp.IsNull()) {
                lua_pushstring(L, MMC_LUA_MMGETPARAMTYPE": ParamSlot does seem to hold no parameter");
                lua_error(L);
                return 0;
            }

            vislib::RawStorage pspdef;
            psp->Definition(pspdef);
            // not nice, but we make HEX (base64 would be better, but I don't care)
            std::string answer(pspdef.GetSize() * 2, ' ');
            for (SIZE_T i = 0; i < pspdef.GetSize(); ++i) {
                uint8_t b = *pspdef.AsAt<uint8_t>(i);
                uint8_t bh[2] = { static_cast<uint8_t>(b / 16), static_cast<uint8_t>(b % 16) };
                for (unsigned int j = 0; j < 2; ++j) answer[i * 2 + j] = (bh[j] < 10u) ? ('0' + bh[j]) : ('A' + (bh[j] - 10u));
            }

            lua_pushstring(L, answer.c_str());
            return 1;
        } else {
            // the error is already thrown
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::GetParamDescription(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMGETPARAMDESCRIPTION)) {
        auto paramName = luaL_checkstring(L, 1);

        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());
        core::param::ParamSlot *ps = nullptr;
        if (getParamSlot(MMC_LUA_MMGETPARAMDESCRIPTION, paramName, &ps)) {

            vislib::StringA valUTF8;
            vislib::UTF8Encoder::Encode(valUTF8, ps->Description());

            lua_pushstring(L, valUTF8);
            return 1;
        } else {
            // the error is already thrown
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::GetParamValue(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMGETPARAMVALUE)) {
        auto paramName = luaL_checkstring(L, 1);

        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());
        core::param::ParamSlot *ps = nullptr;
        if (getParamSlot(MMC_LUA_MMGETPARAMVALUE, paramName, &ps)) {

            auto psp = ps->Parameter();
            if (psp.IsNull()) {
                lua_pushstring(L, MMC_LUA_MMGETPARAMVALUE": ParamSlot does seem to hold no parameter");
                lua_error(L);
                return 0;
            }

            vislib::StringA valUTF8;
            vislib::UTF8Encoder::Encode(valUTF8, psp->ValueString());

            lua_pushstring(L, valUTF8);
            return 1;
        } else {
            // the error is already thrown
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::SetParamValue(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMSETPARAMVALUE)) {
        auto paramName = luaL_checkstring(L, 1);
        auto paramValue = luaL_checkstring(L, 2);

        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());
        core::param::ParamSlot *ps = nullptr;
        if (getParamSlot(MMC_LUA_MMSETPARAMVALUE, paramName, &ps)) {

            auto psp = ps->Parameter();
            if (psp.IsNull()) {
                lua_pushstring(L, MMC_LUA_MMSETPARAMVALUE": ParamSlot does seem to hold no parameter");
                lua_error(L);
                return 0;
            }

            vislib::TString val;
            vislib::UTF8Encoder::Decode(val, paramValue);

            if (psp->ParseValue(val)) {
                lua_pushstring(L, psp->ValueString());
                return 1;
            } else {
                lua_pushstring(L, MMC_LUA_MMSETPARAMVALUE": ParseValue failed");
                lua_error(L);
                return 0;
            }
        } else {
            // the error is already thrown
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::CreateModule(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMCREATEMODULE)) {
        //auto viewjobName = luaL_checkstring(L, 1);
        auto className = luaL_checkstring(L, 1);
        std::string instanceName(luaL_checkstring(L, 2));

        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());

        factories::ModuleDescription::ptr md = this->coreInst->GetModuleDescriptionManager().Find(vislib::StringA(className));
        if (md == NULL) {
            std::string out = "module class \"" + std::string(className) + "\" not found.";
            lua_pushstring(L, out.c_str());
            lua_error(L);
            return 0;
        }

        //core::ViewInstance *vi = nullptr;
        //core::JobInstance *ji = nullptr;
        //if (this->getView("mmCreateModule", viewjobName, &vi)) {

        //} else if (this->getJob("mmCreateModule", viewjobName, &ji)) {

        //} else {

        //}

        if (instanceName.compare(0, 2, "::") != 0) {
            std::string out = "instance name must be global (starting with \"::\")";
            lua_pushstring(L, out.c_str());
            lua_error(L);
            return 0;
        }

        auto mod = this->coreInst->instantiateModule(instanceName.c_str(), md);
        if (mod == nullptr) {
            std::string out = "could not create module (check MegaMol log)";
            lua_pushstring(L, out.c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::DeleteModule(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMDELETEMODULE)) {

    }
    return 0;
}


int megamol::core::LuaState::CreateCall(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMCREATECALL)) {
        auto from = luaL_checkstring(L, 1);
        auto to = luaL_checkstring(L, 2);
        auto className = luaL_checkstring(L, 3);

        factories::CallDescription::ptr cd = this->coreInst->GetCallDescriptionManager().Find(vislib::StringA(className));
        if (cd == NULL) {
            std::string out = "call class \"" + std::string(className) + "\" not found.";
            lua_pushstring(L, out.c_str());
            lua_error(L);
            return 0;
        }

        auto ca = this->coreInst->InstantiateCall(from, to, cd);
        if (ca == nullptr) {
            std::string out = "could not create call (check MegaMol log)";
            lua_pushstring(L, out.c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::DeleteCall(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMDELETECALL)) {

    }
    return 0;
}


int megamol::core::LuaState::CreateJob(lua_State *L) {
    lua_pushstring(L, "not implemented yet!");
    lua_error(L);
    return 0;
}


int megamol::core::LuaState::DeleteJob(lua_State *L) {
    lua_pushstring(L, "not implemented yet!");
    lua_error(L);
    return 0;
}


int megamol::core::LuaState::CreateView(lua_State *L) {
    lua_pushstring(L, "not implemented yet!");
    lua_error(L);
    return 0;
}


int megamol::core::LuaState::DeleteView(lua_State *L) {
    lua_pushstring(L, "not implemented yet!");
    lua_error(L);
    return 0;
}


void megamol::core::LuaState::queryModules(std::stringstream& reply, core::AbstractNamedObjectContainer::const_ptr_type anoc) {
    if (!anoc) return;

    reply << "Module: " << anoc.get()->FullName() << std::endl;
    reply << "Children:" << std::endl;
    auto it_end = anoc->ChildList_End();
    for (auto it = anoc->ChildList_Begin(); it != it_end; ++it) {
        AbstractNamedObject::const_ptr_type ano = *it;
        AbstractNamedObjectContainer::const_ptr_type anoc = std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
        if (anoc) {
            reply << anoc.get()->FullName() << std::endl;
        }
    }
    for (auto it = anoc->ChildList_Begin(); it != it_end; ++it) {
        AbstractNamedObject::const_ptr_type ano = *it;
        AbstractNamedObjectContainer::const_ptr_type anoc = std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
        if (anoc) {
            queryModules(reply, anoc);
        }
    }
}


int megamol::core::LuaState::QueryModules(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMQUERYMODULES)) {
        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());

        AbstractNamedObject::const_ptr_type ano = this->coreInst->ModuleGraphRoot();
        AbstractNamedObjectContainer::const_ptr_type anoc = std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
        if (!anoc) {
            lua_pushstring(L, MMC_LUA_MMQUERYMODULES": no root");
            lua_error(L);
            return 0;
        }

        std::stringstream answer;

        queryModules(answer, anoc);
        lua_pushstring(L, answer.str().c_str());
        return 1;
    }
    return 0;
}


int megamol::core::LuaState::Help(lua_State *L) {
    std::stringstream out;
    out << "MegaMol Lua Help:" << std::endl;
    for (auto &p : MM_LUA_HELP) {
        out << p.second << std::endl;
    }
    lua_pushstring(L, out.str().c_str());
    return 1;
}
