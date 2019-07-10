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
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
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
#include <map>
#include "vislib/sys/sysfunctions.h"
#include "vislib/sys/Process.h"
#include "vislib/sys/Environment.h"

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

// clang-format off
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
#define MMC_LUA_MMGETCONFIGVALUE "mmGetConfigValue"
#define MMC_LUA_MMGETPROCESSID "mmGetProcessID"
#define MMC_LUA_MMGETPARAMTYPE "mmGetParamType"
#define MMC_LUA_MMGETPARAMDESCRIPTION "mmGetParamDescription"
#define MMC_LUA_MMGETPARAMVALUE "mmGetParamValue"
#define MMC_LUA_MMSETPARAMVALUE "mmSetParamValue"
#define MMC_LUA_MMCREATEPARAMGROUP "mmCreateParamGroup"
#define MMC_LUA_MMSETPARAMGROUPVALUE "mmSetParamGroupValue"
#define MMC_LUA_MMCREATEMODULE "mmCreateModule"
#define MMC_LUA_MMDELETEMODULE "mmDeleteModule"
#define MMC_LUA_MMCREATECALL "mmCreateCall"
#define MMC_LUA_MMCREATECHAINCALL "mmCreateChainCall"
#define MMC_LUA_MMDELETECALL "mmDeleteCall"
#define MMC_LUA_MMCREATEVIEW "mmCreateView"
#define MMC_LUA_MMDELETEVIEW "mmDeleteView"
#define MMC_LUA_MMCREATEJOB "mmCreateJob"
#define MMC_LUA_MMDELETEJOB "mmDeleteJob"
#define MMC_LUA_MMQUERYMODULEGRAPH "mmQueryModuleGraph"
#define MMC_LUA_MMLISTMODULES "mmListModules"
#define MMC_LUA_MMLISTCALLS "mmListCalls"
#define MMC_LUA_MMLISTINSTANTIATIONS "mmListInstantiations"
#define MMC_LUA_MMGETENVVALUE "mmGetEnvValue"
#define MMC_LUA_MMHELP "mmHelp"
#define MMC_LUA_MMQUIT "mmQuit"
#define MMC_LUA_MMREADTEXTFILE "mmReadTextFile"
#define MMC_LUA_MMFLUSH "mmFlush"
#define MMC_LUA_MMCURRENTSCRIPTPATH "mmCurrentScriptPath"
#define MMC_LUA_MMLISTPARAMETERS "mmListParameters"


const std::map<std::string, std::string> MM_LUA_HELP = {
    { MMC_LUA_MMLOG, MMC_LUA_MMLOG"(int level, ...)\n\tLog to MegaMol console. Level constants are LOGINFO, LOGWARNING, LOGERROR." },
    { MMC_LUA_MMLOGINFO, MMC_LUA_MMLOGINFO"(...)\n\tLog to MegaMol console with LOGINFO level." },
    { MMC_LUA_MMGETBITHWIDTH, MMC_LUA_MMGETBITHWIDTH"()\n\tReturns the bit width of the compiled executable." },
    { MMC_LUA_MMGETCONFIGURATION, MMC_LUA_MMGETCONFIGURATION"()\n\tReturns the configuration ('debug' or 'release')." },
    { MMC_LUA_MMGETOS, MMC_LUA_MMGETOS"()\n\tReturns the operating system ('windows', 'linux', or 'unknown')."},
    { MMC_LUA_MMGETPROCESSID, MMC_LUA_MMGETPROCESSID"()\n\tReturns the process id of the running MegaMol." },
    { MMC_LUA_MMGETMACHINENAME, MMC_LUA_MMGETMACHINENAME"()\n\tReturns the machine name." },
    { MMC_LUA_MMSETAPPDIR, MMC_LUA_MMSETAPPDIR "(string dir)\n\tSets the path where the mmconsole.exe is located."},
    { MMC_LUA_MMADDSHADERDIR, MMC_LUA_MMADDSHADERDIR "(string dir)\n\tAdds a shader/btf search path."},
    { MMC_LUA_MMADDRESOURCEDIR, MMC_LUA_MMADDRESOURCEDIR "(string dir)\n\tAdds a resource search path."},
    { MMC_LUA_MMPLUGINLOADERINFO, MMC_LUA_MMPLUGINLOADERINFO"(string glob, string action)\n\tTell the core how to load plugins. Glob a path and ('include' | 'exclude') it." },
    { MMC_LUA_MMSETLOGFILE, MMC_LUA_MMSETLOGFILE "(string path)\n\tSets the full path of the log file."},
    { MMC_LUA_MMSETLOGLEVEL, MMC_LUA_MMSETLOGLEVEL"(int level)\n\tSets the level of log events to include. Level constants are: LOGINFO, LOGWARNING, LOGERROR." },
    { MMC_LUA_MMSETECHOLEVEL, MMC_LUA_MMSETECHOLEVEL"(int level)\n\tSets the level of log events to output to the console (see above)." },
    { MMC_LUA_MMSETCONFIGVALUE, MMC_LUA_MMSETCONFIGVALUE"(string name, string value)\n\tSets the config value <name> to <value>." },
    { MMC_LUA_MMGETCONFIGVALUE, MMC_LUA_MMGETCONFIGVALUE "(string name)\n\tGets the value of config value <name>."},
    { MMC_LUA_MMGETMODULEPARAMS, MMC_LUA_MMGETMODULEPARAMS"(string name)\n\tReturns a 0x1-separated list of module name and all parameters."
                                  "\n\tFor each parameter the name, description, definition, and value are returned."},
    { MMC_LUA_MMGETPARAMTYPE, MMC_LUA_MMGETPARAMTYPE"(string name)\n\tReturn the HEX type descriptor of a parameter slot." },
    { MMC_LUA_MMGETPARAMDESCRIPTION, MMC_LUA_MMGETPARAMDESCRIPTION"(string name)\n\tReturn the description of a parameter slot." },
    { MMC_LUA_MMGETPARAMVALUE, MMC_LUA_MMGETPARAMVALUE "(string name)\n\tReturn the value of a parameter slot."},
    { MMC_LUA_MMSETPARAMVALUE, MMC_LUA_MMSETPARAMVALUE"(string name, string value)\n\tSet the value of a parameter slot." },
    { MMC_LUA_MMCREATEPARAMGROUP, MMC_LUA_MMCREATEPARAMGROUP "(string name, string size)\n\tGenerate a param group that can only be set at once. Sets are queued until size is reached."},
    {MMC_LUA_MMSETPARAMGROUPVALUE, MMC_LUA_MMSETPARAMGROUPVALUE "(string groupname, string paramname, string value)\n\tQueue the value of a grouped parameter."},
    { MMC_LUA_MMCREATEMODULE, MMC_LUA_MMCREATEMODULE"(string className, string moduleName)\n\tCreate a module instance of class <className> called <moduleName>." },
    { MMC_LUA_MMDELETEMODULE, MMC_LUA_MMDELETEMODULE "(string name)\n\tDelete the module called <name>."},
    { MMC_LUA_MMCREATECALL, MMC_LUA_MMCREATECALL"(string className, string from, string to)\n\tCreate a call of type <className>, connecting CallerSlot <from> and CalleeSlot <to>." },
    { MMC_LUA_MMDELETECALL, MMC_LUA_MMDELETECALL"(string from, string to)\n\tDelete the call connecting CallerSlot <from> and CalleeSlot <to>." },
    { MMC_LUA_MMCREATECHAINCALL, MMC_LUA_MMCREATECHAINCALL
        "(string className, string chainStart, string to)\n\tAppend a call of type "
        "<className>, connection the rightmost CallerSlot starting at <chainStart> and CalleeSlot <to>."},
    { MMC_LUA_MMQUERYMODULEGRAPH, MMC_LUA_MMQUERYMODULEGRAPH "()\n\tShow the instantiated modules and their children."},
    { MMC_LUA_MMHELP, MMC_LUA_MMHELP "()\n\tShow this help."},
    { MMC_LUA_MMCREATEVIEW, MMC_LUA_MMCREATEVIEW"(string viewName, string viewModuleClass, string viewModuleName)"
        "\n\tCreate a new window/view and the according namespace <viewName> alongside it."
        "\n\tAlso, instantiate a view module called <viewModuleName> of <viewModuleClass> inside that window."},
    { MMC_LUA_MMDELETEVIEW, MMC_LUA_MMDELETEVIEW "TODO"},
    { MMC_LUA_MMCREATEJOB, MMC_LUA_MMCREATEJOB"(string jobName, string jobModuleClass, string jobModuleName)"
        "\n\tCreate a new background job and the according namespace <jobName> alongside it."
        "\n\tAlso, instantiate a job module called <jobModuleName> of <jobModuleClass> inside that window."},
    { MMC_LUA_MMDELETEJOB, MMC_LUA_MMDELETEJOB "TODO"},
    { MMC_LUA_MMGETENVVALUE, MMC_LUA_MMGETENVVALUE "(string name)\n\tReturn the value of env variable <name>."},
    { MMC_LUA_MMLISTCALLS, MMC_LUA_MMLISTCALLS"()\n\tReturn a list of instantiated calls (class id, instance id, from, to)."},
    { MMC_LUA_MMLISTINSTANTIATIONS, MMC_LUA_MMLISTINSTANTIATIONS "()\n\tReturn a list of instantiation names"},
    { MMC_LUA_MMLISTMODULES, MMC_LUA_MMLISTMODULES"(string basemodule_or_namespace)"
        "\n\tReturn a list of instantiated modules (class id, instance id), starting from a certain module downstream or inside a namespace."
        "\n\tWill use the graph root if an empty string is passed."},
    { MMC_LUA_MMQUIT, MMC_LUA_MMQUIT"()\n\tClose the MegaMol instance."},
    {MMC_LUA_MMREADTEXTFILE, MMC_LUA_MMREADTEXTFILE "(string fileName, function func)\n\tReturn the file contents after processing it with func(content)."},
    {MMC_LUA_MMFLUSH, MMC_LUA_MMFLUSH "()\n\tInserts a flush event into graph manipulation queues."},
    {MMC_LUA_MMCURRENTSCRIPTPATH, MMC_LUA_MMCURRENTSCRIPTPATH "()\n\tReturns the path of the currently running script, if possible. Empty string otherwise."},
    {MMC_LUA_MMLISTPARAMETERS, MMC_LUA_MMLISTPARAMETERS "(string baseModule_or_namespace)"
        "\n\tReturn all parameters, their type and value, starting from a certain module downstream or inside a namespace."
        "\n\tWill use the graph root if an empty string is passed."}
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
MMC_LUA_MMGETCONFIGVALUE "=" MMC_LUA_MMGETCONFIGVALUE ","
MMC_LUA_MMGETPROCESSID "=" MMC_LUA_MMGETPROCESSID ","
MMC_LUA_MMGETPARAMTYPE "=" MMC_LUA_MMGETPARAMTYPE ","
MMC_LUA_MMGETPARAMDESCRIPTION "=" MMC_LUA_MMGETPARAMDESCRIPTION ","
MMC_LUA_MMGETPARAMVALUE "=" MMC_LUA_MMGETPARAMVALUE ","
MMC_LUA_MMSETPARAMVALUE "=" MMC_LUA_MMSETPARAMVALUE ","
MMC_LUA_MMCREATEPARAMGROUP "=" MMC_LUA_MMCREATEPARAMGROUP ","
MMC_LUA_MMSETPARAMGROUPVALUE "=" MMC_LUA_MMSETPARAMGROUPVALUE ","
MMC_LUA_MMCREATEMODULE "=" MMC_LUA_MMCREATEMODULE ","
MMC_LUA_MMDELETEMODULE "=" MMC_LUA_MMDELETEMODULE ","
MMC_LUA_MMCREATECALL "=" MMC_LUA_MMCREATECALL ","
MMC_LUA_MMCREATECHAINCALL "=" MMC_LUA_MMCREATECHAINCALL ","
MMC_LUA_MMDELETECALL "=" MMC_LUA_MMDELETECALL ","
MMC_LUA_MMQUERYMODULEGRAPH "=" MMC_LUA_MMQUERYMODULEGRAPH ","
MMC_LUA_MMHELP "=" MMC_LUA_MMHELP ","
MMC_LUA_MMCREATEVIEW "=" MMC_LUA_MMCREATEVIEW ","
MMC_LUA_MMDELETEVIEW "=" MMC_LUA_MMDELETEVIEW ","
MMC_LUA_MMCREATEJOB "=" MMC_LUA_MMCREATEJOB ","
MMC_LUA_MMDELETEJOB "=" MMC_LUA_MMDELETEJOB ","
MMC_LUA_MMGETENVVALUE "=" MMC_LUA_MMGETENVVALUE ","
MMC_LUA_MMLISTCALLS "=" MMC_LUA_MMLISTCALLS ","
MMC_LUA_MMLISTMODULES "=" MMC_LUA_MMLISTMODULES ","
MMC_LUA_MMLISTINSTANTIATIONS "=" MMC_LUA_MMLISTINSTANTIATIONS ","
MMC_LUA_MMQUIT "=" MMC_LUA_MMQUIT ","
MMC_LUA_MMREADTEXTFILE "=" MMC_LUA_MMREADTEXTFILE ","
MMC_LUA_MMFLUSH "=" MMC_LUA_MMFLUSH ","
MMC_LUA_MMCURRENTSCRIPTPATH "=" MMC_LUA_MMCURRENTSCRIPTPATH ","
MMC_LUA_MMLISTPARAMETERS "=" MMC_LUA_MMLISTPARAMETERS ","
"  ipairs = ipairs,"
"  load = load,"
"  next = next,"
"  pairs = pairs,"
"  pcall = pcall,"
"  require = require,"
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
"      sort = table.sort, concat = table.concat },"
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
// clang-format on

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

void megamol::core::LuaState::consumeError(int error, char const* file, int line) const {
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
        lua_register(L, MMC_LUA_MMGETCONFIGVALUE, &dispatch<&LuaState::GetConfigValue>);

        lua_register(L, MMC_LUA_MMGETPROCESSID, &dispatch<&LuaState::GetProcessID>);
        lua_register(L, MMC_LUA_MMGETMODULEPARAMS, &dispatch<&LuaState::GetModuleParams>);
        lua_register(L, MMC_LUA_MMGETPARAMTYPE, &dispatch<&LuaState::GetParamType>);
        lua_register(L, MMC_LUA_MMGETPARAMDESCRIPTION, &dispatch<&LuaState::GetParamDescription>);
        lua_register(L, MMC_LUA_MMGETPARAMVALUE, &dispatch<&LuaState::GetParamValue>);
        lua_register(L, MMC_LUA_MMSETPARAMVALUE, &dispatch<&LuaState::SetParamValue>);
        lua_register(L, MMC_LUA_MMCREATEPARAMGROUP, &dispatch<&LuaState::CreateParamGroup>);
        lua_register(L, MMC_LUA_MMSETPARAMGROUPVALUE, &dispatch<&LuaState::SetParamGroupValue>);

        lua_register(L, MMC_LUA_MMCREATEMODULE, &dispatch<&LuaState::CreateModule>);
        lua_register(L, MMC_LUA_MMDELETEMODULE, &dispatch<&LuaState::DeleteModule>);
        lua_register(L, MMC_LUA_MMCREATECALL, &dispatch<&LuaState::CreateCall>);
        lua_register(L, MMC_LUA_MMCREATECHAINCALL, &dispatch<&LuaState::CreateChainCall>);
        lua_register(L, MMC_LUA_MMDELETECALL, &dispatch<&LuaState::DeleteCall>);

        lua_register(L, MMC_LUA_MMCREATEVIEW, &dispatch<&LuaState::CreateView>);
        lua_register(L, MMC_LUA_MMDELETEVIEW, &dispatch<&LuaState::DeleteView>);
        lua_register(L, MMC_LUA_MMCREATEJOB, &dispatch<&LuaState::CreateJob>);
        lua_register(L, MMC_LUA_MMDELETEJOB, &dispatch<&LuaState::DeleteJob>);

        lua_register(L, MMC_LUA_MMQUERYMODULEGRAPH, &dispatch<&LuaState::QueryModuleGraph>);

        lua_register(L, MMC_LUA_MMGETENVVALUE, &dispatch<&LuaState::GetEnvValue>);
        lua_register(L, MMC_LUA_MMLISTCALLS, &dispatch<&LuaState::ListCalls>);
        lua_register(L, MMC_LUA_MMLISTMODULES, &dispatch<&LuaState::ListModules>);
        lua_register(L, MMC_LUA_MMLISTINSTANTIATIONS, &dispatch<&LuaState::ListInstatiations>);
        lua_register(L, MMC_LUA_MMLISTPARAMETERS, &dispatch<&LuaState::ListParameters>);

        lua_register(L, MMC_LUA_MMGETENVVALUE, &dispatch<&LuaState::GetEnvValue>);

        lua_register(L, MMC_LUA_MMHELP, &dispatch<&LuaState::Help>);
        lua_register(L, MMC_LUA_MMQUIT, &dispatch<&LuaState::Quit>);

        lua_register(L, MMC_LUA_MMREADTEXTFILE, &dispatch<&LuaState::ReadTextFile>);

        lua_register(L, MMC_LUA_MMFLUSH, &dispatch<&LuaState::Flush>);
        lua_register(L, MMC_LUA_MMCURRENTSCRIPTPATH, &dispatch<&LuaState::CurrentScriptPath>);

#ifdef LUA_FULL_ENVIRONMENT
        // load all environment
        luaL_openlibs(L);
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
        luaL_requiref(L, LUA_LOADLIBNAME, luaopen_package, 1);
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


std::string megamol::core::LuaState::GetScriptPath(void) {
    return this->currentScriptPath;
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
        return RunString(envName, buffer.str(), result, fileName);
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
        const std::string narrowFileName(fileName.begin(), fileName.end());
        return RunString(envName, std::string(contents), result, narrowFileName);
    } else {
        return false;
    }
}


bool megamol::core::LuaState::RunString(const std::string& envName, const std::string& script, std::string& result, std::string scriptPath) {
    // no two threads can touch L at the same time
    std::lock_guard<std::mutex> stateGuard(this->stateLock);
    this->currentScriptPath = scriptPath;
    if (L != nullptr) {
        //vislib::sys::Log::DefaultLog.WriteInfo("trying to execute: %s", script.c_str());
        luaL_loadbuffer(L, script.c_str(), script.length(), "LuaState::RunString");
        lua_getglobal(L, envName.c_str());
        lua_setupvalue(L, -2, 1); // replace the environment with the one loaded from env.lua, disallowing some functions
        int old_n = lua_gettop(L);
        const int ret = lua_pcall(L, 0, LUA_MULTRET, 0);
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


bool megamol::core::LuaState::RunString(const std::string& script, std::string& result, std::string scriptPath) {
    return RunString("megamol_env", script, result, scriptPath);
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


int megamol::core::LuaState::GetConfigValue(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMGETCONFIGVALUE)) {
        std::stringstream out;
        auto name = luaL_checkstring(L, 1);
        mmcValueType t;
        const void *val = this->coreInst->Configuration().GetValue(MMC_CFGID_VARIABLE, name, &t);
        switch (t) {
            case MMC_TYPE_INT32:
                out << *(static_cast<const int32_t*>(val));
                break;
            case MMC_TYPE_UINT32:
                out << *(static_cast<const uint32_t*>(val));
                break;
            case MMC_TYPE_INT64:
                out << *(static_cast<const int64_t*>(val));
                break;
            case MMC_TYPE_UINT64:
                out << *(static_cast<const uint64_t*>(val));
                break;
            case MMC_TYPE_BYTE:
                out << *(static_cast<const char*>(val));
                break;
            case MMC_TYPE_BOOL:
                out << *(static_cast<const bool*>(val));
                break;
            case MMC_TYPE_FLOAT:
                out << *(static_cast<const float*>(val));
                break;
            case MMC_TYPE_CSTR:
                out << *(static_cast<const char*>(val));
                break;
            case MMC_TYPE_WSTR:
                out << vislib::StringA(vislib::StringW(static_cast<const wchar_t*>(val)));
                break;
            default:
                // also includes MMC_TYPE_VOIDP
                out << "unknown";
                break;
        }
        lua_pushstring(L, out.str().c_str());
        return 1;
    }
    return 0;
}


int megamol::core::LuaState::GetEnvValue(lua_State *L) {
    auto name = luaL_checkstring(L, 1);
    if (vislib::sys::Environment::IsSet(name)) {
        lua_pushstring(L, vislib::sys::Environment::GetVariable(name));
        return 1;
    } else {
        lua_pushstring(L, "undef");
        return 1;
    }
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


int megamol::core::LuaState::GetModuleParams(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMGETMODULEPARAMS)) {
        auto moduleName = luaL_checkstring(L, 1);

        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());


        auto ret = this->coreInst->EnumerateParameterSlotsNoLock<megamol::core::Module>(
            moduleName, [L](param::ParamSlot& ps) {
                std::stringstream answer;
                Module* mod = dynamic_cast<Module*>(ps.Parent().get());
                if (mod != nullptr) {
                    vislib::StringA name(mod->FullName());
                    answer << name << "\1";
                    AbstractNamedObjectContainer::child_list_type::iterator si, se;
                    se = mod->ChildList_End();
                    for (si = mod->ChildList_Begin(); si != se; ++si) {
                        param::ParamSlot* slot = dynamic_cast<param::ParamSlot*>((*si).get());
                        if (slot != NULL) {
                            // name.Append("::");
                            // name.Append(slot->Name());

                            answer << slot->Name() << "\1";

                            vislib::StringA descUTF8;
                            vislib::UTF8Encoder::Encode(descUTF8, slot->Description());
                            answer << descUTF8 << "\1";

                            auto psp = slot->Parameter();
                            if (psp.IsNull()) {
                                std::ostringstream err;
                                err << MMC_LUA_MMGETMODULEPARAMS ": ParamSlot " << slot->FullName()
                                    << " does seem to hold no parameter";
                                lua_pushstring(L, err.str().c_str());
                                lua_error(L);
                            }

                            vislib::RawStorage pspdef;
                            psp->Definition(pspdef);
                            // not nice, but we make HEX (base64 would be better, but I don't care)
                            std::string answer2(pspdef.GetSize() * 2, ' ');
                            for (SIZE_T i = 0; i < pspdef.GetSize(); ++i) {
                                uint8_t b = *pspdef.AsAt<uint8_t>(i);
                                uint8_t bh[2] = {static_cast<uint8_t>(b / 16), static_cast<uint8_t>(b % 16)};
                                for (unsigned int j = 0; j < 2; ++j)
                                    answer2[i * 2 + j] = (bh[j] < 10u) ? ('0' + bh[j]) : ('A' + (bh[j] - 10u));
                            }
                            answer << answer2 << "\1";

                            vislib::StringA valUTF8;
                            vislib::UTF8Encoder::Encode(valUTF8, psp->ValueString());

                            answer << valUTF8 << "\1";
                        }
                    }
                    lua_pushstring(L, answer.str().c_str());
                } else {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "LuaState: ParamSlot %s has a parent which is not a Module!", ps.FullName().PeekBuffer());
                }
            });
        return ret ? 1 : 0;
    }
    return 0;
}


bool megamol::core::LuaState::getParamSlot(const std::string routine, const char *paramName, core::param::ParamSlot **out) {

    AbstractNamedObjectContainer::ptr_type root = std::dynamic_pointer_cast<AbstractNamedObjectContainer>(this->coreInst->namespaceRoot);
    if (!root) {
        std::string err = routine + ": no root";
        lua_pushstring(L, err.c_str());
        lua_error(L);
        return false;
    }
    AbstractNamedObject::ptr_type obj = root.get()->FindNamedObject(paramName);
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

    AbstractNamedObjectContainer::ptr_type root = AbstractNamedObjectContainer::dynamic_pointer_cast(this->coreInst->namespaceRoot);
    if (!root) {
        std::string err = routine + ": no root";
        lua_pushstring(L, err.c_str());
        lua_error(L);
        return false;
    }
    AbstractNamedObject::ptr_type obj = root.get()->FindNamedObject(viewName);
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
    AbstractNamedObjectContainer::ptr_type root = AbstractNamedObjectContainer::dynamic_pointer_cast(this->coreInst->namespaceRoot);
    if (!root) {
        std::string err = routine + ": no root";
        lua_pushstring(L, err.c_str());
        lua_error(L);
        return false;
    }
    AbstractNamedObject::ptr_type obj = root.get()->FindNamedObject(jobName);
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

        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
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

        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
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

        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
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

        if (!this->coreInst->RequestParamValue(paramName, paramValue)) {
            std::stringstream out;
            out << "could not set \"";
            out << paramName;
            out << "\" to \"";
            out << paramValue;
            out << "\" (check MegaMol log)";
            lua_pushstring(L, out.str().c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::CreateParamGroup(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMCREATEPARAMGROUP)) {
        auto groupName = luaL_checkstring(L, 1);
        auto groupSize = luaL_checkinteger(L, 2);

        if (!this->coreInst->CreateParamGroup(groupName, groupSize)) {
            std::stringstream out;
            out << "could not create param group \"";
            out << groupName;
            out << "\" with size \"";
            out << groupSize;
            out << "\" (check MegaMol log)";
            lua_pushstring(L, out.str().c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::SetParamGroupValue(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMSETPARAMGROUPVALUE)) {
        auto paramGroup = luaL_checkstring(L, 1);
        auto paramName = luaL_checkstring(L, 2);
        auto paramValue = luaL_checkstring(L, 3);

        if (!this->coreInst->RequestParamGroupValue(paramGroup, paramName, paramValue)) {
            std::stringstream out;
            out << "could not set \"";
            out << paramName;
            out << "\" in group \"";
            out << paramGroup;
            out << "\" to \"";
            out << paramValue;
            out << "\" (check MegaMol log)";
            lua_pushstring(L, out.str().c_str());
            lua_error(L);
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

        if (instanceName.compare(0, 2, "::") != 0) {
            std::string out = "instance name \"" + instanceName +
                "\" must be global (starting with \"::\")";
            lua_pushstring(L, out.c_str());
            lua_error(L);
            return 0;
        }

        if (!this->coreInst->RequestModuleInstantiation(className, instanceName.c_str())) {
            std::stringstream out;
            out << "could not create \"";
            out << className;
            out << "\" module (check MegaMol log)";
            lua_pushstring(L, out.str().c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::DeleteModule(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMDELETEMODULE)) {
        auto moduleName = luaL_checkstring(L, 1);

        if (!this->coreInst->RequestModuleDeletion(moduleName)) {
            lua_pushstring(L, ("cannot delete module \"" + std::string(moduleName) +
                "\" (check MegaMol log)").c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::CreateCall(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMCREATECALL)) {
        auto className = luaL_checkstring(L, 1);
        auto from = luaL_checkstring(L, 2);
        auto to = luaL_checkstring(L, 3);

        if(!this->coreInst->RequestCallInstantiation(className, from, to)) {
            std::stringstream out;
            out << "could not create \"";
            out << className;
            out << "\" call (check MegaMol log)";
            lua_pushstring(L, out.str().c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}

int megamol::core::LuaState::CreateChainCall(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMCREATECHAINCALL)) {
        auto className = luaL_checkstring(L, 1);
        std::string chainStart = luaL_checkstring(L, 2);
        std::string to = luaL_checkstring(L, 3);

        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
         vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());

        auto pos = chainStart.find_last_of("::");
        if (pos < 4 || chainStart.length() < pos + 2) {
            lua_pushstring(L, MMC_LUA_MMCREATECHAINCALL ": chainStart module/slot name weird");
            lua_error(L);
            return 0;
        }
        auto moduleName = chainStart.substr(0, pos-1);
        auto slotName = chainStart.substr(pos + 1, -1);

        if (!this->coreInst->RequestChainCallInstantiation(className, chainStart.c_str(), to.c_str())) {
            std::stringstream out;
            out << "could not create \"";
            out << className;
            out << "\" call (check MegaMol log)";
            lua_pushstring(L, out.str().c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::DeleteCall(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMDELETECALL)) {
        auto from = luaL_checkstring(L, 1);
        auto to = luaL_checkstring(L, 2);

        if (!this->coreInst->RequestCallDeletion(from, to)) {
            lua_pushstring(L, ("cannot delete call from \"" + std::string(from) +
                "\" to \"" + std::string(to) + "\" (check MegaMol log)").c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::CreateJob(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMCREATEJOB)) {
        auto jobName = luaL_checkstring(L, 1);
        auto className = luaL_checkstring(L, 2);
        auto moduleName = luaL_checkstring(L, 3);

        auto jd = std::make_shared<JobDescription>(jobName);
        jd->AddModule(this->coreInst->GetModuleDescriptionManager().Find(className), moduleName);
        jd->SetJobModuleID(moduleName);
        try {
            this->coreInst->projJobDescs.Register(jd);
            this->coreInst->RequestJobInstantiation(jd.get(), jobName);
        } catch (vislib::AlreadyExistsException) {
            lua_pushstring(L, ("job \"" + std::string(jobName) + "\" already exists.").c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::DeleteJob(lua_State *L) {
    lua_pushstring(L, "not implemented yet!");
    lua_error(L);
    return 0;
}


int megamol::core::LuaState::CreateView(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMCREATEVIEW)) {
        auto viewName = luaL_checkstring(L, 1);
        auto className = luaL_checkstring(L, 2);
        auto moduleName = luaL_checkstring(L, 3);

        auto vd = std::make_shared<ViewDescription>(viewName);
        vd->AddModule(this->coreInst->GetModuleDescriptionManager().Find(className), moduleName);
        vd->SetViewModuleID(moduleName);
        try {
            this->coreInst->projViewDescs.Register(vd);
            this->coreInst->RequestViewInstantiation(vd.get(), viewName);
        } catch (vislib::AlreadyExistsException) {
            lua_pushstring(L, ("view \"" + std::string(viewName) + "\" already exists.").c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::DeleteView(lua_State *L) {
    lua_pushstring(L, "not implemented yet!");
    lua_error(L);
    return 0;
}


int megamol::core::LuaState::QueryModuleGraph(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMQUERYMODULEGRAPH)) {
        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());

        AbstractNamedObject::const_ptr_type ano = this->coreInst->ModuleGraphRoot();
        AbstractNamedObjectContainer::const_ptr_type anoc = std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
        if (!anoc) {
            lua_pushstring(L, MMC_LUA_MMQUERYMODULEGRAPH": no root");
            lua_error(L);
            return 0;
        }

        std::stringstream answer;

        //queryModules(answer, anoc);
        std::vector<AbstractNamedObjectContainer::const_ptr_type> anoStack;
        anoStack.push_back(anoc);
        while (!anoStack.empty()) {
            anoc = anoStack.back();
            anoStack.pop_back();
            
            if (anoc) {
                const auto m = Module::dynamic_pointer_cast(anoc);
                answer << (m != nullptr ? "Module:    " : "Container: ") << anoc.get()->FullName() << std::endl;
                if (anoc.get()->Parent() != nullptr) {
                    answer << "Parent:    " << anoc.get()->Parent()->FullName() << std::endl;
                } else {
                    answer << "Parent:    none" << std::endl;
                }
                const char *cn = nullptr;
                if (m != nullptr) {
                    cn = m->ClassName();
                }
                answer << "Class:     " << ((cn != nullptr) ? cn : "unknown") << std::endl;
                answer << "Children:  ";
                auto it_end = anoc->ChildList_End();
                int numChildren = 0;
                for (auto it = anoc->ChildList_Begin(); it != it_end; ++it) {
                    AbstractNamedObject::const_ptr_type ano = *it;
                    AbstractNamedObjectContainer::const_ptr_type anoc = std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
                    if (anoc) {
                        if (numChildren == 0) {
                            answer << std::endl;
                        }
                        answer << anoc.get()->FullName() << std::endl;
                        numChildren++;
                    }
                }
                for (auto it = anoc->ChildList_Begin(); it != it_end; ++it) {
                    AbstractNamedObject::const_ptr_type ano = *it;
                    AbstractNamedObjectContainer::const_ptr_type anoc = std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
                    if (anoc) {
                        anoStack.push_back(anoc);
                    }
                }
                if (numChildren == 0) {
                    answer << "none" << std::endl;
                }
            }
        }

        lua_pushstring(L, answer.str().c_str());
        return 1;
    }
    return 0;
}

int megamol::core::LuaState::ListCalls(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMLISTCALLS)) {

        const int n = lua_gettop(L);

        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());

        std::stringstream answer;

        const auto fun = [&answer](Module* mod) {
            AbstractNamedObjectContainer::child_list_type::const_iterator se = mod->ChildList_End();
            for (AbstractNamedObjectContainer::child_list_type::const_iterator si = mod->ChildList_Begin(); si != se; ++si) {
                const auto slot = dynamic_cast<CallerSlot*>((*si).get());
                if (slot) {
                    const Call *c = const_cast<CallerSlot *>(slot)->CallAs<Call>();
                    if (c != nullptr) {
                        answer << c->ClassName() << ";"
                        << c->PeekCallerSlot()->Parent()->Name() << "," << c->PeekCalleeSlot()->Parent()->Name() << ";"
                        << c->PeekCallerSlot()->Name() << "," << c->PeekCalleeSlot()->Name() << std::endl;
                    }
                }
            }
        };

        if (n == 1) {
            const auto starting_point = luaL_checkstring(L, 1);
            if (!std::string(starting_point).empty()) {
                this->coreInst->EnumModulesNoLock(starting_point, fun);
            } else {
                this->coreInst->EnumModulesNoLock(nullptr, fun);
            }
        } else {
            this->coreInst->EnumModulesNoLock(nullptr, fun);
        }
        
        lua_pushstring(L, answer.str().c_str());
        return 1;
    }
    return 0;
}


int megamol::core::LuaState::ListModules(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMLISTMODULES)) {

        const int n = lua_gettop(L);

        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
        //vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());
        if (this->coreInst->ModuleGraphRoot()->ModuleGraphLock().TryLock(100)) {

            std::stringstream answer;

            const auto fun = [&answer](Module* mod) { answer << mod->ClassName() << ";" << mod->Name() << std::endl; };

            if (n == 1) {
                const auto starting_point = luaL_checkstring(L, 1);
                if (!std::string(starting_point).empty()) {
                    this->coreInst->EnumModulesNoLock(starting_point, fun);
                } else {
                    this->coreInst->EnumModulesNoLock(nullptr, fun);
                }
            } else {
                this->coreInst->EnumModulesNoLock(nullptr, fun);
            }

            lua_pushstring(L, answer.str().c_str());
            this->coreInst->ModuleGraphRoot()->ModuleGraphLock().Unlock();
            return 1;            
        } else {
            std::stringstream answer;
            answer << "Could not acquire module graph lock" << std::endl;
            lua_pushstring(L, answer.str().c_str());
            return 1;
        }
    }
    return 0;
}

int megamol::core::LuaState::ListInstatiations(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMLISTINSTANTIATIONS)) {
        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());

        AbstractNamedObject::const_ptr_type ano = this->coreInst->ModuleGraphRoot();
        AbstractNamedObjectContainer::const_ptr_type anor = std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
        if (!ano) {
            lua_pushstring(L, MMC_LUA_MMLISTINSTANTIATIONS": no root");
            lua_error(L);
            return 0;
        }

        std::stringstream answer;

        if (anor) {
            const auto it_end = anor->ChildList_End();
            for (auto it = anor->ChildList_Begin(); it != it_end; ++it) {
                if (!dynamic_cast<const Module *>(it->get())) {
                    AbstractNamedObjectContainer::const_ptr_type anoc = std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(*it);
                    answer << anoc->FullName() << std::endl;
                    // TODO: the immediate child view should be it, generally
                }
            }
        }

        lua_pushstring(L, answer.str().c_str());
        return 1;
    }
    return 0;
}

int megamol::core::LuaState::ListParameters(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMLISTPARAMETERS)) {

        const int n = lua_gettop(L);

        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());

        std::stringstream answer;

        const auto fun = [&answer](Module* mod) {
            AbstractNamedObjectContainer::child_list_type::const_iterator se = mod->ChildList_End();
            for (AbstractNamedObjectContainer::child_list_type::const_iterator si = mod->ChildList_Begin(); si != se; ++si) {
                const auto slot = dynamic_cast<param::ParamSlot*>((*si).get());
                if (slot) {
                    answer << slot->FullName() << "\1" << slot->Parameter()->ValueString() << "\1";
                }
            }
        };

        if (n == 1) {
            const auto starting_point = luaL_checkstring(L, 1);
            if (!std::string(starting_point).empty()) {
                this->coreInst->EnumModulesNoLock(starting_point, fun);
            } else {
                this->coreInst->EnumModulesNoLock(nullptr, fun);
            }
        } else {
            this->coreInst->EnumModulesNoLock(nullptr, fun);
        }
        
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

int megamol::core::LuaState::Quit(lua_State *L) {
    if (this->checkRunning(MMC_LUA_MMQUIT)) {
        this->coreInst->Shutdown();
    }
    return 0;
}

int megamol::core::LuaState::ReadTextFile(lua_State* L) {
    int n = lua_gettop(L);
    if (n == 2) {
        const auto filename = luaL_checkstring(L, 1);
        std::ifstream t(filename);
        if (t.good()) {
            std::stringstream buffer;
            buffer << t.rdbuf();

            //vislib::sys::Log::DefaultLog.WriteInfo(MMC_LUA_MMREADTEXTFILE ": read from file '%s':\n%s\n", filename, buffer.str().c_str());

            lua_remove(L, 1); // get rid of the filename on the stack, leaving the function pointer
            lua_pushstring(L, buffer.str().c_str()); // put string parameter on top of stack
            if (lua_type(L, 1) == LUA_TNIL) {
                // no transformation function, just return the string
                return 1;
            } else {
                // call the function pointer
                lua_pcall(L, 1, 1, 0);
                n = lua_gettop(L);
                if (n != 1) {
                    std::string err = MMC_LUA_MMREADTEXTFILE ": function did not return a string, this is bad.";
                    lua_pushstring(L, err.c_str());
                    lua_error(L);
                } else {
                    const auto newString = luaL_checkstring(L, 1);
                    // vislib::sys::Log::DefaultLog.WriteInfo(MMC_LUA_MMREADTEXTFILE ": transformed into:\n%s\n",
                    // newString);
                    return 1;
                }
            }
        } else {
            std::string err = MMC_LUA_MMREADTEXTFILE ": cannot open file '";
            err += filename;
            err += "'.";
            lua_pushstring(L, err.c_str());
            lua_error(L);
        }
    } else {
        std::string err =
            MMC_LUA_MMREADTEXTFILE " requires two parameters, fileName and a function pointer";
        lua_pushstring(L, err.c_str());
        lua_error(L);
    }
    return 0;
}

int megamol::core::LuaState::Flush(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMFLUSH)) {
        this->coreInst->FlushGraphUpdates();
    }

    return 0;
}

int megamol::core::LuaState::CurrentScriptPath(struct lua_State* L) {
    lua_pushstring(L, this->currentScriptPath.c_str());
    return 1;
}

