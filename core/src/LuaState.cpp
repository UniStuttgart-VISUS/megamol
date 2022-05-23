/*
 * LuaState.cpp
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#if (_MSC_VER > 1000)
#pragma warning(disable : 4996)
#endif /* (_MSC_VER > 1000) */
#if (_MSC_VER > 1000)
#pragma warning(default : 4996)
#endif /* (_MSC_VER > 1000) */

#include "mmcore/LuaState.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/utility/Configuration.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/sys/SystemInformation.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/Environment.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/sysfunctions.h"
#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "lua.hpp"

#ifdef _WIN32
#include <Windows.h>
#else /* _WIN32 */
#include <unistd.h>
#endif /* _WIN32 */

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
#define MMC_LUA_MMQUIT "mmQuit"
#define MMC_LUA_MMREADTEXTFILE "mmReadTextFile"
#define MMC_LUA_MMFLUSH "mmFlush"
#define MMC_LUA_MMCURRENTSCRIPTPATH "mmCurrentScriptPath"
#define MMC_LUA_MMLISTPARAMETERS "mmListParameters"


bool megamol::core::LuaState::checkConfiguring(const std::string where) {
    if (this->conf != nullptr) {
        return true;
    } else {
        std::string err = where + "is only legal when reading the configuration";
        theLua.ThrowError(err);
        return false;
    }
}


bool megamol::core::LuaState::checkRunning(const std::string where) {
    if (this->coreInst != nullptr) {
        return true;
    } else {
        std::string err = where + "is only legal when MegaMol is running";
        theLua.ThrowError(err);
        return false;
    }
}


void megamol::core::LuaState::commonInit() {
    theLua.RegisterCallback<LuaState, &LuaState::GetOS>(MMC_LUA_MMGETOS, "()\n\tReturns the operating system ('windows', 'linux', or 'unknown').");
    theLua.RegisterCallback<LuaState, &LuaState::GetBitWidth>(MMC_LUA_MMGETBITHWIDTH, "()\n\tReturns the bit width of the compiled executable.");
    theLua.RegisterCallback<LuaState, &LuaState::GetConfiguration>(MMC_LUA_MMGETCONFIGURATION, "()\n\tReturns the configuration ('debug' or 'release').");
    theLua.RegisterCallback<LuaState, &LuaState::GetMachineName>(MMC_LUA_MMGETMACHINENAME, "()\n\tReturns the machine name.");

    theLua.RegisterCallback<LuaState, &LuaState::SetAppDir>(MMC_LUA_MMSETAPPDIR, "(string dir)\n\tSets the path where the mmconsole.exe is located.");
    theLua.RegisterCallback<LuaState, &LuaState::AddShaderDir>(MMC_LUA_MMADDSHADERDIR, "(string dir)\n\tAdds a shader/btf search path.");
    theLua.RegisterCallback<LuaState, &LuaState::AddResourceDir>(MMC_LUA_MMADDRESOURCEDIR, "(string dir)\n\tAdds a resource search path.");
    theLua.RegisterCallback<LuaState, &LuaState::PluginLoaderInfo>(MMC_LUA_MMPLUGINLOADERINFO, "(string glob, string action)\n\tTell the core how to load plugins. Glob a path and ('include' | 'exclude') it.");

    theLua.RegisterCallback<LuaState, &LuaState::SetLogFile>(MMC_LUA_MMSETLOGFILE, "(string path)\n\tSets the full path of the log file.");
    theLua.RegisterCallback<LuaState, &LuaState::SetLogLevel>(MMC_LUA_MMSETLOGLEVEL, "(int level)\n\tSets the level of log events to include. Level constants are: LOGINFO, LOGWARNING, LOGERROR.");
    theLua.RegisterCallback<LuaState, &LuaState::SetEchoLevel>(MMC_LUA_MMSETECHOLEVEL, "(int level)\n\tSets the level of log events to output to the console (see above).");

    theLua.RegisterCallback<LuaState, &LuaState::SetConfigValue>(MMC_LUA_MMSETCONFIGVALUE, "(string name, string value)\n\tSets the config value <name> to <value>.");
    theLua.RegisterCallback<LuaState, &LuaState::GetConfigValue>(MMC_LUA_MMGETCONFIGVALUE, "(string name)\n\tGets the value of config value <name>.");

    theLua.RegisterCallback<LuaState, &LuaState::GetProcessID>(MMC_LUA_MMGETPROCESSID, "()\n\tReturns the process id of the running MegaMol.");
    theLua.RegisterCallback<LuaState, &LuaState::GetModuleParams>(MMC_LUA_MMGETMODULEPARAMS, "(string name)\n\tReturns a 0x1-separated list of module name and all parameters."
                              "\n\tFor each parameter the name, description, definition, and value are returned.");
    theLua.RegisterCallback<LuaState, &LuaState::GetParamType>(MMC_LUA_MMGETPARAMTYPE, "(string name)\n\tReturn the HEX type descriptor of a parameter slot.");
    theLua.RegisterCallback<LuaState, &LuaState::GetParamDescription>(MMC_LUA_MMGETPARAMDESCRIPTION, "(string name)\n\tReturn the description of a parameter slot.");
    theLua.RegisterCallback<LuaState, &LuaState::GetParamValue>(MMC_LUA_MMGETPARAMVALUE, "(string name)\n\tReturn the value of a parameter slot.");
    theLua.RegisterCallback<LuaState, &LuaState::SetParamValue>(MMC_LUA_MMSETPARAMVALUE, "(string name, string value)\n\tSet the value of a parameter slot.");
    theLua.RegisterCallback<LuaState, &LuaState::CreateParamGroup>(MMC_LUA_MMCREATEPARAMGROUP, "(string name, string size)\n\tGenerate a param group that can only be set at once. Sets are queued until size is reached.");
    theLua.RegisterCallback<LuaState, &LuaState::SetParamGroupValue>(MMC_LUA_MMSETPARAMGROUPVALUE, "(string groupname, string paramname, string value)\n\tQueue the value of a grouped parameter.");

    theLua.RegisterCallback<LuaState, &LuaState::CreateModule>(MMC_LUA_MMCREATEMODULE, "(string className, string moduleName)\n\tCreate a module instance of class <className> called <moduleName>.");
    theLua.RegisterCallback<LuaState, &LuaState::DeleteModule>(MMC_LUA_MMDELETEMODULE, "(string name)\n\tDelete the module called <name>.");
    theLua.RegisterCallback<LuaState, &LuaState::CreateCall>(MMC_LUA_MMCREATECALL, "(string className, string from, string to)\n\tCreate a call of type <className>, connecting CallerSlot <from> and CalleeSlot <to>.");
    theLua.RegisterCallback<LuaState, &LuaState::CreateChainCall>(MMC_LUA_MMCREATECHAINCALL, "(string className, string chainStart, string to)\n\tAppend a call of type "
    "<className>, connection the rightmost CallerSlot starting at <chainStart> and CalleeSlot <to>.");
    theLua.RegisterCallback<LuaState, &LuaState::DeleteCall>(MMC_LUA_MMDELETECALL, "(string from, string to)\n\tDelete the call connecting CallerSlot <from> and CalleeSlot <to>.");

    theLua.RegisterCallback<LuaState, &LuaState::CreateView>(MMC_LUA_MMCREATEVIEW, "(string viewName, string viewModuleClass, string viewModuleName)"
    "\n\tCreate a new window/view and the according namespace <viewName> alongside it."
    "\n\tAlso, instantiate a view module called <viewModuleName> of <viewModuleClass> inside that window.");
    theLua.RegisterCallback<LuaState, &LuaState::DeleteView>(MMC_LUA_MMDELETEVIEW, "TODO");
    theLua.RegisterCallback<LuaState, &LuaState::CreateJob>(MMC_LUA_MMCREATEJOB, "(string jobName, string jobModuleClass, string jobModuleName)"
    "\n\tCreate a new background job and the according namespace <jobName> alongside it."
    "\n\tAlso, instantiate a job module called <jobModuleName> of <jobModuleClass> inside that job.");
    theLua.RegisterCallback<LuaState, &LuaState::DeleteJob>(MMC_LUA_MMDELETEJOB, "TODO");

    theLua.RegisterCallback<LuaState, &LuaState::QueryModuleGraph>(MMC_LUA_MMQUERYMODULEGRAPH, "()\n\tShow the instantiated modules and their children.");

    theLua.RegisterCallback<LuaState, &LuaState::GetEnvValue>(MMC_LUA_MMGETENVVALUE, "(string name)\n\tReturn the value of env variable <name>.");
    theLua.RegisterCallback<LuaState, &LuaState::ListCalls>(MMC_LUA_MMLISTCALLS, "()\n\tReturn a list of instantiated calls (class id, instance id, from, to).");
    theLua.RegisterCallback<LuaState, &LuaState::ListModules>(MMC_LUA_MMLISTMODULES, "(string basemodule_or_namespace)"
    "\n\tReturn a list of instantiated modules (class id, instance id), starting from a certain module downstream or inside a namespace."
    "\n\tWill use the graph root if an empty string is passed.");
    theLua.RegisterCallback<LuaState, &LuaState::ListInstatiations>(MMC_LUA_MMLISTINSTANTIATIONS, "()\n\tReturn a list of instantiation names");
    theLua.RegisterCallback<LuaState, &LuaState::ListParameters>(MMC_LUA_MMLISTPARAMETERS, "(string baseModule_or_namespace)"
    "\n\tReturn all parameters, their type and value, starting from a certain module downstream or inside a namespace."
    "\n\tWill use the graph root if an empty string is passed.");

    theLua.RegisterCallback<LuaState, &LuaState::Quit>(MMC_LUA_MMQUIT, "()\n\tClose the MegaMol instance.");

    theLua.RegisterCallback<LuaState, &LuaState::ReadTextFile>(MMC_LUA_MMREADTEXTFILE, "(string fileName, function func)\n\tReturn the file contents after processing it with func(content).");

    theLua.RegisterCallback<LuaState, &LuaState::Flush>(MMC_LUA_MMFLUSH, "()\n\tInserts a flush event into graph manipulation queues.");
    theLua.RegisterCallback<LuaState, &LuaState::CurrentScriptPath>(MMC_LUA_MMCURRENTSCRIPTPATH, "()\n\tReturns the path of the currently running script, if possible. Empty string otherwise.");
}


/*
 * megamol::core::LuaState::LuaState
 */
megamol::core::LuaState::LuaState(CoreInstance* inst) : coreInst(inst), conf(nullptr), theLua(this) {
    this->commonInit();
}


megamol::core::LuaState::LuaState(utility::Configuration* conf) : coreInst(nullptr), conf(conf), theLua(this) {
    this->commonInit();
}


/*
 * megamol::core::LuaState::~LuaState
 */
megamol::core::LuaState::~LuaState() {

}


bool megamol::core::LuaState::StateOk() { return true; }


std::string megamol::core::LuaState::GetScriptPath(void) { return this->currentScriptPath; }


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
    if (input.Open(fileName.c_str(), vislib::sys::File::AccessMode::READ_ONLY, vislib::sys::File::ShareMode::SHARE_READ,
            vislib::sys::File::CreationMode::OPEN_ONLY)) {
        vislib::StringA contents;
        vislib::sys::ReadTextFile(contents, input);
        input.Close();
        const std::string narrowFileName(fileName.begin(), fileName.end());
        return RunString(envName, std::string(contents), result, narrowFileName);
    } else {
        return false;
    }
}


bool megamol::core::LuaState::RunString(
    const std::string& envName, const std::string& script, std::string& result, std::string scriptPath) {
    // todo: locking!!!
    // no two threads can touch L at the same time
    std::lock_guard<std::mutex> stateGuard(this->stateLock);
    if (this->currentScriptPath.empty() && !scriptPath.empty()) {
        // the information got better, at least
        this->currentScriptPath = scriptPath;
    }
    return theLua.RunString(envName, script, result);
}


bool megamol::core::LuaState::RunFile(const std::string& fileName, std::string& result) {
    return RunFile("default_env", fileName, result);
}


bool megamol::core::LuaState::RunFile(const std::wstring& fileName, std::string& result) {
    return RunFile("default_env", fileName, result);
}


bool megamol::core::LuaState::RunString(const std::string& script, std::string& result, std::string scriptPath) {
    return RunString("default_env", script, result, scriptPath);
}


int megamol::core::LuaState::GetBitWidth(lua_State* L) {
    lua_pushinteger(L, vislib::sys::SystemInformation::SelfWordSize());
    return 1;
}


int megamol::core::LuaState::GetConfiguration(lua_State* L) {
#ifdef _DEBUG
    lua_pushstring(L, "debug");
#else
    lua_pushstring(L, "release");
#endif
    return 1;
}


int megamol::core::LuaState::GetOS(lua_State* L) {
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


int megamol::core::LuaState::GetMachineName(lua_State* L) {
    lua_pushstring(L, vislib::sys::SystemInformation::ComputerNameA());
    return 1;
}


int megamol::core::LuaState::SetAppDir(lua_State* L) {
    if (this->checkConfiguring(MMC_LUA_MMSETAPPDIR)) {
        // TODO do we need to make an OS-dependent path here?
        auto p = luaL_checkstring(L, 1);
        this->conf->appDir = vislib::StringW(p);
    }
    return 0;
}


int megamol::core::LuaState::AddShaderDir(lua_State* L) {
    if (this->checkConfiguring(MMC_LUA_MMADDSHADERDIR)) {
        // TODO do we need to make an OS-dependent path here?
        auto p = luaL_checkstring(L, 1);
        this->conf->AddShaderDirectory(p);
    }
    return 0;
}


int megamol::core::LuaState::AddResourceDir(lua_State* L) {
    if (this->checkConfiguring(MMC_LUA_MMADDRESOURCEDIR)) {
        // TODO do we need to make an OS-dependent path here?
        auto p = luaL_checkstring(L, 1);
        this->conf->AddResourceDirectory(p);
    }
    return 0;
}


int megamol::core::LuaState::PluginLoaderInfo(lua_State* L) {
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
            theLua.ThrowError("the third parameter of mmPluginLoaderInfo must be 'include' or 'exclude'.");
        }
        this->conf->AddPluginLoadInfo(vislib::TString(p), vislib::TString(f), inc);
    }
    return 0;
}


int megamol::core::LuaState::SetLogFile(lua_State* L) {
    if (this->checkConfiguring(MMC_LUA_MMSETLOGFILE)) {
        // TODO do we need to make an OS-dependent path here?
        auto p = luaL_checkstring(L, 1);
        if (!megamol::core::utility::Configuration::logFilenameLocked) {
            megamol::core::utility::log::Log::DefaultLog.SetLogFileName(vislib::sys::Path::Resolve(p), false);
        }
    }
    return 0;
}


int megamol::core::LuaState::SetLogLevel(lua_State* L) {
    if (this->checkConfiguring(MMC_LUA_MMSETLOGLEVEL)) {
        auto l = luaL_checkstring(L, 1);
        if (!megamol::core::utility::Configuration::logLevelLocked) {
            megamol::core::utility::log::Log::DefaultLog.SetLevel(parseLevelAttribute(l));
        }
    }
    return 0;
}


int megamol::core::LuaState::SetEchoLevel(lua_State* L) {
    if (this->checkConfiguring(MMC_LUA_MMSETECHOLEVEL)) {
        auto l = luaL_checkstring(L, 1);
        if (!megamol::core::utility::Configuration::logEchoLevelLocked) {
            megamol::core::utility::log::Log::DefaultLog.SetEchoLevel(parseLevelAttribute(l));
        }
    }
    return 0;
}


int megamol::core::LuaState::SetConfigValue(lua_State* L) {
    if (this->checkConfiguring(MMC_LUA_MMSETCONFIGVALUE)) {
        auto name = luaL_checkstring(L, 1);
        auto value = luaL_checkstring(L, 2);
        this->conf->setConfigValue(name, value);
    }
    return 0;
}


int megamol::core::LuaState::GetConfigValue(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMGETCONFIGVALUE)) {
        std::stringstream out;
        auto name = luaL_checkstring(L, 1);
        mmcValueType t;
        const void* val = this->coreInst->Configuration().GetValue(MMC_CFGID_VARIABLE, name, &t);
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


int megamol::core::LuaState::GetEnvValue(lua_State* L) {
    auto name = luaL_checkstring(L, 1);
    if (vislib::sys::Environment::IsSet(name)) {
        lua_pushstring(L, vislib::sys::Environment::GetVariable(name));
        return 1;
    } else {
        //lua_pushstring(L, "undef");
        lua_pushnil(L);
        return 1;
    }
}


UINT megamol::core::LuaState::parseLevelAttribute(const std::string attr) {
    UINT retval = megamol::core::utility::log::Log::LEVEL_ERROR;
    if (iequals(attr, "error")) {
        retval = megamol::core::utility::log::Log::LEVEL_ERROR;
    } else if (iequals(attr, "warn")) {
        retval = megamol::core::utility::log::Log::LEVEL_WARN;
    } else if (iequals(attr, "warning")) {
        retval = megamol::core::utility::log::Log::LEVEL_WARN;
    } else if (iequals(attr, "info")) {
        retval = megamol::core::utility::log::Log::LEVEL_INFO;
    } else if (iequals(attr, "none")) {
        retval = megamol::core::utility::log::Log::LEVEL_NONE;
    } else if (iequals(attr, "null")) {
        retval = megamol::core::utility::log::Log::LEVEL_NONE;
    } else if (iequals(attr, "zero")) {
        retval = megamol::core::utility::log::Log::LEVEL_NONE;
    } else if (iequals(attr, "all")) {
        retval = megamol::core::utility::log::Log::LEVEL_ALL;
    } else if (iequals(attr, "*")) {
        retval = megamol::core::utility::log::Log::LEVEL_ALL;
    } else {
        try {
            retval = std::stoi(attr);
        } catch (...) {
            retval = megamol::core::utility::log::Log::LEVEL_ERROR;
        }
    }
    return retval;
}


int megamol::core::LuaState::GetProcessID(lua_State* L) {
    //    if (this->checkRunning("mmGetProcessID")) {
    vislib::StringA str;
#ifdef _WIN32
    unsigned int id = GetCurrentProcessId();
#else /* _WIN32 */
    unsigned int id = getpid();
#endif /* _WIN32 */
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


        auto ret =
            this->coreInst->EnumerateParameterSlotsNoLock<megamol::core::Module>(moduleName, [L](param::ParamSlot& ps) {
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

                            auto const pspdef = psp->Definition();
                            // not nice, but we make HEX (base64 would be better, but I don't care)
                            std::string answer2(pspdef.size() * 2, ' ');
                            for (SIZE_T i = 0; i < pspdef.size(); ++i) {
                                uint8_t b = pspdef[i];
                                uint8_t bh[2] = {static_cast<uint8_t>(b / 16), static_cast<uint8_t>(b % 16)};
                                for (unsigned int j = 0; j < 2; ++j)
                                    answer2[i * 2 + j] = (bh[j] < 10u) ? ('0' + bh[j]) : ('A' + (bh[j] - 10u));
                            }
                            answer << answer2 << "\1";

                            vislib::StringA valUTF8;
                            vislib::UTF8Encoder::Encode(valUTF8, psp->ValueString().c_str());

                            answer << valUTF8 << "\1";
                        }
                    }
                    lua_pushstring(L, answer.str().c_str());
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "LuaState: ParamSlot %s has a parent which is not a Module!", ps.FullName().PeekBuffer());
                }
            });
        return ret ? 1 : 0;
    }
    return 0;
}


bool megamol::core::LuaState::getParamSlot(
    const std::string routine, const char* paramName, core::param::ParamSlot** out) {

    AbstractNamedObjectContainer::ptr_type root =
        std::dynamic_pointer_cast<AbstractNamedObjectContainer>(this->coreInst->namespaceRoot);
    if (!root) {
        std::string err = routine + ": no root";
        theLua.ThrowError(err);
        return false;
    }
    AbstractNamedObject::ptr_type obj = root.get()->FindNamedObject(paramName);
    if (!obj) {
        std::string err = routine + ": parameter \"" + paramName + "\" not found";
        theLua.ThrowError(err);
        return false;
    }
    *out = dynamic_cast<core::param::ParamSlot*>(obj.get());
    if (*out == nullptr) {
        std::string err = routine + ": parameter name \"" + paramName + "\" did not refer to a ParamSlot";
        theLua.ThrowError(err);
        return false;
    }
    return true;
}


bool megamol::core::LuaState::getView(const std::string routine, const char* viewName, core::ViewInstance** out) {

    // AbstractNamedObjectContainer::ptr_type anoc =
    // AbstractNamedObjectContainer::dynamic_pointer_cast(this->RootModule()); AbstractNamedObject::ptr_type ano =
    // anoc->FindChild(mvn); ViewInstance *vi = dynamic_cast<ViewInstance *>(ano.get());

    AbstractNamedObjectContainer::ptr_type root =
        AbstractNamedObjectContainer::dynamic_pointer_cast(this->coreInst->namespaceRoot);
    if (!root) {
        std::string err = routine + ": no root";
        theLua.ThrowError(err);
        return false;
    }
    AbstractNamedObject::ptr_type obj = root.get()->FindNamedObject(viewName);
    if (!obj) {
        std::string err = routine + ": view \"" + std::string(viewName) + "\" not found";
        theLua.ThrowError(err);
        return false;
    }
    *out = dynamic_cast<ViewInstance*>(obj.get());
    return true;
}


bool megamol::core::LuaState::getJob(const std::string routine, const char* jobName, core::JobInstance** out) {
    AbstractNamedObjectContainer::ptr_type root =
        AbstractNamedObjectContainer::dynamic_pointer_cast(this->coreInst->namespaceRoot);
    if (!root) {
        std::string err = routine + ": no root";
        theLua.ThrowError(err);
        return false;
    }
    AbstractNamedObject::ptr_type obj = root.get()->FindNamedObject(jobName);
    if (!obj) {
        std::string err = routine + ": job \"" + std::string(jobName) + "\" not found";
        theLua.ThrowError(err);
        return false;
    }
    *out = dynamic_cast<JobInstance*>(obj.get());
    return true;
}


int megamol::core::LuaState::GetParamType(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMGETPARAMTYPE)) {
        auto paramName = luaL_checkstring(L, 1);

        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());

        core::param::ParamSlot* ps = nullptr;
        if (getParamSlot(MMC_LUA_MMGETPARAMTYPE, paramName, &ps)) {

            auto psp = ps->Parameter();
            if (psp.IsNull()) {
                lua_pushstring(L, MMC_LUA_MMGETPARAMTYPE ": ParamSlot does seem to hold no parameter");
                lua_error(L);
                return 0;
            }

            auto const pspdef = psp->Definition();
            // not nice, but we make HEX (base64 would be better, but I don't care)
            std::string answer(pspdef.size() * 2, ' ');
            for (SIZE_T i = 0; i < pspdef.size(); ++i) {
                uint8_t b = pspdef[i];
                uint8_t bh[2] = {static_cast<uint8_t>(b / 16), static_cast<uint8_t>(b % 16)};
                for (unsigned int j = 0; j < 2; ++j)
                    answer[i * 2 + j] = (bh[j] < 10u) ? ('0' + bh[j]) : ('A' + (bh[j] - 10u));
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


int megamol::core::LuaState::GetParamDescription(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMGETPARAMDESCRIPTION)) {
        auto paramName = luaL_checkstring(L, 1);

        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());

        core::param::ParamSlot* ps = nullptr;
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


int megamol::core::LuaState::GetParamValue(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMGETPARAMVALUE)) {
        auto paramName = luaL_checkstring(L, 1);

        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());
        core::param::ParamSlot* ps = nullptr;
        if (getParamSlot(MMC_LUA_MMGETPARAMVALUE, paramName, &ps)) {

            auto psp = ps->Parameter();
            if (psp.IsNull()) {
                lua_pushstring(L, MMC_LUA_MMGETPARAMVALUE ": ParamSlot does seem to hold no parameter");
                lua_error(L);
                return 0;
            }

            vislib::StringA valUTF8;
            vislib::UTF8Encoder::Encode(valUTF8, psp->ValueString().c_str());

            lua_pushstring(L, valUTF8);
            return 1;
        } else {
            // the error is already thrown
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::SetParamValue(lua_State* L) {

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


int megamol::core::LuaState::CreateParamGroup(lua_State* L) {
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


int megamol::core::LuaState::CreateModule(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMCREATEMODULE)) {
        // auto viewjobName = luaL_checkstring(L, 1);
        auto className = luaL_checkstring(L, 1);
        std::string instanceName(luaL_checkstring(L, 2));

        if (instanceName.compare(0, 2, "::") != 0) {
            std::string out = "instance name \"" + instanceName + "\" must be global (starting with \"::\")";
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


int megamol::core::LuaState::DeleteModule(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMDELETEMODULE)) {
        auto moduleName = luaL_checkstring(L, 1);

        if (!this->coreInst->RequestModuleDeletion(moduleName)) {
            lua_pushstring(L, ("cannot delete module \"" + std::string(moduleName) + "\" (check MegaMol log)").c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::CreateCall(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMCREATECALL)) {
        auto className = luaL_checkstring(L, 1);
        auto from = luaL_checkstring(L, 2);
        auto to = luaL_checkstring(L, 3);

        if (!this->coreInst->RequestCallInstantiation(className, from, to)) {
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
        auto moduleName = chainStart.substr(0, pos - 1);
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


int megamol::core::LuaState::DeleteCall(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMDELETECALL)) {
        auto from = luaL_checkstring(L, 1);
        auto to = luaL_checkstring(L, 2);

        if (!this->coreInst->RequestCallDeletion(from, to)) {
            lua_pushstring(L, ("cannot delete call from \"" + std::string(from) + "\" to \"" + std::string(to) +
                                  "\" (check MegaMol log)")
                                  .c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::CreateJob(lua_State* L) {
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
        } catch (std::invalid_argument&) {
            lua_pushstring(L, ("job \"" + std::string(jobName) + "\" already exists.").c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::DeleteJob(lua_State* L) {
    lua_pushstring(L, "not implemented yet!");
    lua_error(L);
    return 0;
}


int megamol::core::LuaState::CreateView(lua_State* L) {
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
        } catch (std::invalid_argument&) {
            lua_pushstring(L, ("view \"" + std::string(viewName) + "\" already exists.").c_str());
            lua_error(L);
            return 0;
        }
    }
    return 0;
}


int megamol::core::LuaState::DeleteView(lua_State* L) {
    lua_pushstring(L, "not implemented yet!");
    lua_error(L);
    return 0;
}


int megamol::core::LuaState::QueryModuleGraph(lua_State* L) {
    if (this->checkRunning(MMC_LUA_MMQUERYMODULEGRAPH)) {
        // TODO I am not sure whether reading information from the MegaMol Graph is safe without locking
        vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());

        AbstractNamedObject::const_ptr_type ano = this->coreInst->ModuleGraphRoot();
        AbstractNamedObjectContainer::const_ptr_type anoc =
            std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
        if (!anoc) {
            lua_pushstring(L, MMC_LUA_MMQUERYMODULEGRAPH ": no root");
            lua_error(L);
            return 0;
        }

        std::stringstream answer;

        // queryModules(answer, anoc);
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
                const char* cn = nullptr;
                if (m != nullptr) {
                    cn = m->ClassName();
                }
                answer << "Class:     " << ((cn != nullptr) ? cn : "unknown") << std::endl;
                answer << "Children:  ";
                auto it_end = anoc->ChildList_End();
                int numChildren = 0;
                for (auto it = anoc->ChildList_Begin(); it != it_end; ++it) {
                    AbstractNamedObject::const_ptr_type ano = *it;
                    AbstractNamedObjectContainer::const_ptr_type anoc =
                        std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
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
                    AbstractNamedObjectContainer::const_ptr_type anoc =
                        std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
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
            for (AbstractNamedObjectContainer::child_list_type::const_iterator si = mod->ChildList_Begin(); si != se;
                 ++si) {
                const auto slot = dynamic_cast<CallerSlot*>((*si).get());
                if (slot) {
                    const Call* c = const_cast<CallerSlot*>(slot)->CallAs<Call>();
                    if (c != nullptr) {
                        answer << c->ClassName() << ";" << c->PeekCallerSlot()->Parent()->Name() << ","
                               << c->PeekCalleeSlot()->Parent()->Name() << ";" << c->PeekCallerSlot()->Name() << ","
                               << c->PeekCalleeSlot()->Name() << std::endl;
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
        // vislib::sys::AutoLock l(this->coreInst->ModuleGraphRoot()->ModuleGraphLock());
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
        AbstractNamedObjectContainer::const_ptr_type anor =
            std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
        if (!ano) {
            lua_pushstring(L, MMC_LUA_MMLISTINSTANTIATIONS ": no root");
            lua_error(L);
            return 0;
        }

        std::stringstream answer;

        if (anor) {
            const auto it_end = anor->ChildList_End();
            for (auto it = anor->ChildList_Begin(); it != it_end; ++it) {
                if (!dynamic_cast<const Module*>(it->get())) {
                    AbstractNamedObjectContainer::const_ptr_type anoc =
                        std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(*it);
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
            for (AbstractNamedObjectContainer::child_list_type::const_iterator si = mod->ChildList_Begin(); si != se;
                 ++si) {
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

int megamol::core::LuaState::Quit(lua_State* L) {
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

            // megamol::core::utility::log::Log::DefaultLog.WriteInfo(MMC_LUA_MMREADTEXTFILE ": read from file '%s':\n%s\n", filename,
            // buffer.str().c_str());

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
                    // megamol::core::utility::log::Log::DefaultLog.WriteInfo(MMC_LUA_MMREADTEXTFILE ": transformed into:\n%s\n",
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
        std::string err = MMC_LUA_MMREADTEXTFILE " requires two parameters, fileName and a function pointer";
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
