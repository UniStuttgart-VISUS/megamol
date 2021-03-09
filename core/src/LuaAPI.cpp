/*
 * LuaState.cpp
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#if (_MSC_VER > 1000)
#    pragma warning(disable : 4996)
#endif /* (_MSC_VER > 1000) */
#if (_MSC_VER > 1000)
#    pragma warning(default : 4996)
#endif /* (_MSC_VER > 1000) */

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <filesystem>

#ifndef _WIN32
#    include <sys/types.h>
#    include <unistd.h>
#endif // _WIN32

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/LuaAPI.h"
#include "mmcore/utility/sys/SystemInformation.h"
#include "mmcore/view/AbstractView_EventConsumption.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/Environment.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/sysfunctions.h"

#include "lua.hpp"

//#define LUA_FULL_ENVIRONMENT

/*****************************************************************************/

// clang-format off
#define MMC_LUA_MMSETCLIOPTION "mmSetCliOption"
#define MMC_LUA_MMSETAPPDIR "mmSetAppDir"
#define MMC_LUA_MMADDRESOURCEDIR "mmAddResourceDir"
#define MMC_LUA_MMADDSHADERDIR "mmAddShaderDir"
#define MMC_LUA_MMSETLOGFILE "mmSetLogFile"
#define MMC_LUA_MMSETLOGLEVEL "mmSetLogLevel"
#define MMC_LUA_MMSETECHOLEVEL "mmSetEchoLevel"
#define MMC_LUA_MMLOADPROJECT "mmLoadProject"
#define MMC_LUA_MMSETGLOBALVALUE "mmSetGlobalValue"
#define MMC_LUA_MMGETGLOBALVALUE "mmGetGlobalValue"

#define MMC_LUA_MMGETENVVALUE "mmGetEnvValue"
#define MMC_LUA_MMGETCOMPILEMODE "mmGetCompileMode"
#define MMC_LUA_MMGETBITHWIDTH "mmGetBitWidth"
#define MMC_LUA_MMGETOS "mmGetOS"
#define MMC_LUA_MMGETMACHINENAME "mmGetMachineName"
#define MMC_LUA_MMPLUGINLOADERINFO "mmPluginLoaderInfo"
#define MMC_LUA_MMGETMODULEPARAMS "mmGetModuleParams"
#define MMC_LUA_MMGETPROCESSID "mmGetProcessID"
#define MMC_LUA_MMGETPARAMDESCRIPTION "mmGetParamDescription"
#define MMC_LUA_MMGETPARAMVALUE "mmGetParamValue"
#define MMC_LUA_MMSETPARAMVALUE "mmSetParamValue"
#define MMC_LUA_MMCREATEPARAMGROUP "mmCreateParamGroup"
#define MMC_LUA_MMSETPARAMGROUPVALUE "mmSetParamGroupValue"
#define MMC_LUA_MMAPPLYPARAMGROUPVALUES "mmApplyParamGroupValues"
#define MMC_LUA_MMCREATEVIEW "mmCreateView"
#define MMC_LUA_MMCREATEMODULE "mmCreateModule"
#define MMC_LUA_MMDELETEMODULE "mmDeleteModule"
#define MMC_LUA_MMCREATECALL "mmCreateCall"
#define MMC_LUA_MMCREATECHAINCALL "mmCreateChainCall"
#define MMC_LUA_MMDELETECALL "mmDeleteCall"
#define MMC_LUA_MMQUERYMODULEGRAPH "mmQueryModuleGraph"
#define MMC_LUA_MMLISTMODULES "mmListModules"
#define MMC_LUA_MMLISTCALLS "mmListCalls"
#define MMC_LUA_MMLISTRESOURCES "mmListResources"
#define MMC_LUA_MMLISTINSTANTIATIONS "mmListInstantiations"
#define MMC_LUA_MMQUIT "mmQuit"
#define MMC_LUA_MMREADTEXTFILE "mmReadTextFile"
#define MMC_LUA_MMWRITETEXTFILE "mmWriteTextFile"
#define MMC_LUA_MMFLUSH "mmRenderNextFrame"
#define MMC_LUA_MMCURRENTSCRIPTPATH "mmCurrentScriptPath"
#define MMC_LUA_MMLISTPARAMETERS "mmListParameters"
#define MMC_LUA_MMINVOKE "mmInvoke"
#define MMC_LUA_MMSCREENSHOT "mmScreenShot"
#define MMC_LUA_MMLASTFRAMETIME "mmLastFrameTime"
#define MMC_LUA_MMSETFRAMEBUFFERSIZE "mmSetFramebufferSize"
#define MMC_LUA_MMSETWINDOWPOSITION "mmSetWindowPosition"
#define MMC_LUA_MMSETFULLSCREEN "mmSetFullscreen"
#define MMC_LUA_MMSETVSYNC "mmSetVSync"
#define MMC_LUA_MMSETGUISTATE "mmSetGUIState"
#define MMC_LUA_MMSHOWGUI "mmShowGUI"
#define MMC_LUA_MMSCALEGUI "mmScaleGUI"

void megamol::core::LuaAPI::commonInit() {
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetEnvValue>(MMC_LUA_MMGETENVVALUE, "(string name)\n\tReturn the value of env variable <name>.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetCompileMode>(MMC_LUA_MMGETCOMPILEMODE, "()\n\tReturns the compilation mode ('debug' or 'release').");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetOS>(MMC_LUA_MMGETOS, "()\n\tReturns the operating system ('windows', 'linux', or 'unknown').");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetBitWidth>(MMC_LUA_MMGETBITHWIDTH, "()\n\tReturns the bit width of the compiled executable.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetMachineName>(MMC_LUA_MMGETMACHINENAME, "()\n\tReturns the machine name.");

    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetProcessID>(MMC_LUA_MMGETPROCESSID, "()\n\tReturns the process id of the running MegaMol.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::SetParamValue>(MMC_LUA_MMSETPARAMVALUE, "(string name, string value)\n\tSet the value of a parameter slot.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::CreateParamGroup>(MMC_LUA_MMCREATEPARAMGROUP, "(string name, string size)\n\tGenerate a param group that can only be set at once. Sets are queued until size is reached.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::SetParamGroupValue>(MMC_LUA_MMSETPARAMGROUPVALUE, "(string groupname, string paramname, string value)\n\tQueue the value of a grouped parameter.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::ApplyParamGroupValues>(MMC_LUA_MMAPPLYPARAMGROUPVALUES, "(string groupname)\n\tApply queued parameter values of group to graph.");

    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::CreateView>(MMC_LUA_MMCREATEVIEW, "(string graphName, string className, string moduleName)\n\tCreate a view module instance of class <className> called <moduleName>. The view module is registered as graph entry point. <graphName> is ignored.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::CreateModule>(MMC_LUA_MMCREATEMODULE, "(string className, string moduleName)\n\tCreate a module instance of class <className> called <moduleName>.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::DeleteModule>(MMC_LUA_MMDELETEMODULE, "(string name)\n\tDelete the module called <name>.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::CreateCall>(MMC_LUA_MMCREATECALL, "(string className, string from, string to)\n\tCreate a call of type <className>, connecting CallerSlot <from> and CalleeSlot <to>.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::CreateChainCall>(MMC_LUA_MMCREATECHAINCALL, "(string className, string chainStart, string to)\n\tAppend a call of type "
        "<className>, connection the rightmost CallerSlot starting at <chainStart> and CalleeSlot <to>.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::DeleteCall>(MMC_LUA_MMDELETECALL, "(string from, string to)\n\tDelete the call connecting CallerSlot <from> and CalleeSlot <to>.");

    // TODO: imperative?
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::Quit>(MMC_LUA_MMQUIT, "()\n\tClose the MegaMol instance.");

    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::ReadTextFile>(MMC_LUA_MMREADTEXTFILE, "(string fileName, function func)\n\tReturn the file contents after processing it with func(content).");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::WriteTextFile>(MMC_LUA_MMWRITETEXTFILE, "(string fileName, string content)\n\tWrite content to file. You CANNOT overwrite existing files!");

    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::CurrentScriptPath>(MMC_LUA_MMCURRENTSCRIPTPATH, "()\n\tReturns the path of the currently running script, if possible. Empty string otherwise.");
    // TODO: imperative?
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::Invoke>(MMC_LUA_MMINVOKE, "(string command)\n\tInvoke an abstracted input command like 'move_left'.");

    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::Screenshot>(MMC_LUA_MMSCREENSHOT, "(string filename)\n\tSave a screen shot of the GL front buffer under 'filename'.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::LastFrameTime>(MMC_LUA_MMLASTFRAMETIME, "()\n\tReturns the graph execution time of the last frame in ms.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::SetFramebufferSize>(MMC_LUA_MMSETFRAMEBUFFERSIZE, "(int width, int height)\n\tSet framebuffer dimensions to width x height.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::SetWindowPosition>(MMC_LUA_MMSETWINDOWPOSITION, "(int x, int y)\n\tSet window position to x,y.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::SetFullscreen>(MMC_LUA_MMSETFULLSCREEN, "(bool fullscreen)\n\tSet window to fullscreen (or restore).");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::SetVSync>(MMC_LUA_MMSETVSYNC, "(bool state)\n\tSet window VSync off (false) or on (true).");

    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::SetGUIState>(MMC_LUA_MMSETGUISTATE, "(string json)\n\tSet GUI state from given 'json' string.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::ShowGUI>(MMC_LUA_MMSHOWGUI, "(bool state)\n\Show (true) or hide (false) the GUI.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::ScaleGUI>(MMC_LUA_MMSCALEGUI, "(float scale)\n\Set GUI scaling factor.");

    if (!imperative_only_) {
        luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetModuleParams>(MMC_LUA_MMGETMODULEPARAMS, "(string name)\n\tReturns a 0x1-separated list of module name and all parameters."
            "\n\tFor each parameter the name, description, definition, and value are returned.");
        luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetParamDescription>(MMC_LUA_MMGETPARAMDESCRIPTION, "(string name)\n\tReturn the description of a parameter slot.");
        luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetParamValue>(MMC_LUA_MMGETPARAMVALUE, "(string name)\n\tReturn the value of a parameter slot.");
        luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::QueryModuleGraph>(MMC_LUA_MMQUERYMODULEGRAPH, "()\n\tShow the instantiated modules and their children.");
        luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::ListCalls>(MMC_LUA_MMLISTCALLS, "()\n\tReturn a list of instantiated calls.");
        luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::ListResources>(MMC_LUA_MMLISTRESOURCES, "()\n\tReturn a list of available resources in the frontend.");
        luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::ListModules>(MMC_LUA_MMLISTMODULES, "(string basemodule_or_namespace)"
            "\n\tReturn a list of instantiated modules (class id, instance id), starting from a certain module downstream or inside a namespace."
            "\n\tWill use the graph root if an empty string is passed.");
        luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::ListInstatiations>(MMC_LUA_MMLISTINSTANTIATIONS, "()\n\tReturn a list of instantiation names");
        luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::ListParameters>(MMC_LUA_MMLISTPARAMETERS, "(string baseModule_or_namespace)"
            "\n\tReturn all parameters, their type and value, starting from a certain module downstream or inside a namespace."
            "\n\tWill use the graph root if an empty string is passed.");
        luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::Flush>(MMC_LUA_MMFLUSH, "()\n\tInserts a flush event into graph manipulation queues.");
    }
}


/*
 * megamol::core::LuaAPI::LuaAPI
 */
megamol::core::LuaAPI::LuaAPI(bool imperativeOnly) : graph_ptr_(nullptr), luaApiInterpreter_(this), imperative_only_(imperativeOnly) {
    this->commonInit();
}


/*
 * megamol::core::LuaAPI::~LuaAPI
 */
megamol::core::LuaAPI::~LuaAPI() {

}


bool megamol::core::LuaAPI::StateOk() { return true; }


std::string megamol::core::LuaAPI::GetScriptPath(void) { return this->currentScriptPath; }

void megamol::core::LuaAPI::SetScriptPath(std::string const& scriptPath) { this->currentScriptPath = scriptPath; }

void megamol::core::LuaAPI::SetMegaMolGraph(megamol::core::MegaMolGraph& graph) { this->graph_ptr_ = &graph; }

bool megamol::core::LuaAPI::FillConfigFromString(const std::string& script, std::string& result, LuaConfigCallbacks const& config) {
    this->config_callbacks_ = config;

    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::SetCliOption  >(MMC_LUA_MMSETCLIOPTION, "(string name, string value)\n\tSet CLI option to a specific value.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::SetAppDir     >(MMC_LUA_MMSETAPPDIR, "(string dir)\n\tSets the path where the mmconsole.exe is located.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::AddShaderDir  >(MMC_LUA_MMADDSHADERDIR, "(string dir)\n\tAdds a shader/btf search path.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::AddResourceDir>(MMC_LUA_MMADDRESOURCEDIR, "(string dir)\n\tAdds a resource search path.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::SetLogFile    >(MMC_LUA_MMSETLOGFILE, "(string path)\n\tSets the full path of the log file.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::SetLogLevel   >(MMC_LUA_MMSETLOGLEVEL, "(int level)\n\tSets the level of log events to include. Level constants are: LOGINFO, LOGWARNING, LOGERROR.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::SetEchoLevel  >(MMC_LUA_MMSETECHOLEVEL, "(int level)\n\tSets the level of log events to output to the console (see above).");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::SetGlobalValue>(MMC_LUA_MMSETGLOBALVALUE, "(string key, string value)\n\tSets the value of name <key> to <value> in the global key-value store.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::LoadProject   >(MMC_LUA_MMLOADPROJECT, "(string path)\n\tLoad lua (project) file after MegaMol startup.");

    bool execution_result = RunString(script, result);

    luaApiInterpreter_.UnregisterCallback(MMC_LUA_MMSETCLIOPTION);
    luaApiInterpreter_.UnregisterCallback(MMC_LUA_MMSETAPPDIR);
    luaApiInterpreter_.UnregisterCallback(MMC_LUA_MMADDSHADERDIR);
    luaApiInterpreter_.UnregisterCallback(MMC_LUA_MMADDRESOURCEDIR);
    luaApiInterpreter_.UnregisterCallback(MMC_LUA_MMSETLOGFILE);
    luaApiInterpreter_.UnregisterCallback(MMC_LUA_MMSETLOGLEVEL);
    luaApiInterpreter_.UnregisterCallback(MMC_LUA_MMSETECHOLEVEL);
    luaApiInterpreter_.UnregisterCallback(MMC_LUA_MMSETGLOBALVALUE);
    luaApiInterpreter_.UnregisterCallback(MMC_LUA_MMLOADPROJECT);

    return execution_result;
}

bool megamol::core::LuaAPI::RunFile(const std::string& envName, const std::string& fileName, std::string& result) {
    std::ifstream input(fileName, std::ios::in);
    if (!input.fail()) {
        std::ostringstream buffer;
        buffer << input.rdbuf();
        return RunString(envName, buffer.str(), result, fileName);
    }
    else {
        return false;
    }
}


bool megamol::core::LuaAPI::RunFile(const std::string& envName, const std::wstring& fileName, std::string& result) {
    vislib::sys::File input;
    if (input.Open(fileName.c_str(), vislib::sys::File::AccessMode::READ_ONLY, vislib::sys::File::ShareMode::SHARE_READ,
        vislib::sys::File::CreationMode::OPEN_ONLY)) {
        vislib::StringA contents;
        vislib::sys::ReadTextFile(contents, input);
        input.Close();
        const std::string narrowFileName(fileName.begin(), fileName.end());
        return RunString(envName, std::string(contents), result, narrowFileName);
    }
    else {
        return false;
    }
}


bool megamol::core::LuaAPI::RunString(
    const std::string& envName, const std::string& script, std::string& result, std::string scriptPath) {
    // todo: locking!!!
    // no two threads can touch L at the same time
    std::lock_guard<std::mutex> stateGuard(this->stateLock);
    this->currentScriptPath = scriptPath;
    return luaApiInterpreter_.RunString(envName, script, result);
}


bool megamol::core::LuaAPI::RunFile(const std::string& fileName, std::string& result) {
    return RunFile("default_env", fileName, result);
}


bool megamol::core::LuaAPI::RunFile(const std::wstring& fileName, std::string& result) {
    return RunFile("default_env", fileName, result);
}


bool megamol::core::LuaAPI::RunString(const std::string& script, std::string& result, std::string scriptPath) {
    return RunString("default_env", script, result, scriptPath);
}


int megamol::core::LuaAPI::SetAppDir(lua_State* L) {
    auto p = luaL_checkstring(L, 1);

    if(!config_callbacks_.mmSetAppDir_callback_(p))
        luaApiInterpreter_.ThrowError("mmSetAppDir: Application directory \"" + std::string(p) + "\" does not exist.");

    return 0;
}


int megamol::core::LuaAPI::AddResourceDir(lua_State* L) {
    // TODO do we need to make an OS-dependent path here?
    auto p = luaL_checkstring(L, 1);

    if(!config_callbacks_.mmAddResourceDir_callback_(p))
        luaApiInterpreter_.ThrowError("mmAddResourceDir: Resource directory \"" + std::string(p) + "\" does not exist.");

    return 0;
}


int megamol::core::LuaAPI::AddShaderDir(lua_State* L) {
    // TODO do we need to make an OS-dependent path here?
    auto p = luaL_checkstring(L, 1);

    if(!config_callbacks_.mmAddShaderDir_callback_(p))
        luaApiInterpreter_.ThrowError("mmAddShaderDir: Shader directory \"" + std::string(p) + "\" does not exist.");

    return 0;
}


int megamol::core::LuaAPI::SetLogFile(lua_State* L) {
    auto p = luaL_checkstring(L, 1);

    config_callbacks_.mmSetLogFile_callback_(p);

    return 0;
}


int megamol::core::LuaAPI::SetLogLevel(lua_State* L) {
    auto l = luaL_checkstring(L, 1);

    unsigned int result = 0;
    try {
        if (std::string(l).find_first_of("0123456789") != std::string::npos) {
            result = std::stoi(l);

            if(result < 0)
                throw std::exception();
        }
        else {
            result = parseLevelAttribute(l);
        }

        config_callbacks_.mmSetLogLevel_callback_(std::to_string(result));
    }
    catch (...) {
        luaApiInterpreter_.ThrowError("mmSetLogLevel: Could not parse valid log level string or positive integer from argument \"" + std::string(l) + "\"");
    }

    return 0;
}


int megamol::core::LuaAPI::SetEchoLevel(lua_State* L) {
    auto l = luaL_checkstring(L, 1);

    unsigned int result = 0;
    try {
        if (std::string(l).find_first_of("0123456789") != std::string::npos) {
            result = std::stoi(l);

            if(result < 0)
                throw std::exception();
        }
        else {
            result = parseLevelAttribute(l);
        }

        config_callbacks_.mmSetEchoLevel_callback_(std::to_string(result));
    }
    catch (...) {
        luaApiInterpreter_.ThrowError("mmSetEchoLevel: Could not parse valid echo level string or positive integer from argument \"" + std::string(l) + "\"");
    }

    return 0;
}


int megamol::core::LuaAPI::SetGlobalValue(lua_State* L) {
    auto key = luaL_checkstring(L, 1);
    auto value = luaL_checkstring(L, 2);

    if(!config_callbacks_.mmSetGlobalValue_callback_(key, value))
        luaApiInterpreter_.ThrowError("mmSetGlobalValue: Global Key-value pair \n\tkey=" + std::string(key) + "\n\tvalue=" + std::string(value) + "\nnot in valid format. Can not contain space or =");

    return 0;
}


int megamol::core::LuaAPI::SetCliOption(lua_State* L) {
    auto name = luaL_checkstring(L, 1);
    auto value = luaL_checkstring(L, 2);

    if(!config_callbacks_.mmSetCliOption_callback_(name, value))
        luaApiInterpreter_.ThrowError("mmSetCliOption: CLI option \"" + std::string(name) + "\" or value \"" + std::string(value) + "\" not valid.");

    return 0;
}

int megamol::core::LuaAPI::LoadProject(lua_State* L) {
    auto f = luaL_checkstring(L, 1);

    if(!config_callbacks_.mmLoadProject_callback_(f))
        luaApiInterpreter_.ThrowError("mmLoadProject: no such file \"" + std::string(f) + "\"");

    return 0;
}


int megamol::core::LuaAPI::GetBitWidth(lua_State* L) {
    lua_pushinteger(L, vislib::sys::SystemInformation::SelfWordSize());
    return 1;
}


int megamol::core::LuaAPI::GetCompileMode(lua_State* L) {
#ifdef _DEBUG
    lua_pushstring(L, "debug");
#else
    lua_pushstring(L, "release");
#endif
    return 1;
}


int megamol::core::LuaAPI::GetOS(lua_State* L) {
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


int megamol::core::LuaAPI::GetMachineName(lua_State* L) {
    lua_pushstring(L, vislib::sys::SystemInformation::ComputerNameA());
    return 1;
}


int megamol::core::LuaAPI::GetConfigValue(lua_State* L) {
    std::ostringstream out;
    auto name = luaL_checkstring(L, 1);
    mmcValueType t;
    // TODO
    int64_t v = 0;
    const void* val = &v;
    //const void* val = this->coreInst->Configuration().GetValue(MMC_CFGID_VARIABLE, name, &t);
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


int megamol::core::LuaAPI::GetEnvValue(lua_State* L) {
    auto name = luaL_checkstring(L, 1);
    if (vislib::sys::Environment::IsSet(name)) {
        lua_pushstring(L, vislib::sys::Environment::GetVariable(name));
        return 1;
    }
    else {
        lua_pushnil(L);
        return 1;
    }
}


// TODO move helper function into Log, such that for a string we get the corresponding Log Level
// The CLI parser will benefit form this, so the user can also provide the shorthand notation via CLI
UINT megamol::core::LuaAPI::parseLevelAttribute(const std::string attr) {
    UINT retval = megamol::core::utility::log::Log::LEVEL_ERROR;
    if (iequals(attr, "error")) {
        retval = megamol::core::utility::log::Log::LEVEL_ERROR;
    }
    else if (iequals(attr, "warn")) {
        retval = megamol::core::utility::log::Log::LEVEL_WARN;
    }
    else if (iequals(attr, "warning")) {
        retval = megamol::core::utility::log::Log::LEVEL_WARN;
    }
    else if (iequals(attr, "info")) {
        retval = megamol::core::utility::log::Log::LEVEL_INFO;
    }
    else if (iequals(attr, "none")) {
        retval = megamol::core::utility::log::Log::LEVEL_NONE;
    }
    else if (iequals(attr, "null")) {
        retval = megamol::core::utility::log::Log::LEVEL_NONE;
    }
    else if (iequals(attr, "zero")) {
        retval = megamol::core::utility::log::Log::LEVEL_NONE;
    }
    else if (iequals(attr, "all")) {
        retval = megamol::core::utility::log::Log::LEVEL_ALL;
    }
    else if (iequals(attr, "*")) {
        retval = megamol::core::utility::log::Log::LEVEL_ALL;
    }
    else {
        retval = std::stoi(attr);
        // dont catch stoi exceptions
        // let exception be handled by the one who called me
    }
    return retval;
}


int megamol::core::LuaAPI::GetProcessID(lua_State* L) {
    vislib::StringA str;
#ifdef _WIN32    
    unsigned int id = GetCurrentProcessId();
#else
    unsigned int id = static_cast<unsigned int>(getpid());
#endif // _WIN32
    str.Format("%u", id);
    lua_pushstring(L, str.PeekBuffer());
    return 1;
}

#define graph_ (*graph_ptr_)

int megamol::core::LuaAPI::GetModuleParams(lua_State* L) {
    const auto *moduleName = luaL_checkstring(L, 1);

    auto slots = graph_.EnumerateModuleParameterSlots(moduleName);
    auto mod = graph_.FindModule(moduleName);
    std::ostringstream answer;
    if (mod != nullptr) {
        answer << mod->FullName() << "\1";
        for (auto &ps : slots) {
            answer << ps->Name() << "\1";
            answer << ps->Description() << "\1";
            auto par = ps->Parameter();
            if (par.IsNull()) {
                std::ostringstream err;
                err << MMC_LUA_MMGETMODULEPARAMS ": ParamSlot " << ps->FullName()
                    << " does not seem to hold a parameter!";
                luaApiInterpreter_.ThrowError(err.str());
            }
            answer << par->ValueString() << "\1";
        }
        lua_pushstring(L, answer.str().c_str());
        return 1;
    }
    else {
        answer << "Cannot find module \"" << moduleName << "\"";
        luaApiInterpreter_.ThrowError(answer.str());
        return 0;
    }
}


bool megamol::core::LuaAPI::getParamSlot(
    const std::string routine, const char* paramName, core::param::ParamSlot** out) {

    core::param::ParamSlot* slotPtr = graph_.FindParameterSlot(paramName);

    if (slotPtr == nullptr) {
        std::string err = routine + ": parameter \"" + paramName + "\" not found";
        luaApiInterpreter_.ThrowError(err);
        return false;
    }

    *out = slotPtr;

    return true;
}


int megamol::core::LuaAPI::GetParamDescription(lua_State* L) {
    auto paramName = luaL_checkstring(L, 1);

    core::param::ParamSlot* ps = nullptr;
    if (this->getParamSlot(MMC_LUA_MMGETPARAMDESCRIPTION, paramName, &ps)) {

        vislib::StringA valUTF8;
        vislib::UTF8Encoder::Encode(valUTF8, ps->Description());

        lua_pushstring(L, valUTF8);
        return 1;
    }
    // the error is already thrown
    return 0;
}


int megamol::core::LuaAPI::GetParamValue(lua_State* L) {
    const auto *paramName = luaL_checkstring(L, 1);

    const auto *param = graph_.FindParameter(paramName);
    if (param == nullptr) {
        std::ostringstream out;
        out << "could not find parameter \"";
        out << paramName;
        out << "\".";
        luaApiInterpreter_.ThrowError(out.str());
        return 0;
    }

    // TODO: before, that stuff was encoded to UTF8. why?
    lua_pushstring(L, param->ValueString());
    return 1;
}


int megamol::core::LuaAPI::SetParamValue(lua_State* L) {
    auto *paramName = luaL_checkstring(L, 1);
    auto *paramValue = luaL_checkstring(L, 2);

    auto *param = graph_.FindParameter(paramName);
    if (param == nullptr) {
        std::ostringstream out;
        out << "could not find parameter \"";
        out << paramName;
        out << "\".";
        luaApiInterpreter_.ThrowError(out.str());
        return 0;
    }
    if (!param->ParseValue(paramValue)) {
        std::ostringstream out;
        out << "could not set \"";
        out << paramName;
        out << "\" to \"";
        out << paramValue;
        out << "\" (check MegaMol log)";
        luaApiInterpreter_.ThrowError(out.str());
        return 0;
    }
    return 0;
}


int megamol::core::LuaAPI::CreateParamGroup(lua_State* L) {
    auto groupName = luaL_checkstring(L, 1);
    auto groupSize = luaL_checkinteger(L, 2);

    graph_.Convenience().CreateParameterGroup(groupName);
    // groupSize is ignored because why do we need it?

    return 0;
}


int megamol::core::LuaAPI::SetParamGroupValue(lua_State* L) {
    auto paramGroup = luaL_checkstring(L, 1);
    auto paramName = luaL_checkstring(L, 2);
    auto paramValue = luaL_checkstring(L, 3);

    const auto errorMsg = [&]() {
        std::ostringstream out;
        out << "could not set \"";
        out << paramName;
        out << "\" in group \"";
        out << paramGroup;
        out << "\" to \"";
        out << paramValue;
        out << "\" (check MegaMol log)";
        luaApiInterpreter_.ThrowError(out.str());
    };

    auto groupPtr = graph_.Convenience().FindParameterGroup(paramGroup);
    if (!groupPtr) {
        errorMsg();
        return 0;
    }

    bool queued = groupPtr->QueueParameterValue(paramName, paramValue);
    if (!queued) {
        errorMsg();
        return 0;
    }

    return 0;
}

int megamol::core::LuaAPI::ApplyParamGroupValues(lua_State* L) {
    auto paramGroup = luaL_checkstring(L, 1);

    const auto errorMsg = [&](std::string detail) {
        std::ostringstream out;
        out << "could not apply group\"";
        out << paramGroup;
        out << "\" values to graph ("<< detail <<")";
        luaApiInterpreter_.ThrowError(out.str());
    };

    auto groupPtr = graph_.Convenience().FindParameterGroup(paramGroup);
    if (!groupPtr) {
        errorMsg("no such group");
        return 0;
    }

    bool queued = groupPtr->ApplyQueuedParameterValues();
    if (!queued) {
        errorMsg("some parameter values did not parse");
        return 0;
    }

    return 0;
}


int megamol::core::LuaAPI::CreateView(lua_State* L) {
    const std::string baseName(luaL_checkstring(L, 1));
    const std::string className(luaL_checkstring(L, 2));
    const std::string instanceName(luaL_checkstring(L, 3));

    const auto errorMsg = [&](std::string detail) {
        std::ostringstream out;
        out << "could not create \"";
        out << className;
        out << "\" module ("+ detail +")";
        luaApiInterpreter_.ThrowError(out.str());
    };

    if (!graph_.CreateModule(className, instanceName)) {
        errorMsg("could not create module");
        return 0;
    }

    if (!graph_.SetGraphEntryPoint(
        instanceName,
        megamol::core::view::get_gl_view_runtime_resources_requests(),
        megamol::core::view::view_rendering_execution,
        megamol::core::view::view_init_rendering_state))
    {
        errorMsg("could not set graph entry point");
    }

    return 0;
}

int megamol::core::LuaAPI::CreateModule(lua_State* L) {
    const auto *className = luaL_checkstring(L, 1);
    const std::string instanceName(luaL_checkstring(L, 2));

    //if (instanceName.compare(0, 2, "::") != 0) {
    //    std::string out = "instance name \"" + instanceName + R"(" must be global (starting with "::"))";
    //    luaApiInterpreter_.ThrowError(out);
    //    return 0;
    //}
    if (!graph_.CreateModule(className, instanceName)) {
        std::ostringstream out;
        out << "could not create \"";
        out << className;
        out << "\" module (check MegaMol log)";
        luaApiInterpreter_.ThrowError(out.str());
    }
    return 0;
}


int megamol::core::LuaAPI::DeleteModule(lua_State* L) {
    const auto *moduleName = luaL_checkstring(L, 1);

    if (!graph_.DeleteModule(moduleName)) {
        luaApiInterpreter_.ThrowError("cannot delete module \"" + std::string(moduleName) + "\" (check MegaMol log)");
    }
    return 0;
}


int megamol::core::LuaAPI::CreateCall(lua_State* L) {
    const auto *className = luaL_checkstring(L, 1);
    const auto *from = luaL_checkstring(L, 2);
    const auto *to = luaL_checkstring(L, 3);

    if (!graph_.CreateCall(className, from, to)) {
        std::ostringstream out;
        out << "could not create \"";
        out << className;
        out << "\" call (check MegaMol log)";
        luaApiInterpreter_.ThrowError(out.str());
    }
    return 0;
}

int megamol::core::LuaAPI::CreateChainCall(lua_State* L) {
    auto className = luaL_checkstring(L, 1);
    std::string chainStart = luaL_checkstring(L, 2);
    std::string to = luaL_checkstring(L, 3);

    return graph_.Convenience().CreateChainCall(className, chainStart, to);
}


int megamol::core::LuaAPI::DeleteCall(lua_State* L) {
    const auto *from = luaL_checkstring(L, 1);
    const auto *to = luaL_checkstring(L, 2);

    if (!graph_.DeleteCall(from, to)) {
        luaApiInterpreter_.ThrowError("cannot delete call from \"" + std::string(from) + "\" to \"" + std::string(to) +
            "\" (check MegaMol log)");
    }
    return 0;
}


int megamol::core::LuaAPI::QueryModuleGraph(lua_State* L) {

    std::ostringstream answer;

    // TODO

    // queryModules(answer, anoc);
    //std::vector<AbstractNamedObjectContainer::const_ptr_type> anoStack;
    //anoStack.push_back(anoc);
    //while (!anoStack.empty()) {
    //    anoc = anoStack.back();
    //    anoStack.pop_back();

    //    if (anoc) {
    //        const auto m = Module::dynamic_pointer_cast(anoc);
    //        answer << (m != nullptr ? "Module:    " : "Container: ") << anoc.get()->FullName() << std::endl;
    //        if (anoc.get()->Parent() != nullptr) {
    //            answer << "Parent:    " << anoc.get()->Parent()->FullName() << std::endl;
    //        } else {
    //            answer << "Parent:    none" << std::endl;
    //        }
    //        const char* cn = nullptr;
    //        if (m != nullptr) {
    //            cn = m->ClassName();
    //        }
    //        answer << "Class:     " << ((cn != nullptr) ? cn : "unknown") << std::endl;
    //        answer << "Children:  ";
    //        auto it_end = anoc->ChildList_End();
    //        int numChildren = 0;
    //        for (auto it = anoc->ChildList_Begin(); it != it_end; ++it) {
    //            AbstractNamedObject::const_ptr_type ano = *it;
    //            AbstractNamedObjectContainer::const_ptr_type anoc =
    //                std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
    //            if (anoc) {
    //                if (numChildren == 0) {
    //                    answer << std::endl;
    //                }
    //                answer << anoc.get()->FullName() << std::endl;
    //                numChildren++;
    //            }
    //        }
    //        for (auto it = anoc->ChildList_Begin(); it != it_end; ++it) {
    //            AbstractNamedObject::const_ptr_type ano = *it;
    //            AbstractNamedObjectContainer::const_ptr_type anoc =
    //                std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
    //            if (anoc) {
    //                anoStack.push_back(anoc);
    //            }
    //        }
    //        if (numChildren == 0) {
    //            answer << "none" << std::endl;
    //        }
    //    }
    //}

    lua_pushstring(L, answer.str().c_str());
    return 1;
}

int megamol::core::LuaAPI::ListCalls(lua_State* L) {

    const int n = lua_gettop(L);
    std::ostringstream answer;
    auto& calls_list = graph_.ListCalls();
    for (auto& call: calls_list) {
        answer << call.callPtr->ClassName() << ";" << call.callPtr->PeekCallerSlot()->Parent()->Name() << ","
               << call.callPtr->PeekCalleeSlot()->Parent()->Name() << ";" << call.callPtr->PeekCallerSlot()->Name() << ","
               << call.callPtr->PeekCalleeSlot()->Name() << std::endl;
    }

    if (calls_list.empty()) {
        answer << "(none)" << std::endl;
    }

    // TODO

    //const auto fun = [&answer](Module* mod) {
    //    AbstractNamedObjectContainer::child_list_type::const_iterator se = mod->ChildList_End();
    //    for (AbstractNamedObjectContainer::child_list_type::const_iterator si = mod->ChildList_Begin(); si != se;
    //         ++si) {
    //        const auto slot = dynamic_cast<CallerSlot*>((*si).get());
    //        if (slot) {
    //            const Call* c = const_cast<CallerSlot*>(slot)->CallAs<Call>();
    //            if (c != nullptr) {
    //                answer << c->ClassName() << ";" << c->PeekCallerSlot()->Parent()->Name() << ","
    //                       << c->PeekCalleeSlot()->Parent()->Name() << ";" << c->PeekCallerSlot()->Name() << ","
    //                       << c->PeekCalleeSlot()->Name() << std::endl;
    //            }
    //        }
    //    }
    //};

    //if (n == 1) {
    //    const auto starting_point = luaL_checkstring(L, 1);
    //    if (!std::string(starting_point).empty()) {
    //        this->coreInst->EnumModulesNoLock(starting_point, fun);
    //    } else {
    //        this->coreInst->EnumModulesNoLock(nullptr, fun);
    //    }
    //} else {
    //    this->coreInst->EnumModulesNoLock(nullptr, fun);
    //}

    lua_pushstring(L, answer.str().c_str());
    return 1;
}

int megamol::core::LuaAPI::ListResources(lua_State* L) {

    const int n = lua_gettop(L);
    std::ostringstream answer;
    auto resources_list = this->callbacks_.mmListResources_callback_();

    for (auto& resource_name: resources_list) {
        answer << resource_name << std::endl;
    }

    if (resources_list.empty()) {
        answer << "(none)" << std::endl;
    }

    lua_pushstring(L, answer.str().c_str());
    return 1;
}

int megamol::core::LuaAPI::ListModules(lua_State* L) {
    const int n = lua_gettop(L);

    std::ostringstream answer;
    const auto *starting_point = (n == 1) ? luaL_checkstring(L, 1) : "";
    // actually putting an empty string as an argument on purpose is OK too
    ModuleList_t modules_list = std::string(starting_point).empty() ? graph_.ListModules() : graph_.Convenience().ListModules(starting_point);

    for (auto& module: modules_list) {
        answer << module.modulePtr->ClassName() << ";" << module.modulePtr->Name() << std::endl;
    }

    if (modules_list.empty()) {
        answer << "(none)" << std::endl;
    }

    lua_pushstring(L, answer.str().c_str());
    return 1;
}

int megamol::core::LuaAPI::ListInstatiations(lua_State* L) {

    std::ostringstream answer;

    // TODO

    //AbstractNamedObject::const_ptr_type ano = this->coreInst->ModuleGraphRoot();
    //AbstractNamedObjectContainer::const_ptr_type anor =
    //    std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(ano);
    //if (!ano) {
    //    luaApiInterpreter_.ThrowError(MMC_LUA_MMLISTINSTANTIATIONS ": no root");
    //    return 0;
    //}


    //if (anor) {
    //    const auto it_end = anor->ChildList_End();
    //    for (auto it = anor->ChildList_Begin(); it != it_end; ++it) {
    //        if (!dynamic_cast<const Module*>(it->get())) {
    //            AbstractNamedObjectContainer::const_ptr_type anoc =
    //                std::dynamic_pointer_cast<const AbstractNamedObjectContainer>(*it);
    //            answer << anoc->FullName() << std::endl;
    //            // TODO: the immediate child view should be it, generally
    //        }
    //    }
    //}

    lua_pushstring(L, answer.str().c_str());
    return 1;
}

int megamol::core::LuaAPI::ListParameters(lua_State* L) {

    const int n = lua_gettop(L);

    std::ostringstream answer;

    // TODO

    //const auto fun = [&answer](Module* mod) {
    //    AbstractNamedObjectContainer::child_list_type::const_iterator se = mod->ChildList_End();
    //    for (AbstractNamedObjectContainer::child_list_type::const_iterator si = mod->ChildList_Begin(); si != se;
    //         ++si) {
    //        const auto slot = dynamic_cast<param::ParamSlot*>((*si).get());
    //        if (slot) {
    //            answer << slot->FullName() << "\1" << slot->Parameter()->ValueString() << "\1";
    //        }
    //    }
    //};

    //if (n == 1) {
    //    const auto starting_point = luaL_checkstring(L, 1);
    //    if (!std::string(starting_point).empty()) {
    //        this->coreInst->EnumModulesNoLock(starting_point, fun);
    //    } else {
    //        this->coreInst->EnumModulesNoLock(nullptr, fun);
    //    }
    //} else {
    //    this->coreInst->EnumModulesNoLock(nullptr, fun);
    //}

    lua_pushstring(L, answer.str().c_str());
    return 1;
}

int megamol::core::LuaAPI::Quit(lua_State* L) {
    this->shutdown_ = true;
    return 0;
}

int megamol::core::LuaAPI::ReadTextFile(lua_State* L) {
    int n = lua_gettop(L);
    if (n == 2) {
        const auto filename = luaL_checkstring(L, 1);
        std::ifstream t(filename);
        if (t.good()) {
            std::ostringstream buffer;
            buffer << t.rdbuf();

            // megamol::core::utility::log::Log::DefaultLog.WriteInfo(MMC_LUA_MMREADTEXTFILE ": read from file '%s':\n%s\n", filename,
            // buffer.str().c_str());

            lua_remove(L, 1); // get rid of the filename on the stack, leaving the function pointer
            lua_pushstring(L, buffer.str().c_str()); // put string parameter on top of stack
            if (lua_type(L, 1) == LUA_TNIL) {
                // no transformation function, just return the string
                return 1;
            }
            else {
                // call the function pointer
                lua_pcall(L, 1, 1, 0);
                n = lua_gettop(L);
                if (n != 1) {
                    luaApiInterpreter_.ThrowError(MMC_LUA_MMREADTEXTFILE ": function did not return a string, this is bad.");
                }
                else {
                    const auto newString = luaL_checkstring(L, 1);
                    // megamol::core::utility::log::Log::DefaultLog.WriteInfo(MMC_LUA_MMREADTEXTFILE ": transformed into:\n%s\n",
                    // newString);
                    return 1;
                }
            }
        }
        else {
            std::string err = MMC_LUA_MMREADTEXTFILE ": cannot open file '";
            err += filename;
            err += "'.";
            luaApiInterpreter_.ThrowError(err);
        }
    }
    else {
        luaApiInterpreter_.ThrowError(MMC_LUA_MMREADTEXTFILE " requires two parameters, fileName and a function pointer");
    }
    return 0;
}

int megamol::core::LuaAPI::WriteTextFile(lua_State* L) {
    int n = lua_gettop(L);
    if (n == 2) {
        const auto filename = luaL_checkstring(L, 1);
        if (std::filesystem::exists(filename)) {
            std::string err = MMC_LUA_MMWRITETEXTFILE ": Overwriting existing file '";
            err += filename;
            err += "' is not allowed!";
            luaApiInterpreter_.ThrowError(err);
        } else {
            std::ofstream t(filename, std::ofstream::out);
            std::string output = luaL_checkstring(L, 2);
            lua_remove(L, 1); // get rid of the filename on the stack
            lua_remove(L, 1); // get rid of the string on the stack

            if (t.good()) {
                t.write(output.c_str(), output.length());
            } else {
                std::string err = MMC_LUA_MMWRITETEXTFILE ": cannot open file '";
                err += filename;
                err += "'.";
                luaApiInterpreter_.ThrowError(err);
            }
        }
    } else {
        luaApiInterpreter_.ThrowError(MMC_LUA_MMREADTEXTFILE " requires two parameters, fileName and a function pointer");
    }
    return 0;
}

void megamol::core::LuaAPI::setFlushCallback(std::function<bool()> const& callback) {
    mmFlush_callback_ = callback;
}

int megamol::core::LuaAPI::Flush(lua_State* L) {
    bool result = mmFlush_callback_();

    return result ? 0 : 1;
}

int megamol::core::LuaAPI::CurrentScriptPath(struct lua_State* L) {
    lua_pushstring(L, this->currentScriptPath.c_str());
    return 1;
}


int megamol::core::LuaAPI::Invoke(lua_State *L) {
    // TODO
    return 0;
}


int megamol::core::LuaAPI::Screenshot(lua_State* L) {
    int n = lua_gettop(L);
    if (n == 1) {
        const std::string filename (luaL_checkstring(L, 1));
        callbacks_.mmScreenshot_callback_(filename);
    } else {
        luaApiInterpreter_.ThrowError(MMC_LUA_MMSCREENSHOT " requires one parameter: fileName");
    }

    return 0;
}


int megamol::core::LuaAPI::LastFrameTime(lua_State *L) {
    lua_pushnumber(L, callbacks_.mmLastFrameTime_callback_());
    return 1;
}


int megamol::core::LuaAPI::SetFramebufferSize(lua_State *L) {
    int n = lua_gettop(L);
    if (n == 2) {
        unsigned int width = 0, height = 0;
        width = luaL_checkinteger(L,1);
        height = luaL_checkinteger(L, 2);

        callbacks_.mmSetFramebufferSize_callback_(width, height);
    } else {
        luaApiInterpreter_.ThrowError(MMC_LUA_MMSETFRAMEBUFFERSIZE " requires two parameters: width, height");
    }

  return 0;
}

int megamol::core::LuaAPI::SetWindowPosition(lua_State *L) {
    int n = lua_gettop(L);
    if (n == 2) {
        unsigned int x = 0, y = 0;
        x = luaL_checkinteger(L,1);
        y = luaL_checkinteger(L, 2);

        callbacks_.mmSetWindowPosition_callback_(x, y);
    } else {
        luaApiInterpreter_.ThrowError(MMC_LUA_MMSETWINDOWPOSITION " requires two parameters: x, y");
    }

  return 0;
}

int megamol::core::LuaAPI::SetFullscreen(lua_State *L) {
    int n = lua_gettop(L);
    if (n == 1) {
        bool fs = false;
        fs = lua_toboolean(L, 1);

        callbacks_.mmSetFullscreen_callback_(fs);
    } else {
        luaApiInterpreter_.ThrowError(MMC_LUA_MMSETFULLSCREEN " requires one bool parameter");
    }

  return 0;
}

int megamol::core::LuaAPI::SetVSync(lua_State *L) {
    int n = lua_gettop(L);
    if (n == 1) {
        bool fs = false;
        fs = lua_toboolean(L, 1);

        callbacks_.mmSetVsync_callback_(fs);
    } else {
        luaApiInterpreter_.ThrowError(MMC_LUA_MMSETVSYNC " requires one bool parameter");
    }

  return 0;
}

int megamol::core::LuaAPI::SetGUIState(lua_State *L) {
    int n = lua_gettop(L);
    if (n == 1) {
        std::string fs = lua_tostring(L, 1);

        callbacks_.mmSetGUIState_callback_(fs);
    } else {
        luaApiInterpreter_.ThrowError(MMC_LUA_MMSETGUISTATE " requires one string parameter");
    }

  return 0;
}

int megamol::core::LuaAPI::ShowGUI(lua_State *L) {
    int n = lua_gettop(L);
    if (n == 1) {
        bool fs = false;
        fs = lua_toboolean(L, 1);

        callbacks_.mmShowGUI_callback_(fs);
    } else {
        luaApiInterpreter_.ThrowError(MMC_LUA_MMSHOWGUI " requires one bool parameter");
    }

  return 0;
}

int megamol::core::LuaAPI::ScaleGUI(lua_State *L) {
    int n = lua_gettop(L);
    if (n == 1) {
        double scale = 1.0;
        scale = luaL_checknumber(L, 1);

        callbacks_.mmScaleGUI_callback_(static_cast<float>(scale));
    } else {
        luaApiInterpreter_.ThrowError(MMC_LUA_MMSCALEGUI " requires one float parameter");
    }

  return 0;
}
