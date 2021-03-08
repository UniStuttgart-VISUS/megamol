/*
 * LuaAPI.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_LUAAPI_H_INCLUDED
#define MEGAMOLCORE_LUAAPI_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <mutex>
#include <string>
#include "LuaInterpreter.h"
#include "mmcore/MegaMolGraph.h"

struct lua_State; // lua includes should stay in the core

namespace megamol {
namespace core {

/**
 * This class holds a Lua state. It can be used to interact with a MegaMol instance.
 * For sandboxing, a standard environment megamol_env is provided that only
 * allows lua/lib calls that are considered safe and additionally redirects
 * print() to the MegaMol log output. By default only loads base, coroutine,
 * string, table, math, package, and os (see LUA_FULL_ENVIRONMENT define).
 * Lua constants LOGINFO, LOGWARNING, LOGERROR are provided for MegaMol log output.
 */
class MEGAMOLCORE_API LuaAPI {
public:
    static const std::string MEGAMOL_ENV;

    typedef struct {
        std::function<bool(std::string const&, std::string const&)> mmSetCliOption_callback_;
        std::function<bool(std::string const&, std::string const&)> mmSetGlobalValue_callback_;
        std::function<bool(std::string const&)> mmSetAppDir_callback_;
        std::function<bool(std::string const&)> mmAddResourceDir_callback_;
        std::function<bool(std::string const&)> mmAddShaderDir_callback_;
        std::function<bool(std::string const&)> mmSetLogFile_callback_;
        std::function<bool(std::string const&)> mmSetLogLevel_callback_;
        std::function<bool(std::string const&)> mmSetEchoLevel_callback_;
        std::function<bool(std::string const&)> mmLoadProject_callback_;
    } LuaConfigCallbacks;

    typedef struct {
        std::function<std::vector<std::string>()> mmListResources_callback_; // returns list of resources available in frontend
        std::function<void(std::string const&)> mmScreenshot_callback_;
        std::function<float()> mmLastFrameTime_callback_;
        std::function<void(const unsigned int, const unsigned int)> mmSetFramebufferSize_callback_;
        std::function<void(const unsigned int, const unsigned int)> mmSetWindowPosition_callback_;
        std::function<void(const bool)> mmSetFullscreen_callback_;
        std::function<void(const bool)> mmSetVsync_callback_;
        std::function<void(const std::string)> mmSetGUIState_callback_;
        std::function<void(const bool)> mmShowGUI_callback_;
        std::function<void(const float)> mmScaleGUI_callback_;
    } LuaCallbacks;

    /**
     * @param imperativeOnly choose whether only reply-less commands will be made available
     * to avoid having round-trips across frames/threads etc. Basically config/project scripts
     * are reply-less and the LuaHost can get replies.
     */
    LuaAPI(bool imperativeOnly);

    ~LuaAPI();

    // TODO forbid copy-contructor? assignment?

    bool FillConfigFromString(const std::string& script, std::string& result, LuaConfigCallbacks const& config);

    /**
     * Run a script file, sandboxed in the environment provided.
     */
    bool RunFile(const std::string& envName, const std::string& fileName, std::string& result);
    /**
     * Run a script file, sandboxed in the environment provided.
     */
    bool RunFile(const std::string& envName, const std::wstring& fileName, std::string& result);
    /**
     * Run a script string, sandboxed in the environment provided.
     */
    bool RunString(
        const std::string& envName, const std::string& script, std::string& result, std::string scriptPath = "");

    /**
     * Run a script file, sandboxed in the standard megamol_env.
     */
    bool RunFile(const std::string& fileName, std::string& result);
    /**
     * Run a script file, sandboxed in the standard megamol_env.
     */
    bool RunFile(const std::wstring& fileName, std::string& result);
    /**
     * Run a script string, sandboxed in the standard megamol_env.
     */
    bool RunString(const std::string& script, std::string& result, std::string scriptPath = "");

    /**
     * Answer whether the wrapped lua state is valid
     */
    bool StateOk();

    /**
     * Answers the current project file path
     */
    std::string GetScriptPath(void);

    /**
     * Sets the current project file path
     */
    void SetScriptPath(std::string const& scriptPath);

    void SetMegaMolGraph(megamol::core::MegaMolGraph& graph);

    /**
     * Sets the function callback used to trigger rendering of a frame due to mmFlush.
     */
    void setFlushCallback(std::function<bool()> const& callback);

    /**
     * Sets the function callback used to trigger rendering of a frame due to mmFlush.
     * Sets the function callback used to trigger screenshots from frontbuffer into a png file.
     * Sets the function call used to retrieve the time in millis the last frame took until swapbuffers
     * Sets the function call used to resize the framebuffer/window
     * Sets the function call used to reposition the window
     * Sets the function call used to set/unset fullscreen mode
     */
    void SetCallbacks(LuaCallbacks c) { callbacks_ = c; }

    /**
     * Communicates mmQuit request to rest of MegaMol main loop.
     */
    bool getShutdown() { return shutdown_; }

    // ************************************************************
    // Lua interface routines, published to Lua as mm<name>
    // ************************************************************

protected:

    // ** MegaMol API provided for configuration / startup

    /** mmGetBithWidth get bits of executable (integer) */
    int GetBitWidth(lua_State* L);

    /** mmGetConfiguration: get compile configuration (debug, release) */
    int GetCompileMode(lua_State* L);

    /** mmGetOS: get operating system (windows, linux, unknown) */
    int GetOS(lua_State* L);

    /** mmGetMachineName: get machine name */
    int GetMachineName(lua_State* L);

    /** mmSetAppDir(string path) */
    int SetAppDir(lua_State* L);

    /**
     * mmAddShaderDir(string path): add path for searching shaders
     * and .btf files.
     */
    int AddShaderDir(lua_State* L);

    /**
     * mmAddResourceDir(string path): add path for searching generic
     * resources.
     */
    int AddResourceDir(lua_State* L);

    /** mmSetLogFile(string path): set path of the log file. */
    int SetLogFile(lua_State* L);

    /**
     * mmSetLogLevel(string level): set log level of the log file.
     * level = ('error', 'warn', 'warning', 'info', 'none', 'null',
     * 'zero', 'all', '*')
     */
    int SetLogLevel(lua_State* L);

    /** mmSetEchoLevel(string level): set level of console output, see SetLogLevel. */
    int SetEchoLevel(lua_State* L);

    // set global key-value pair in MegaMol key-value store
    int SetGlobalValue(lua_State* L);

    // set a CLI option to a specific value
    int SetCliOption(lua_State* L);

    /** mmLoadProject(string path): load project file after MegaMol started */
    int LoadProject(lua_State* L);

    /**
     * mmGetConfigValue(string name): get the value of configuration value 'name'
     */
    int GetConfigValue(lua_State* L);

    /**
     * mmGetEnvValue(string name): get the value of environment variable 'name'
     */
    int GetEnvValue(lua_State* L);

    // ** MegaMol API provided for runtime manipulation / Configurator live connection

    /** answer the ProcessID of the running MegaMol */
    int GetProcessID(lua_State* L);

    /**
     * mmGetModuleParams(string name): list all parameters of a module
     * along with their description, type and value.
     */
    int GetModuleParams(lua_State* L);

    /**
     * mmGetParamDescription(string name): get the description of a specific parameter.
     */
    int GetParamDescription(lua_State* L);

    /**
     * mmGetParamValue(string name): get the value of a specific parameter.
     */
    int GetParamValue(lua_State* L);

    /**
     * mmSetParamValue(string name, string value):
     * set the value of a specific parameter.
     */
    int SetParamValue(lua_State* L);

    int CreateParamGroup(lua_State* L);
    int SetParamGroupValue(lua_State* L);
    int ApplyParamGroupValues(lua_State* L);

    int CreateView(lua_State* L);
    int CreateModule(lua_State* L);
    int DeleteModule(lua_State* L);
    int CreateCall(lua_State* L);
    int CreateChainCall(lua_State* L);
    int DeleteCall(lua_State* L);

    int QueryModuleGraph(lua_State* L);
    int ListCalls(lua_State* L);
    int ListResources(lua_State* L);
    int ListModules(lua_State* L);
    int ListInstatiations(lua_State* L);
    int ListParameters(lua_State* L);

    int Help(lua_State* L);
    int Quit(lua_State* L);

    int ReadTextFile(lua_State* L);
    int WriteTextFile(lua_State* L);

    int Flush(lua_State* L);
    int CurrentScriptPath(lua_State* L);

    int Invoke(lua_State* L);
    int Screenshot(lua_State* L);
    int LastFrameTime(lua_State* L);
    int SetFramebufferSize(lua_State *L);
    int SetWindowPosition(lua_State *L);
    int SetFullscreen(lua_State *L);
    int SetVSync(lua_State *L);

    int SetGUIState(lua_State* L);
    int ShowGUI(lua_State* L);
    int ScaleGUI(lua_State* L);

private:

    /** all of the Lua startup code */
    void commonInit();

    /**
     * shorthand to ask graph for the param slot, returned in 'out'.
     * WARNING: assumes the graph is ALREADY LOCKED!
     */
    bool getParamSlot(const std::string routine, const char* paramName, core::param::ParamSlot** out);

    /** gets a string from the stack position i. returns false if it's not a string */
    //bool getString(int i, std::string& out);

    /** interpret string log levels */
    static UINT parseLevelAttribute(const std::string attr);

    inline static bool iequals(const std::string& one, const std::string& other) {

        return ((one.size() == other.size()) &&
                std::equal(one.begin(), one.end(), other.begin(),
                    [](const char& c1, const char& c2) { return (c1 == c2 || std::toupper(c1) == std::toupper(c2)); }));
    }

    /** the one Lua state */
    LuaInterpreter<LuaAPI> luaApiInterpreter_;

    /** the respective MegaMol graph */
    megamol::core::MegaMolGraph* graph_ptr_ = nullptr;

    LuaConfigCallbacks config_callbacks_;
    LuaCallbacks callbacks_;
    // this one is special since the frontend provides it
    std::function<bool()> mmFlush_callback_; // renders one next frame via main loop


    bool shutdown_ = false;

    /** no two threads must interfere with the reentrant L */
    std::mutex stateLock;

    std::string currentScriptPath = "";

    bool imperative_only_;
};

} /* namespace core */
} /* namespace megamol */
#endif /* MEGAMOLCORE_LUAAPI_H_INCLUDED */
