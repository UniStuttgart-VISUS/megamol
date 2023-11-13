/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <mutex>
#include <string>

//#define SOL_ALL_SAFETIES_ON 1
//#define SOL_NO_EXCEPTIONS 1
#define SOL_PRINT_ERRORS 0
#include "mmcore/MegaMolGraph.h"
#include <sol/sol.hpp>

struct lua_State; // lua includes should stay in the core

namespace megamol::core {

/**
 * This class holds a Lua state. It can be used to interact with a MegaMol instance.
 * No sandboxing is performed, the environment is complete with the exception of
 * print being redirected to the MegaMol log as information.
 * Lua constants LOGINFO, LOGWARNING, LOGERROR are provided for MegaMol log output.
 */
class LuaAPI {
public:
    LuaAPI();

    ~LuaAPI();

    // TODO forbid copy-constructor? assignment?

    /**
     * Run a script string.
     */
    sol::safe_function_result RunString(const std::string& script, std::string scriptPath = "");

    /**
     * Invoke the error-generating mechanism of lua. The current implementation throws an exception
     * that sol uses to generate into a safe_result.
     */
    static void ThrowError(const std::string& description);

    /**
     * Get Lua error if res is not valid
     */
    std::string GetError(const sol::protected_function_result& res) const;

    /**
     * Answers the current project file path
     */
    std::string GetScriptPath();

    /**
     * Sets the current project file path
     */
    void SetScriptPath(std::string const& scriptPath);

    template<typename... Args>
    void RegisterCallback(std::string const& name, std::string const& help, Args&&... args) {
        luaApiInterpreter_.set_function(name, std::forward<Args>(args)...);
        helpContainer[name] = help;
    }
    // something like this to retrofit tracy zones everywhere would be nice.
    // but I am not able to write this down.
    //template<typename Func, typename... Args>
    //void RegisterCallbackWithTracy(std::string const& name, std::string const& help, Func&& func(Args&&...args)) {
    //    luaApiInterpreter_.set_function(name, [](Func&& func, Args&&... args) {
    //        printf("Tracy happening");
    //        return func(args...);
    //    });
    //    helpContainer[name] = help;
    //}

    void UnregisterCallback(std::string const& name) {
        // TODO: are we sure this nukes the function
        luaApiInterpreter_[name].set(sol::nil);
        if (auto const it = helpContainer.find(name); it != helpContainer.end()) {
            helpContainer.erase(it);
        }
    }

    static std::string TypeToString(sol::safe_function_result& res, int index_offset = 0);

protected:
    // ** MegaMol API provided for configuration / startup

    /** mmGetBithWidth get bits of executable (integer) */
    static unsigned int GetBitWidth();

    /** mmGetConfiguration: get compile configuration (debug, release) */
    static std::string GetCompileMode();

    /** mmGetOS: get operating system (windows, linux, unknown) */
    static std::string GetOS();

    /** mmGetMachineName: get machine name */
    static std::string GetMachineName();

    /** mmGetEnvValue(string name): get the value of environment variable 'name' */
    static std::string GetEnvValue(const std::string& variable);

    /** answer the ProcessID of the running MegaMol */
    static unsigned int GetProcessID();

    /** prints out the help text */
    std::string Help() const;

    static std::string ReadTextFile(std::string filename, sol::optional<sol::function> transformation);
    static void WriteTextFile(std::string filename, std::string content);

    /** expose current script path to lua */
    std::string CurrentScriptPath();

private:
    /** all of the Lua startup code */
    void commonInit();

    /** the one Lua state */
    sol::state luaApiInterpreter_;

    std::string currentScriptPath = "";

    std::map<std::string, std::string> helpContainer;
};

} // namespace megamol::core
