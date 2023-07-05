/*
 * LuaAPI.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#ifndef _WIN32
#include <sys/types.h>
#include <unistd.h>
#endif // _WIN32

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/LuaAPI.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/Environment.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/sys/sysfunctions.h"

#include "lua.hpp"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

//#define LUA_FULL_ENVIRONMENT

/*****************************************************************************/

// clang-format off
#define MMC_LUA_MMGETENVVALUE "mmGetEnvValue"
#define MMC_LUA_MMGETCOMPILEMODE "mmGetCompileMode"
#define MMC_LUA_MMGETBITHWIDTH "mmGetBitWidth"
#define MMC_LUA_MMGETOS "mmGetOS"
#define MMC_LUA_MMGETMACHINENAME "mmGetMachineName"
#define MMC_LUA_MMGETPROCESSID "mmGetProcessID"

#define MMC_LUA_MMPLUGINLOADERINFO "mmPluginLoaderInfo"

#define MMC_LUA_MMCURRENTSCRIPTPATH "mmCurrentScriptPath"

#define MMC_LUA_MMREADTEXTFILE "mmReadTextFile"
#define MMC_LUA_MMWRITETEXTFILE "mmWriteTextFile"
#define MMC_LUA_MMINVOKE "mmInvoke"

void megamol::core::LuaAPI::commonInit() {
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetEnvValue>(MMC_LUA_MMGETENVVALUE, "(string name)\n\tReturn the value of env variable <name>.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetCompileMode>(MMC_LUA_MMGETCOMPILEMODE, "()\n\tReturns the compilation mode ('debug' or 'release').");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetOS>(MMC_LUA_MMGETOS, "()\n\tReturns the operating system ('windows', 'linux', or 'unknown').");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetBitWidth>(MMC_LUA_MMGETBITHWIDTH, "()\n\tReturns the bit width of the compiled executable.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetMachineName>(MMC_LUA_MMGETMACHINENAME, "()\n\tReturns the machine name.");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::GetProcessID>(MMC_LUA_MMGETPROCESSID, "()\n\tReturns the process id of the running MegaMol.");

    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::ReadTextFile>(MMC_LUA_MMREADTEXTFILE, "(string fileName, function func)\n\tReturn the file contents after processing it with func(content).");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::WriteTextFile>(MMC_LUA_MMWRITETEXTFILE, "(string fileName, string content)\n\tWrite content to file. You CANNOT overwrite existing files!");
    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::Invoke>(MMC_LUA_MMINVOKE, "(string command)\n\tInvoke an abstracted input command like 'move_left'.");

    luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::CurrentScriptPath>(MMC_LUA_MMCURRENTSCRIPTPATH, "()\n\tReturns the path of the currently running script, if possible. Empty string otherwise.");
}



// we need the static variable to expose the interpreter to the register mechanism of arbitrary lambdas as lua callbacks
static megamol::core::LuaInterpreter<megamol::core::LuaAPI>* luaApiInterpreter_ptr = nullptr;
/*
 * megamol::core::LuaAPI::LuaAPI
 */
megamol::core::LuaAPI::LuaAPI() : luaApiInterpreter_(this) {
    this->commonInit();
    luaApiInterpreter_ptr = &luaApiInterpreter_;
}


/*
 * megamol::core::LuaAPI::~LuaAPI
 */
megamol::core::LuaAPI::~LuaAPI() {
    luaApiInterpreter_ptr = nullptr;
}


bool megamol::core::LuaAPI::StateOk() { return true; }


std::string megamol::core::LuaAPI::GetScriptPath() { return this->currentScriptPath; }

void megamol::core::LuaAPI::SetScriptPath(std::string const& scriptPath) { this->currentScriptPath = scriptPath; }


static auto const tc_lua_cmd = 0x65B5E4;


bool megamol::core::LuaAPI::RunFile(const std::string& envName, const std::string& fileName, std::string& result) {
    #ifdef MEGAMOL_USE_TRACY
    ZoneScopedNC("LuaAPI::RunFile", tc_lua_cmd);
    #endif
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
    #ifdef MEGAMOL_USE_TRACY
    ZoneScopedNC("LuaAPI::RunFile", tc_lua_cmd);
    #endif
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
    #ifdef MEGAMOL_USE_TRACY
    ZoneScopedNC("LuaAPI::RunString", tc_lua_cmd);
    #endif
    // todo: locking!!!
    // no two threads can touch L at the same time
    std::lock_guard<std::mutex> stateGuard(this->stateLock);
    if (this->currentScriptPath.empty() && !scriptPath.empty()) {
        // the information got better, at least
        this->currentScriptPath = scriptPath;
    }
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

int megamol::core::LuaAPI::CurrentScriptPath(struct lua_State* L) {
    lua_pushstring(L, this->currentScriptPath.c_str());
    return 1;
}


int megamol::core::LuaAPI::Invoke(lua_State *L) {
    // TODO
    return 0;
}


void megamol::core::LuaAPI::AddCallbacks(megamol::frontend_resources::LuaCallbacksCollection const& callbacks) {
    verbatim_lambda_callbacks_.push_back(callbacks);

    for (auto& c : callbacks.callbacks) {
        assert(!std::get<0>(c).empty());
    }

    register_callbacks(verbatim_lambda_callbacks_.back());
}

static std::vector<std::string> getNames(megamol::frontend_resources::LuaCallbacksCollection const& callbacks) {
    std::vector<std::string> names{};
    names.reserve(callbacks.callbacks.size());

    for (auto& c : callbacks.callbacks) {
        names.push_back(std::get<0>(c));
    }

    return names;
}

void megamol::core::LuaAPI::RemoveCallbacks(megamol::frontend_resources::LuaCallbacksCollection const& callbacks, bool delete_verbatim) {
    RemoveCallbacks( getNames(callbacks) );

    if(delete_verbatim)
    verbatim_lambda_callbacks_.remove_if([&](auto& item){
        return item.callbacks.empty() ||
            (item.callbacks.size() == callbacks.callbacks.size()
                && std::get<0>(item.callbacks.front()) == std::get<0>(callbacks.callbacks.front()));
        });
}

void megamol::core::LuaAPI::RemoveCallbacks(std::vector<std::string> const& callback_names) {
    for (auto& name : callback_names) {
        luaApiInterpreter_.UnregisterCallback(name);
        wrapped_lambda_callbacks_.remove_if([&](auto& item){ return std::get<0>(item) == name; });
    }
}

void megamol::core::LuaAPI::ClearCallbacks() {
    for (auto& callbacks: verbatim_lambda_callbacks_) {
        RemoveCallbacks( getNames(callbacks) );
    }
    assert(wrapped_lambda_callbacks_.empty());

    verbatim_lambda_callbacks_.clear();
}

void megamol::core::LuaAPI::register_callbacks(megamol::frontend_resources::LuaCallbacksCollection& callbacks) {
    if (!callbacks.is_registered) {
        for (auto& c : callbacks.callbacks) {
            auto& name = std::get<0>(c);
            auto& description = std::get<1>(c);
            auto& func = std::get<2>(c);

            if (auto found_it = std::find_if(wrapped_lambda_callbacks_.begin(), wrapped_lambda_callbacks_.end(), [&](auto const& elem) { return std::get<0>(elem) == name; });
                found_it != wrapped_lambda_callbacks_.end()) {

                RemoveCallbacks({std::get<0>(*found_it)});
            }

            wrapped_lambda_callbacks_.push_back({
                name,
                std::function<int(lua_State*)>{
                    [=](lua_State *L) -> int { return func({L}); }
            }});
            luaApiInterpreter_.RegisterCallback(name, description, std::get<1>(wrapped_lambda_callbacks_.back()));
        }
        callbacks.is_registered = true;
    }

}

void megamol::frontend_resources::LuaCallbacksCollection::LuaState::error(std::string reason) {
    luaApiInterpreter_ptr->ThrowError(reason);
}

#define lua_state \
    reinterpret_cast<lua_State*>(this->state_ptr)

int megamol::frontend_resources::LuaCallbacksCollection::LuaState::stack_size() {
    return lua_gettop(lua_state);
}

template <>
typename std::remove_reference<bool>::type
megamol::frontend_resources::LuaCallbacksCollection::LuaState::read<typename std::remove_reference<bool>::type>(size_t index) {
        bool b = lua_toboolean(lua_state, index);
        return b;
}
template <>
void megamol::frontend_resources::LuaCallbacksCollection::LuaState::write(bool item) {
    lua_pushboolean(lua_state, static_cast<int>(item));
}

template <>
typename std::remove_reference<int>::type
megamol::frontend_resources::LuaCallbacksCollection::LuaState::read<typename std::remove_reference<int>::type>(size_t index) {
    int i = luaL_checkinteger(lua_state, index);
    return i;
}
template <>
void megamol::frontend_resources::LuaCallbacksCollection::LuaState::write(int item) {
    lua_pushinteger(lua_state, item);
}

template <>
typename std::remove_reference<long>::type
megamol::frontend_resources::LuaCallbacksCollection::LuaState::read<typename std::remove_reference<long>::type>(size_t index) {
    long l = luaL_checkinteger(lua_state, index);
    return l;
}
template <>
void megamol::frontend_resources::LuaCallbacksCollection::LuaState::write(long item) {
    lua_pushinteger(lua_state, item);
}

template <>
typename std::remove_reference<float>::type
megamol::frontend_resources::LuaCallbacksCollection::LuaState::read<typename std::remove_reference<float>::type>(size_t index) {
    float f = luaL_checknumber(lua_state, index);
    return f;
}
template <>
void megamol::frontend_resources::LuaCallbacksCollection::LuaState::write(float item) {
    lua_pushnumber(lua_state, item);
}

template <>
typename std::remove_reference<double>::type
megamol::frontend_resources::LuaCallbacksCollection::LuaState::read<typename std::remove_reference<double>::type>(size_t index) {
    double d = luaL_checknumber(lua_state, index);
    return d;
}
template <>
void megamol::frontend_resources::LuaCallbacksCollection::LuaState::write(double item) {
    lua_pushnumber(lua_state, item);
}

template <>
typename std::remove_reference<std::string>::type
megamol::frontend_resources::LuaCallbacksCollection::LuaState::read<typename std::remove_reference<std::string>::type>(size_t index) {
    std::string s = luaL_checkstring(lua_state, index);
    return s;
}
template <>
void megamol::frontend_resources::LuaCallbacksCollection::LuaState::write(std::string item) {
    lua_pushstring(lua_state, item.c_str());
}

