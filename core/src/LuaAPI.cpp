/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/LuaAPI.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include <sol/stack.hpp>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "vislib/UTF8Encoder.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/Environment.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/sys/sysfunctions.h"

#ifndef _WIN32
#include <sys/types.h>
#include <unistd.h>
#endif // _WIN32

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
#define MMC_LUA_MMHELP "mmHelp"

int exceptionHandler(lua_State* L, sol::optional<const std::exception&> maybe_exception, sol::string_view description) {
    std::stringstream ss;
    ss << "LuaAPI: An exception occurred in a function. ";
    if (maybe_exception) {
        ss << "Exception: ";
        const std::exception& ex = *maybe_exception;
        ss << ex.what() << std::endl;
    } else {
        ss << "Description: ";
        std::cout.write(description.data(), static_cast<std::streamsize>(description.size()));
        ss << std::endl;
    }

    // you must push 1 element onto the stack to be
    // transported through as the error object in Lua
    // note that Lua -- and 99.5% of all Lua users and libraries -- expects a string
    // so we push a single string (in our case, the description of the error)
    megamol::core::utility::log::Log::DefaultLog.WriteError(ss.str().c_str());
    return sol::stack::push(L, description);
}

void megamol::core::LuaAPI::commonInit() {
    luaApiInterpreter_.set_exception_handler(&exceptionHandler);
    luaApiInterpreter_.open_libraries(); // AKA all of them

    RegisterCallback(MMC_LUA_MMGETENVVALUE, "(string name)\n\tReturn the value of env variable <name>.",
        &LuaAPI::GetEnvValue);
    RegisterCallback(MMC_LUA_MMGETCOMPILEMODE, "()\n\tReturns the compilation mode ('debug' or 'release').", &LuaAPI::GetCompileMode);
    RegisterCallback(MMC_LUA_MMGETOS, "()\n\tReturns the operating system ('windows', 'linux', or 'unknown').", &LuaAPI::GetOS);
    RegisterCallback(MMC_LUA_MMGETBITHWIDTH, "()\n\tReturns the bit width of the compiled executable.", &LuaAPI::GetBitWidth);
    RegisterCallback(MMC_LUA_MMGETMACHINENAME, "()\n\tReturns the machine name.", &LuaAPI::GetMachineName);
    RegisterCallback(MMC_LUA_MMGETPROCESSID, "()\n\tReturns the process id of the running MegaMol.", &LuaAPI::GetProcessID);

    // these need the instance
    RegisterCallback(MMC_LUA_MMHELP, "()\n\tReturns MegaMol Lua functions and help text", &LuaAPI::Help, this);
    RegisterCallback(MMC_LUA_MMCURRENTSCRIPTPATH,
        "()\n\tReturns the path of the currently running script, if possible. Empty string otherwise.",
        &LuaAPI::GetScriptPath, this);

    //luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::ReadTextFile>(MMC_LUA_MMREADTEXTFILE, "(string fileName, function func)\n\tReturn the file contents after processing it with func(content).");
    //luaApiInterpreter_.RegisterCallback<LuaAPI, &LuaAPI::WriteTextFile>(MMC_LUA_MMWRITETEXTFILE, "(string fileName, string content)\n\tWrite content to file. You CANNOT overwrite existing files!");
}

std::string megamol::core::LuaAPI::GetError(const sol::protected_function_result& res) const {
    if (!res.valid()) {
        auto err = sol::stack::get<sol::optional<std::string>>(luaApiInterpreter_.lua_state());
        if (err.has_value()) {
            return err.value();
        } else {
            return"unspecified sol error";
        }
    }
    return "";
}


/*
 * megamol::core::LuaAPI::LuaAPI
 */
megamol::core::LuaAPI::LuaAPI() {
    this->commonInit();
}


/*
 * megamol::core::LuaAPI::~LuaAPI
 */
megamol::core::LuaAPI::~LuaAPI() {}


std::string megamol::core::LuaAPI::GetScriptPath() {
    return this->currentScriptPath;
}

void megamol::core::LuaAPI::SetScriptPath(std::string const& scriptPath) {
    this->currentScriptPath = scriptPath;
}

static auto const tc_lua_cmd = 0x65B5E4;
static auto const error_handler = sol::script_default_on_error;


sol::safe_function_result megamol::core::LuaAPI::RunString(const std::string& script, std::string scriptPath) {
#ifdef MEGAMOL_USE_TRACY
    ZoneScopedNC("LuaAPI::RunString", tc_lua_cmd);
#endif
    // todo: locking!!!
    if (this->currentScriptPath.empty() && !scriptPath.empty()) {
        // the information got better, at least
        this->currentScriptPath = scriptPath;
    }
    auto res = luaApiInterpreter_.safe_script(script, error_handler);
    if (!res.valid()) {
        auto err = GetError(res);
        utility::log::Log::DefaultLog.WriteError(err.c_str());
    }
    return res;
}

void megamol::core::LuaAPI::Error(std::string description) {
    throw std::runtime_error(description);
}


unsigned int megamol::core::LuaAPI::GetBitWidth() {
    return vislib::sys::SystemInformation::SelfWordSize();
}


std::string megamol::core::LuaAPI::GetCompileMode() {
#ifdef _DEBUG
    return "debug";
#else
    return "release";
#endif
}


std::string megamol::core::LuaAPI::GetOS() {
    switch (vislib::sys::SystemInformation::SystemType()) {
    case vislib::sys::SystemInformation::OSTYPE_WINDOWS:
        return "windows";
    case vislib::sys::SystemInformation::OSTYPE_LINUX:
        return "linux";
    case vislib::sys::SystemInformation::OSTYPE_UNKNOWN:
    default:
        return "unknown";
    }
}


std::string megamol::core::LuaAPI::GetMachineName() {
    return vislib::sys::SystemInformation::ComputerNameA().PeekBuffer();
}


std::string megamol::core::LuaAPI::GetEnvValue(const std::string& variable) {
    if (vislib::sys::Environment::IsSet(variable.c_str())) {
        return vislib::sys::Environment::GetVariable(variable.c_str()).PeekBuffer();
    } else {
        throw std::invalid_argument("Environment variable " + variable + " is not defined.");
    }
}


unsigned int megamol::core::LuaAPI::GetProcessID() {
    vislib::StringA str;
#ifdef _WIN32
    unsigned int id = GetCurrentProcessId();
#else
    unsigned int id = static_cast<unsigned int>(getpid());
#endif // _WIN32
    return id;
}

std::string megamol::core::LuaAPI::Help() const {
    std::stringstream out;
    out << "MegaMol Lua Help:" << std::endl;
    for (const auto& item : helpContainer)
        out << item.first + item.second + "\n";
    return out.str().c_str();
}

int megamol::core::LuaAPI::ReadTextFile(lua_State* L) {
    // (string fileName, function func)

    //int n = lua_gettop(L);
    //if (n == 2) {
    //    const auto filename = luaL_checkstring(L, 1);
    //    std::ifstream t(filename);
    //    if (t.good()) {
    //        std::ostringstream buffer;
    //        buffer << t.rdbuf();

    //        // megamol::core::utility::log::Log::DefaultLog.WriteInfo(MMC_LUA_MMREADTEXTFILE ": read from file '%s':\n%s\n", filename,
    //        // buffer.str().c_str());

    //        lua_remove(L, 1); // get rid of the filename on the stack, leaving the function pointer
    //        lua_pushstring(L, buffer.str().c_str()); // put string parameter on top of stack
    //        if (lua_type(L, 1) == LUA_TNIL) {
    //            // no transformation function, just return the string
    //            return 1;
    //        }
    //        else {
    //            // call the function pointer
    //            lua_pcall(L, 1, 1, 0);
    //            n = lua_gettop(L);
    //            if (n != 1) {
    //                luaApiInterpreter_.ThrowError(MMC_LUA_MMREADTEXTFILE ": function did not return a string, this is bad.");
    //            }
    //            else {
    //                const auto newString = luaL_checkstring(L, 1);
    //                // megamol::core::utility::log::Log::DefaultLog.WriteInfo(MMC_LUA_MMREADTEXTFILE ": transformed into:\n%s\n",
    //                // newString);
    //                return 1;
    //            }
    //        }
    //    }
    //    else {
    //        std::string err = MMC_LUA_MMREADTEXTFILE ": cannot open file '";
    //        err += filename;
    //        err += "'.";
    //        luaApiInterpreter_.ThrowError(err);
    //    }
    //}
    //else {
    //    luaApiInterpreter_.ThrowError(MMC_LUA_MMREADTEXTFILE " requires two parameters, fileName and a function pointer");
    //}
    return 0;
}

int megamol::core::LuaAPI::WriteTextFile(lua_State* L) {
    //int n = lua_gettop(L);
    //if (n == 2) {
    //    const auto filename = luaL_checkstring(L, 1);
    //    if (std::filesystem::exists(filename)) {
    //        std::string err = MMC_LUA_MMWRITETEXTFILE ": Overwriting existing file '";
    //        err += filename;
    //        err += "' is not allowed!";
    //        luaApiInterpreter_.ThrowError(err);
    //    } else {
    //        std::ofstream t(filename, std::ofstream::out);
    //        std::string output = luaL_checkstring(L, 2);
    //        lua_remove(L, 1); // get rid of the filename on the stack
    //        lua_remove(L, 1); // get rid of the string on the stack

    //        if (t.good()) {
    //            t.write(output.c_str(), output.length());
    //        } else {
    //            std::string err = MMC_LUA_MMWRITETEXTFILE ": cannot open file '";
    //            err += filename;
    //            err += "'.";
    //            luaApiInterpreter_.ThrowError(err);
    //        }
    //    }
    //} else {
    //    luaApiInterpreter_.ThrowError(MMC_LUA_MMREADTEXTFILE " requires two parameters, fileName and a function pointer");
    //}
    return 0;
}

std::string megamol::core::LuaAPI::CurrentScriptPath() {
    return this->currentScriptPath;
}


//void megamol::core::LuaAPI::AddCallbacks(megamol::frontend_resources::LuaCallbacksCollection const& callbacks) {
//    verbatim_lambda_callbacks_.push_back(callbacks);
//
//    for (auto& c : callbacks.callbacks) {
//        assert(!std::get<0>(c).empty());
//    }
//
//    register_callbacks(verbatim_lambda_callbacks_.back());
//}
//
//static std::vector<std::string> getNames(megamol::frontend_resources::LuaCallbacksCollection const& callbacks) {
//    std::vector<std::string> names{};
//    names.reserve(callbacks.callbacks.size());
//
//    for (auto& c : callbacks.callbacks) {
//        names.push_back(std::get<0>(c));
//    }
//
//    return names;
//}
//
//void megamol::core::LuaAPI::RemoveCallbacks(megamol::frontend_resources::LuaCallbacksCollection const& callbacks,
//    bool delete_verbatim) {
//    RemoveCallbacks(getNames(callbacks));
//
//    if (delete_verbatim)
//        verbatim_lambda_callbacks_.remove_if([&](auto& item) {
//            return item.callbacks.empty() ||
//                   (item.callbacks.size() == callbacks.callbacks.size()
//                    && std::get<0>(item.callbacks.front()) == std::get<0>(callbacks.callbacks.front()));
//        });
//}
//
//void megamol::core::LuaAPI::RemoveCallbacks(std::vector<std::string> const& callback_names) {
//    for (auto& name : callback_names) {
//        UnregisterCallback(name);
//        wrapped_lambda_callbacks_.remove_if([&](auto& item) {
//            return std::get<0>(item) == name;
//        });
//    }
//}
//
//void megamol::core::LuaAPI::ClearCallbacks() {
//    for (auto& callbacks : verbatim_lambda_callbacks_) {
//        RemoveCallbacks(getNames(callbacks));
//    }
//    assert(wrapped_lambda_callbacks_.empty());
//
//    verbatim_lambda_callbacks_.clear();
//}
//
//void megamol::core::LuaAPI::register_callbacks(megamol::frontend_resources::LuaCallbacksCollection& callbacks) {
//    if (!callbacks.is_registered) {
//        for (auto& c : callbacks.callbacks) {
//            auto& name = std::get<0>(c);
//            auto& description = std::get<1>(c);
//            auto& func = std::get<2>(c);
//
//            if (auto found_it = std::find_if(wrapped_lambda_callbacks_.begin(), wrapped_lambda_callbacks_.end(),
//                    [&](auto const& elem) {
//                        return std::get<0>(elem) == name;
//                    });
//                found_it != wrapped_lambda_callbacks_.end()) {
//
//                RemoveCallbacks({std::get<0>(*found_it)});
//            }
//
//            wrapped_lambda_callbacks_.push_back({
//                name,
//                std::function<int(lua_State*)>{
//                    [=](lua_State* L) -> int {
//                        return func({L});
//                    }
//                }});
//            RegisterCallback(name, description, std::get<1>(wrapped_lambda_callbacks_.back()));
//        }
//        callbacks.is_registered = true;
//    }
//
//}
