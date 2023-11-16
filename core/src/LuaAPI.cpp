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

#define MMC_LUA_MMLOG "mmLog"
#define MMC_LUA_MMLOGINFO "mmLogInfo"
#define MMC_LUA_MMDEBUGPRINT "mmDebugPrint"
#define MMC_LUA_MMHELP "mmHelp"

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

int silentExceptionHandler(lua_State* L, sol::optional<const std::exception&> maybe_exception,
    sol::string_view description) {
    return sol::stack::push(L, description);
}

void megamol::core::LuaAPI::commonInit() {
    luaApiInterpreter_.set_exception_handler(&silentExceptionHandler);
    luaApiInterpreter_.open_libraries(); // AKA all of them

    RegisterCallback(MMC_LUA_MMGETENVVALUE, "(string name)\n\tReturn the value of env variable <name>.",
        &LuaAPI::GetEnvValue);
    RegisterCallback(MMC_LUA_MMGETCOMPILEMODE, "()\n\tReturns the compilation mode ('debug' or 'release').",
        &LuaAPI::GetCompileMode);
    RegisterCallback(MMC_LUA_MMGETOS, "()\n\tReturns the operating system ('windows', 'linux', or 'unknown').",
        &LuaAPI::GetOS);
    RegisterCallback(MMC_LUA_MMGETBITHWIDTH, "()\n\tReturns the bit width of the compiled executable.",
        &LuaAPI::GetBitWidth);
    RegisterCallback(MMC_LUA_MMGETMACHINENAME, "()\n\tReturns the machine name.", &LuaAPI::GetMachineName);
    RegisterCallback(MMC_LUA_MMGETPROCESSID, "()\n\tReturns the process id of the running MegaMol.",
        &LuaAPI::GetProcessID);
    RegisterCallback(MMC_LUA_MMREADTEXTFILE,
        "(string fileName, function func)\n\tReturn the file contents after processing it with func(content).",
        &LuaAPI::ReadTextFile);
    RegisterCallback(MMC_LUA_MMWRITETEXTFILE,
        "(string fileName, string content)\n\tWrite content to file. You CANNOT overwrite existing files!",
        &LuaAPI::WriteTextFile);
    RegisterCallback(MMC_LUA_MMLOG, "(int level, ...)\n\tLog to MegaMol console. Level constants are LOGINFO, LOGWARNING, LOGERROR.", &LuaAPI::Log);
    RegisterCallback(MMC_LUA_MMLOGINFO, "(...)\n\tLog to MegaMol console with LOGINFO level.", &LuaAPI::LogInfo);
    RegisterCallback(MMC_LUA_MMDEBUGPRINT, "(...)\n\tLog to MegaMol console with LOGINFO level.", &LuaAPI::LogInfo);
    luaApiInterpreter_["print"] = luaApiInterpreter_[MMC_LUA_MMLOGINFO];
    luaApiInterpreter_["LOGINFO"] = static_cast<int>(utility::log::Log::log_level::info);
    luaApiInterpreter_["LOGWARNING"] = static_cast<int>(utility::log::Log::log_level::warn);
    luaApiInterpreter_["LOGERROR"] = static_cast<int>(utility::log::Log::log_level::error);

    // these need the instance
    RegisterCallback(MMC_LUA_MMHELP, "()\n\tReturns MegaMol Lua functions and help text", &LuaAPI::Help, this);
    RegisterCallback(MMC_LUA_MMCURRENTSCRIPTPATH,
        "()\n\tReturns the path of the currently running script, if possible. Empty string otherwise.",
        &LuaAPI::GetScriptPath, this);
}

std::string megamol::core::LuaAPI::GetError(const sol::protected_function_result& res) const {
    if (!res.valid()) {
        auto err = sol::stack::get<sol::optional<std::string>>(luaApiInterpreter_.lua_state());
        if (err.has_value()) {
            return err.value();
        } else {
            return "unspecified sol error";
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

sol::safe_function_result megamol::core::LuaAPI::RunString(const std::string& script, std::string scriptPath) {
#ifdef MEGAMOL_USE_TRACY
    ZoneScopedNC("LuaAPI::RunString", tc_lua_cmd);
#endif
    // todo: locking!!!
    if (this->currentScriptPath.empty() && !scriptPath.empty()) {
        // the information got better, at least
        this->currentScriptPath = scriptPath;
    }
    // alternative: sol::script_default_on_error
    auto res = luaApiInterpreter_.safe_script(script, sol::script_pass_on_error);
    return res;
}

void megamol::core::LuaAPI::ThrowError(const std::string& description) {
    throw std::runtime_error(description);
}


std::string megamol::core::LuaAPI::TypeToString(sol::safe_function_result& res, int index_offset) {
    switch (res.get_type()) {
    case sol::type::none:
        return "none";
    case sol::type::nil:
        return "nil";
    case sol::type::string:
        return res.get<std::string>(index_offset);
    case sol::type::number:
        return std::to_string(res.get<double>(index_offset));
    case sol::type::thread:
        return "thread: " + std::to_string(res.get<uint64_t>(index_offset));
    case sol::type::boolean:
        return res.get<bool>(index_offset) ? "true" : "false";
    case sol::type::function:
        return "thread: " + std::to_string(res.get<uint64_t>(index_offset));
    case sol::type::userdata:
        return "(userdata)";
    case sol::type::lightuserdata:
        return "(userdata)";
    case sol::type::table:
        return "(table)";
    case sol::type::poly:
        return "(poly)";
    }
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


void megamol::core::LuaAPI::Log(int level, const std::string &message){
    utility::log::Log::DefaultLog.WriteMsg(static_cast<utility::log::Log::log_level>(level), message.c_str());
}


void megamol::core::LuaAPI::LogInfo(const std::string &message){
    utility::log::Log::DefaultLog.WriteInfo(message.c_str());
}


std::string megamol::core::LuaAPI::ReadTextFile(std::string filename, sol::optional<sol::function> transformation) {
    std::ifstream t(filename);
    if (t.good()) {
        std::stringstream stream;
        stream << t.rdbuf();
        if (transformation.has_value()) {
            auto res = transformation.value()(stream.str());
            return res;
        } else {
            return stream.str();
        }
    } else {
        ThrowError("cannot open file " + filename);
        return "";
    }
}


void megamol::core::LuaAPI::WriteTextFile(std::string filename, std::string content) {
    if (std::filesystem::exists(filename)) {
        ThrowError("Overwriting existing file " + filename + " is not allowed!");
    } else {
        std::ofstream t(filename, std::ofstream::out);
        if (t.good()) {
            t.write(content.c_str(), content.length());
        } else {
            ThrowError("Cannot open file " + filename + " for writing.");
        }
    }
}

std::string megamol::core::LuaAPI::CurrentScriptPath() {
    return this->currentScriptPath;
}
