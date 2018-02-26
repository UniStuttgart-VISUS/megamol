#include "stdafx.h"
#include "mmcore/utility/LuaHostService.h"
#include "mmcore/CoreInstance.h"
#include "vislib/math/mathfunctions.h"


using namespace megamol;
using megamol::core::AbstractNamedObject;
using megamol::core::AbstractNamedObjectContainer;

unsigned int megamol::core::utility::LuaHostService::ID = 0;

megamol::core::utility::LuaHostService::LuaHostService(core::CoreInstance& core) : AbstractService(core),
        serverThread(), serverRunning(false), address("tcp://*:33333") {
    // Intentionally empty
}

megamol::core::utility::LuaHostService::~LuaHostService() {
    assert(!this->IsEnabled());
}

bool megamol::core::utility::LuaHostService::Initalize(bool& autoEnable) {
    using vislib::sys::Log;
    context = ZMQContextUser::Instance();

    // fetch configuration
    auto& cfg = GetCoreInstance().Configuration();

    if (cfg.IsConfigValueSet("LRHostAddress")) {
        mmcValueType t;
        const void* d = cfg.GetValue(MMC_CFGID_VARIABLE, "LRHostAddress", &t);
        switch (t) {
        case MMC_TYPE_CSTR:
            address = static_cast<const char*>(d);
            Log::DefaultLog.WriteInfo("Set LRHostAddress = \"%s\"", address.c_str());
            break;
        case MMC_TYPE_WSTR:
            address = vislib::StringA(static_cast<const wchar_t*>(d));
            Log::DefaultLog.WriteInfo("Set LRHostAddress = \"%s\"", address.c_str());
            break;
        default:
            Log::DefaultLog.WriteWarn("Unable to set LRHostAddress: expected string, but found type %d", static_cast<int>(t));
            break;
        }
    } else {
        Log::DefaultLog.WriteInfo("Default LRHostAddress = \"%s\"", address.c_str());
    }

    autoEnable = true; // default behavior

    if (cfg.IsConfigValueSet("LRHostEnable")) {
        mmcValueType t;
        const void* d = cfg.GetValue(MMC_CFGID_VARIABLE, "LRHostEnable", &t);
        bool silent = false;
        switch (t) {
        case MMC_TYPE_INT32: autoEnable = ((*static_cast<const int32_t*>(d)) != 0); break;
        case MMC_TYPE_UINT32: autoEnable = ((*static_cast<const uint32_t*>(d)) != 0); break;
        case MMC_TYPE_INT64: autoEnable = ((*static_cast<const int64_t*>(d)) != 0); break;
        case MMC_TYPE_UINT64: autoEnable = ((*static_cast<const uint64_t*>(d)) != 0); break;
        case MMC_TYPE_BYTE: autoEnable = ((*static_cast<const uint8_t*>(d)) != 0); break;
        case MMC_TYPE_BOOL: autoEnable = *static_cast<const bool*>(d); break;
        case MMC_TYPE_FLOAT: autoEnable = !vislib::math::IsEqual(*static_cast<const float*>(d), 0.0f); break;
        case MMC_TYPE_CSTR:
            autoEnable = vislib::CharTraitsA::ParseBool(static_cast<const char*>(d));
            break;
        case MMC_TYPE_WSTR:
            autoEnable = vislib::CharTraitsW::ParseBool(static_cast<const wchar_t*>(d));
            break;
        default:
            Log::DefaultLog.WriteWarn("Unable to set LRHostEnable: expected string, but found type %d", static_cast<int>(t));
            silent = true;
            break;
        }
        if (!silent) {
            Log::DefaultLog.WriteInfo("Set LRHostEnable = \"%s\"", autoEnable ? "true" : "false");
        }
    } else {
        Log::DefaultLog.WriteInfo("Default LRHostEnable = \"%s\"", autoEnable ? "true" : "false");
    }

    return true;
}

bool megamol::core::utility::LuaHostService::Deinitialize() {
    Disable();
    context.reset();
    return true;
}

void megamol::core::utility::LuaHostService::SetAddress(const std::string& ad) {
    if (serverRunning) {
        // restart server
        disableImpl();
        address = ad;
        enableImpl();
    } else {
        address = ad;
    }

}

bool megamol::core::utility::LuaHostService::enableImpl() {
    assert(serverRunning == false);
    serverThread = std::thread([&](){
        this->serve();
      });
    while (!serverRunning) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return true;
}

bool megamol::core::utility::LuaHostService::disableImpl() {
    serverRunning = false;
    if (serverThread.joinable()) serverThread.join();
    return true;
}


void megamol::core::utility::LuaHostService::serve() {
    using vislib::sys::Log;

    zmq::socket_t socket(*context, ZMQ_REP);

    try {
        serverRunning = true;
        socket.bind(address);
        socket.setsockopt(ZMQ_RCVTIMEO, 100); // message receive time out 100ms

        Log::DefaultLog.WriteInfo("LRH Server socket opened on \"%s\"", address.c_str());

        while (serverRunning) {
            zmq::message_t request;
            while (serverRunning && !socket.recv(&request, ZMQ_DONTWAIT)) {
                // no messages available ATM
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (!serverRunning) break;

            std::string request_str(reinterpret_cast<char*>(request.data()), request.size());
            std::string reply = makeAnswer(request_str);
            //if (reply.empty()) {
            //    reply = "ERR";
            //}
            socket.send(reply.data(), reply.size());
        }

    } catch (std::exception& error) {
        Log::DefaultLog.WriteError("Error on LRH Server: %s", error.what());

    } catch (...) {
        Log::DefaultLog.WriteError("Error on LRH Server: unknown exception");
    }

    try {
        socket.close();
    } catch (...) {}
    Log::DefaultLog.WriteInfo("LRH Server socket closed");

}

std::string megamol::core::utility::LuaHostService::makeAnswer(const std::string& req) {

    if (req.empty()) return std::string("Null Command.");

    std::string result;
    int ok = this->GetCoreInstance().GetLuaState()->RunString(req, result);
    if (ok) {
        //vislib::sys::Log::DefaultLog.WriteInfo("Lua execution is OK and returned '%s'", result.c_str());
    } else {
        vislib::sys::Log::DefaultLog.WriteError("Lua execution is NOT OK and returned '%s'", result.c_str());
        result = "Error: " + result;
    }
    return result;
}
