#pragma once

#include "mmcore/AbstractService.h"
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/AbstractNamedObjectContainer.h"
#include <thread>
#include <zmq.hpp>
#include "ZMQContextUser.h"
#include <string>
//#include "CommandFunctionPtr.h"
#include <map>

namespace megamol {
namespace core {
namespace utility {

    class LuaHostService : public core::AbstractService {
    public:

        static unsigned int ID;

        virtual const char* Name() const {
            return "LuaRemote";
        }

        LuaHostService(core::CoreInstance& core);
        virtual ~LuaHostService();

        virtual bool Initalize(bool& autoEnable);
        virtual bool Deinitialize();

        inline const std::string& GetAddress(void) const {
            return address;
        }
        void SetAddress(const std::string& ad);

    protected:
        virtual bool enableImpl();
        virtual bool disableImpl();

    private:

        void serve();
        std::string makeAnswer(const std::string& req);

        //ModuleGraphAccess mgAccess;
        ZMQContextUser::ptr context;

        std::thread serverThread;
        bool serverRunning;

        std::string address;
    };

} /* namespace utility */
} /* namespace core */
} /* namespace megamol */
