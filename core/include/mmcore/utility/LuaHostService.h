#pragma once

#include <string>
#include <thread>
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/AbstractNamedObjectContainer.h"
#include "mmcore/AbstractService.h"
//#include "CommandFunctionPtr.h"
#include <atomic>
#include <map>
#include "mmcore/utility/ZMQContextUser.h"

namespace megamol {
namespace core {
namespace utility {

class LuaHostService : public core::AbstractService {
public:
    static unsigned int ID;

    virtual const char* Name() const { return "LuaRemote"; }

    LuaHostService(core::CoreInstance& core);
    virtual ~LuaHostService();

    virtual bool Initalize(bool& autoEnable);
    virtual bool Deinitialize();

    inline const std::string& GetAddress(void) const { return address; }
    void SetAddress(const std::string& ad);

protected:
    virtual bool enableImpl();
    virtual bool disableImpl();

private:
    void serve();
    void servePair();
    std::string makeAnswer(const std::string& req);
    std::string makePairAnswer(const std::string& req) const;
    std::atomic<int> lastPairPort;

    // ModuleGraphAccess mgAccess;
    ZMQContextUser::ptr context;

    std::thread serverThread;
    std::vector<std::thread> pairThreads;
    bool serverRunning;

    std::string address;
};


} /* namespace utility */
} /* namespace core */
} /* namespace megamol */

