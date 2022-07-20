#pragma once

#include "mmcore/Module.h"

#include "CUDA_Context.h"

namespace megamol::core {
class ResourceTestModule : public Module {
public:
    std::vector<std::string> requested_lifetime_resources() override {
        auto req = Module::requested_lifetime_resources();
        req.push_back(frontend_resources::CUDA_Context_Req_Name);
        return req;
    }

    static const char* ClassName(void) {
        return "ResourceTestModule";
    }

    static const char* Description(void) {
        return "Showcase for frontend resource handling";
    }

    static bool IsAvailable(void) {
        return true;
    }

    ResourceTestModule();

    virtual ~ResourceTestModule();

protected:
    bool create() override;

    void release() override;

private:
};
} // namespace megamol::core
