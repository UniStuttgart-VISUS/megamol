#pragma once

#include "mmcore/Module.h"

#include "CUDA_Context.h"

namespace megamol::core {
class ResourceTestModule : public Module {
public:
    void requested_lifetime_resources(frontend_resources::ResourceRequest& req) override {
        Module::requested_lifetime_resources(req);
        req.require<frontend_resources::CUDA_Context>();
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
