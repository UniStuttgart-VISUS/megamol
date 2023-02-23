#pragma once

#include "mmcore/Module.h"

#include "CUDA_Context.h"

namespace megamol::core {
class ResourceTestModule : public Module {
public:
    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        Module::requested_lifetime_resources(req);
        req.require<frontend_resources::CUDA_Context>();
    }

    static const char* ClassName() {
        return "ResourceTestModule";
    }

    static const char* Description() {
        return "Showcase for frontend resource handling";
    }

    static bool IsAvailable() {
        return true;
    }

    ResourceTestModule();

    ~ResourceTestModule() override;

protected:
    bool create() override;

    void release() override;

private:
};
} // namespace megamol::core
