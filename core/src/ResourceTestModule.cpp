#include "mmcore/ResourceTestModule.h"


megamol::core::ResourceTestModule::ResourceTestModule() {}


megamol::core::ResourceTestModule::~ResourceTestModule() {
    this->Release();
}


bool megamol::core::ResourceTestModule::create() {
    auto fit = std::find_if(this->frontend_resources.begin(), this->frontend_resources.end(),
        [](auto const& el) { return el.getIdentifier() == "CUDA_Context"; });

    if (fit != this->frontend_resources.end()) {
        core::utility::log::Log::DefaultLog.WriteInfo("[ResourceTestModule] Got CUDA context");

        auto& cuda_res = fit->getResource<frontend_resources::CUDA_Context>();
        if (cuda_res.ctx_ != nullptr) {
            core::utility::log::Log::DefaultLog.WriteInfo("[ResourceTestModule] CUDA context pointer exists");
        }
    }

    return true;
}


void megamol::core::ResourceTestModule::release() {}
