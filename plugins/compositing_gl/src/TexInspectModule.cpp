/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "TexInspectModule.h"

#include "compositing_gl/CompositingCalls.h"

using namespace megamol::compositing_gl;


TexInspectModule::TexInspectModule()
        : megamol::core::Module()
        , tex_inspector_()
        , get_data_slot_("getData", "Slot to fetch data")
        , output_tex_slot_("OutputTexture", "Gives access to the resulting output texture") {
    this->output_tex_slot_.SetCallback(
        compositing::CallTexture2D::ClassName(), "GetData", &TexInspectModule::getDataCallback);
    this->output_tex_slot_.SetCallback(
        compositing::CallTexture2D::ClassName(), "GetMetaData", &TexInspectModule::getMetaDataCallback);
    this->MakeSlotAvailable(&this->output_tex_slot_);

    this->get_data_slot_.SetCompatibleCall<compositing::CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->get_data_slot_);
}

TexInspectModule::~TexInspectModule() {
    this->Release();
}

bool TexInspectModule::create() {
    return true;
}

void TexInspectModule::release() {}

bool TexInspectModule::getDataCallback(core::Call& caller) {
    auto lhs_tc = dynamic_cast<compositing::CallTexture2D*>(&caller);
    auto* rhs_ct2d = this->get_data_slot_.CallAs<compositing::CallTexture2D>();

    if (rhs_ct2d != nullptr) {
        if (!(*rhs_ct2d)(0)) {
            return false;
        }
    } else {
        core::utility::log::Log::DefaultLog.WriteError("Need a rhs connection.");
        return false;
    }

    if (rhs_ct2d->hasUpdate()) {
        auto tex = rhs_ct2d->getData();
        auto tex_handle = tex->getName();
        auto tex_width = tex->getWidth();
        auto tex_height = tex->getHeight();

        tex_inspector_.SetTexture((void*)(intptr_t)tex_handle, tex_width, tex_height);
        tex_inspector_.ShowWindow();
    }

    lhs_tc->setData(rhs_ct2d->getData(), rhs_ct2d->version());

    return true;
}

bool TexInspectModule::getMetaDataCallback(core::Call& caller) {
    return true;
}
