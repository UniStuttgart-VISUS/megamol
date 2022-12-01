/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <glowl/Texture2D.hpp>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "mmstd_gl/ModuleGL.h"

namespace megamol::compositing_gl {

class PNGDataSource : public mmstd_gl::ModuleGL {
public:
    static const char* ClassName() {
        return "PNGDataSource";
    }
    static const char* Description() {
        return "Data source for loading .png files";
    }
    static bool IsAvailable() {
        return true;
    }

    PNGDataSource();
    virtual ~PNGDataSource();

protected:
    virtual bool create();
    virtual void release();

private:
    bool getDataCallback(core::Call& caller);
    bool getMetaDataCallback(core::Call& caller);


    /** Slot for loading the .png file */
    core::param::ParamSlot m_filename_slot;

    /** Slot for showing the image width. Read-only */
    core::param::ParamSlot m_image_width_slot;

    /** Slot for showing the image height. Read-only */
    core::param::ParamSlot m_image_height_slot;

    /** Slot for requesting the output textures from this module, i.e. lhs connection */
    megamol::core::CalleeSlot m_output_tex_slot;

    /** Texture that holds the data from the loaded .png file */
    std::shared_ptr<glowl::Texture2D> m_output_texture;
    glowl::TextureLayout m_output_layout;

    uint32_t m_version;
};

} // namespace megamol::compositing_gl
