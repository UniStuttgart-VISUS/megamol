/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "glowl/Texture2D.hpp"

namespace megamol {
namespace compositing_gl {

class PNGDataSource : public core::Module {
public:
    static const char* ClassName(void) {
        return "PNGDataSource";
    }
    static const char* Description(void) {
        return "Data source for loading .png files";
    }
    static bool IsAvailable(void) {
        return true;
    }

    PNGDataSource(void);
    virtual ~PNGDataSource(void);

protected:
    virtual bool create(void);
    virtual void release(void);

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

} /* end namespace compositing */
} /* end namespace megamol */
