#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/ShaderFactory.h"

#include <glowl/Texture2D.hpp>

#include "Imath/ImathBox.h"
#include "OpenEXR/ImfRgbaFile.h"
#include "OpenEXR/openexr.h"
#include "mmstd_gl/ModuleGL.h"

namespace megamol::compositing_gl {
class OpenEXRReader : public mmstd_gl::ModuleGL {
public:
    static const char* ClassName() {
        return "OpenEXRReader";
    }

    static const char* Description() {
        return "Module to read OpenEXR image files.";
    }

    static bool IsAvailable() {
        return true;
    }

    /**
     * \brief Updates texture format variables.
     *
     * @return 'true' if successfully updated, 'false' otherwise
     */
    bool textureFormatUpdate();

    OpenEXRReader();
    ~OpenEXRReader() override;

protected:
    bool create() override;

    void release() override;

private:
    bool getDataCallback(core::Call& caller);
    bool getMetaDataCallback(core::Call& caller);
    /** Slot for loading the file */
    core::param::ParamSlot m_filename_slot;

    /** Slot for showing the image width. Read-only */
    core::param::ParamSlot m_image_width_slot;

    /** Slot for showing the image height. Read-only */
    core::param::ParamSlot m_image_height_slot;

    /** Slots to choose channel mappings (Input file to output Texture) */
    core::param::ParamSlot red_mapping_slot;
    core::param::ParamSlot green_mapping_slot;
    core::param::ParamSlot blue_mapping_slot;
    core::param::ParamSlot alpha_mapping_slot;

    /** Slot for requesting the output textures from this module, i.e. lhs connection */
    megamol::core::CalleeSlot m_output_tex_slot;

    /** Texture that holds the data from the loaded .exr file */
    std::shared_ptr<glowl::Texture2D> m_output_texture;
    glowl::TextureLayout m_output_layout;

    uint32_t m_version;
};
} // namespace megamol::compositing_gl
