#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/ShaderFactory.h"

#include <glowl/Texture2D.hpp>

#include "mmstd_gl/ModuleGL.h"
#include "OpenEXR/openexr.h"
#include "OpenEXR/ImfRgbaFile.h"
#include "Imath/ImathBox.h"

namespace megamol::compositing_gl {
class OpenEXRWriter : public mmstd_gl::ModuleGL {
public:
    static const char* ClassName() {
        return "OpenEXRWriter";
    }

    static const char* Description() {
        return "Module to write Texture as OpenEXR image file.";
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

    OpenEXRWriter();
    ~OpenEXRWriter() override;

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

    /** Slot for requesting the textures from another module, i.e. rhs connection */
    megamol::core::CallerSlot m_input_tex_slot;

    /** Texture that holds the data from the loaded .exr file */
    std::shared_ptr<glowl::Texture2D> m_output_texture;
    glowl::TextureLayout m_output_layout;

    uint32_t m_version;
};
} // namespace megamol::compositing_gl
