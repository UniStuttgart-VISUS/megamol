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

    bool getDataCallback(core::Call& caller);
    bool getMetaDataCallback(core::Call& caller);

private:
    /** Slot for loading the file */
    core::param::ParamSlot m_filename_slot;

    /** Button triggering writing of exr file */
    core::param::ParamSlot m_button_slot;

    /** String Input Slots to set channel names of output file */
    core::param::ParamSlot m_channel_name_red;
    core::param::ParamSlot m_channel_name_green;
    core::param::ParamSlot m_channel_name_blue;
    core::param::ParamSlot m_channel_name_alpha;

    /** Slot for requesting the textures from another module, i.e. rhs connection */
    megamol::core::CallerSlot m_input_tex_slot;

    /** lhs connection. Passthrough for input texture*/
    megamol::core::CalleeSlot m_texture_pipe_out;

    /** Texture that holds the data from the loaded .exr file */
    std::shared_ptr<glowl::Texture2D> m_output_texture;
    glowl::TextureLayout m_output_layout;

    uint32_t m_version;
    bool saveRequested = false;

    uint32_t version_;

    bool triggerButtonClicked(core::param::ParamSlot& slot);
    int formatToChannelNumber(GLenum format);

    /**
     * \brief Method sets interface to only allow editing of the last currently named channel or adding a new one.
     * This prevents cases like Red and alpha channel being mapped without mapping green and blue.
     *
     *
     */
    void setRelevantParamState();
};
} // namespace megamol::compositing_gl
