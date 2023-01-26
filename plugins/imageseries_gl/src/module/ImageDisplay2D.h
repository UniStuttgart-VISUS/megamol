#pragma once

#include "mmcore_gl/utility/ShaderFactory.h"

#include "mmstd_gl/renderer/CallRender2DGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"

#include "vislib/graphics/BitmapImage.h"

#include "glowl/FramebufferObject.hpp"
#include "glowl/GLSLProgram.hpp"
#include "glowl/Mesh.hpp"
#include "glowl/Texture2D.hpp"

#include <memory>

namespace megamol::core_gl::view {
class CallRender2DGL;
class CallRender3DGL;
} // namespace megamol::core_gl::view

namespace megamol::ImageSeries::GL {

class ImageDisplay2D {
public:
    enum class Mode {
        Auto = 0,
        Color = 1,
        Grayscale = 2,
        Labels = 3,
        TimeDifference = 4,
    };

    ImageDisplay2D(const msf::ShaderFactoryOptionsOpenGL& shaderFactoryOptions);

    bool updateTexture(const vislib::graphics::BitmapImage& image);

    glm::vec2 getImageSize() const;

    bool render(megamol::mmstd_gl::CallRender2DGL& call);
    bool render(megamol::mmstd_gl::CallRender3DGL& call);

    void setDisplayMode(Mode mode);
    Mode getDisplayMode() const;

private:
    Mode getEffectiveDisplayMode() const;

    bool renderImpl(std::shared_ptr<glowl::FramebufferObject> framebuffer, const glm::mat4& matrix);
    static bool textureLayoutEquals(const glowl::TextureLayout& layout1, const glowl::TextureLayout& layout2);

    std::shared_ptr<glowl::GLSLProgram> shader;
    std::unique_ptr<glowl::Texture2D> texture;
    std::unique_ptr<glowl::Mesh> mesh;

    Mode mode = Mode::Auto;
};

} // namespace megamol::ImageSeries::GL
