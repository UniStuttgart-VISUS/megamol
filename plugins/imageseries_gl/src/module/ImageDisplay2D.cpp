#include "ImageDisplay2D.h"

#include "mmcore/utility/log/Log.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmcore_gl/view/CallRender2DGL.h"
#include "mmcore_gl/view/CallRender3DGL.h"

#include "glowl/VertexLayout.hpp"


namespace megamol::ImageSeries::GL {

ImageDisplay2D::ImageDisplay2D(const msf::ShaderFactoryOptionsOpenGL& shaderFactoryOptions) {
    shader = core::utility::make_shared_glowl_shader("ImageSeriesRenderer", shaderFactoryOptions,
        "imageseries_gl/ImageSeriesRenderer.vert.glsl", "imageseries_gl/ImageSeriesRenderer.frag.glsl");

    std::vector<std::vector<float>> vertices = {{0, 0, 0, 1, 1, 0, 1, 1}};
    std::vector<uint32_t> indices = {0, 1, 2, 1, 2, 3};
    std::vector<glowl::VertexLayout> vertexLayout = {
        glowl::VertexLayout(8, {glowl::VertexLayout::Attribute(2, GL_FLOAT, GL_FALSE, 0)})};

    mesh =
        std::make_unique<glowl::Mesh>(vertices, vertexLayout, indices, GL_UNSIGNED_INT, GL_TRIANGLES, GL_STATIC_DRAW);
}

bool ImageDisplay2D::updateTexture(const vislib::graphics::BitmapImage& image) {
    // TODO handle channel layouts
    glowl::TextureLayout textureLayout;

    switch (image.GetChannelCount()) {
    case 1:
        textureLayout.format = GL_RED;
        textureLayout.internal_format = GL_R8;
        break;
    case 2:
        textureLayout.format = GL_RG;
        textureLayout.internal_format = GL_RG8;
        break;
    case 3:
        textureLayout.format = GL_RGB;
        textureLayout.internal_format = GL_RGB8;
        break;
    case 4:
        textureLayout.format = GL_RGBA;
        textureLayout.internal_format = GL_RGBA8;
        break;
    default:
        return false;
    }

    switch (image.GetChannelType()) {
    case vislib::graphics::BitmapImage::ChannelType::CHANNELTYPE_BYTE:
        textureLayout.type = GL_UNSIGNED_BYTE;
        break;
    case vislib::graphics::BitmapImage::ChannelType::CHANNELTYPE_WORD:
        textureLayout.type = GL_UNSIGNED_SHORT;
        break;
    case vislib::graphics::BitmapImage::ChannelType::CHANNELTYPE_FLOAT:
        textureLayout.type = GL_FLOAT;
        break;
    default:
        return false;
    }

    textureLayout.int_parameters.emplace_back(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    textureLayout.int_parameters.emplace_back(GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    textureLayout.width = image.Width();
    textureLayout.height = image.Height();
    textureLayout.levels = 1;

    if (texture && textureLayoutEquals(texture->getTextureLayout(), textureLayout)) {
        // Reuse texture
        glTextureSubImage2D(texture->getName(), 0, 0, 0, textureLayout.width, textureLayout.height,
            textureLayout.format, textureLayout.type, image.PeekData());
    } else {
        // Recreate texture
        texture = std::make_unique<glowl::Texture2D>("ImageSeriesRenderer", textureLayout, image.PeekData());
    }

    return true;
}

bool ImageDisplay2D::render(megamol::core_gl::view::CallRender2DGL& call) {
    auto camera = call.GetCamera();
    return renderImpl(call.GetFramebuffer(), camera.getProjectionMatrix() * camera.getViewMatrix());
}

bool ImageDisplay2D::render(megamol::core_gl::view::CallRender3DGL& call) {
    auto camera = call.GetCamera();
    return renderImpl(call.GetFramebuffer(), camera.getProjectionMatrix() * camera.getViewMatrix());
}

glm::vec2 ImageDisplay2D::getImageSize() const {
    if (texture) {
        // Fit to height of 100 by default (TODO: make this configurable)
        float height = 100.f;
        float width = std::max<float>(texture->getWidth(), 1) / std::max<float>(texture->getHeight(), 1) * height;
        return glm::vec2(width, height);
    } else {
        return glm::vec2(2448.f, 2050.f);
    }
}

void ImageDisplay2D::setDisplayMode(Mode mode) {
    this->mode = mode;
}

ImageDisplay2D::Mode ImageDisplay2D::getDisplayMode() const {
    return mode;
}

ImageDisplay2D::Mode ImageDisplay2D::getEffectiveDisplayMode() const {
    if (mode == Mode::Auto) {
        return texture && texture->getFormat() == GL_RED ? Mode::Grayscale : Mode::Color;
    } else {
        return mode;
    }
}

bool ImageDisplay2D::renderImpl(std::shared_ptr<glowl::FramebufferObject> framebuffer, const glm::mat4& matrix) {
    if (!framebuffer || !shader || !texture || !mesh) {
        return false;
    }

    framebuffer->bindToDraw();

    glActiveTexture(GL_TEXTURE0);
    texture->bindTexture();

    shader->use();
    shader->setUniform("matrix", glm::scale(matrix, glm::vec3(getImageSize(), 1.f)));
    shader->setUniform("image", 0);
    shader->setUniform("displayMode", static_cast<GLint>(getEffectiveDisplayMode()));

    mesh->draw();

    // Reset draw state
    glUseProgram(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return true;
}

bool ImageDisplay2D::textureLayoutEquals(const glowl::TextureLayout& layout1, const glowl::TextureLayout& layout2) {
    return layout1.type == layout2.type && layout1.internal_format == layout2.internal_format &&
           layout1.levels == layout2.levels && layout1.format == layout2.format && layout1.width == layout2.width &&
           layout1.height == layout2.height;
}

} // namespace megamol::ImageSeries::GL
