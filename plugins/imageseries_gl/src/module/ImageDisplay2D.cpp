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
    // TODO reuse textures instead of recreating them
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

    textureLayout.width = image.Width();
    textureLayout.height = image.Height();
    textureLayout.levels = 1;
    texture = std::make_unique<glowl::Texture2D>("ImageSeriesRenderer", textureLayout, image.PeekData());

    return true;
}

bool ImageDisplay2D::render(megamol::core_gl::view::CallRender2DGL& call) {
    return renderImpl(call.GetFramebuffer(), call.GetCamera().getViewMatrix());
}

bool ImageDisplay2D::render(megamol::core_gl::view::CallRender3DGL& call) {
    auto camera = call.GetCamera();
    return renderImpl(call.GetFramebuffer(), camera.getProjectionMatrix() * camera.getViewMatrix());
}

bool ImageDisplay2D::renderImpl(std::shared_ptr<glowl::FramebufferObject> framebuffer, const glm::mat4& matrix) {
    if (!framebuffer || !shader || !texture || !mesh) {
        return false;
    }

    framebuffer->bindToDraw();

    glActiveTexture(GL_TEXTURE0);
    texture->bindTexture();

    shader->use();
    shader->setUniform("matrix", matrix);
    shader->setUniform("image", 0);
    shader->setUniform("grayscale", static_cast<GLint>(texture->getFormat() == GL_RED));

    mesh->draw();

    // Reset draw state
    glUseProgram(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return true;
}

} // namespace megamol::ImageSeries::GL
