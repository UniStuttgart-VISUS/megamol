#include "ImageDisplay2D.h"

#include "mmcore/utility/log/Log.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd_gl/renderer/CallRender2DGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"

#include "glowl/VertexLayout.hpp"

#include <numeric>
#include <sstream>
#include <stdexcept>

namespace megamol::ImageSeries::GL {

ImageDisplay2D::ImageDisplay2D(const msf::ShaderFactoryOptionsOpenGL& shaderFactoryOptions) {
    try {
        // (1) Image
        {
            shader = core::utility::make_shared_glowl_shader("ImageSeriesRenderer", shaderFactoryOptions,
                "imageseries_gl/ImageSeriesRenderer.vert.glsl", "imageseries_gl/ImageSeriesRenderer.frag.glsl");

            std::vector<std::vector<float>> vertices = {{0, 0, 0, 1, 1, 0, 1, 1}};
            std::vector<uint32_t> indices = {0, 1, 2, 1, 2, 3};
            std::vector<glowl::VertexLayout> vertexLayout = {
                glowl::VertexLayout(8, {glowl::VertexLayout::Attribute(2, GL_FLOAT, GL_FALSE, 0)})};

            mesh = std::make_unique<glowl::Mesh>(
                vertices, vertexLayout, indices, GL_UNSIGNED_INT, GL_TRIANGLES, GL_STATIC_DRAW);
        }

        // (2) Edges
        {
            edge_shader = core::utility::make_shared_glowl_shader("GraphEdgeRenderer", shaderFactoryOptions,
                "imageseries_gl/GraphEdgeRenderer.vert.glsl", "imageseries_gl/GraphEdgeRenderer.frag.glsl");
        }

        // (3) Nodes
        {
            node_shader = core::utility::make_shared_glowl_shader("GraphNodeRenderer", shaderFactoryOptions,
                "imageseries_gl/GraphNodeRenderer.vert.glsl", "imageseries_gl/GraphNodeRenderer.geom.glsl",
                "imageseries_gl/GraphNodeRenderer.frag.glsl");

            glGenBuffers(1, &node_radius_buffer);
        }
    } catch (glowl::GLSLProgramException const& ex) {
        std::stringstream ss;
        ss << "Error creating shaders for ImageDisplay: ";
        ss << ex.what() << std::endl;

        throw std::runtime_error(ss.str());
    }
}

ImageDisplay2D::~ImageDisplay2D() {
    glDeleteBuffers(1, &node_radius_buffer);
}

bool ImageDisplay2D::updateTexture(const vislib::graphics::BitmapImage& image) {
    width = image.Width();
    height = image.Height();

    // TODO handle channel layouts
    glowl::TextureLayout textureLayout;

    switch (image.GetChannelCount()) {
    case 1:
        textureLayout.format = GL_RED;
        textureLayout.internal_format = GL_R16;
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

    // Alignment
    GLint packAlignment = 0;
    if (textureLayout.width % 4 != 0) {
        glGetIntegerv(GL_UNPACK_ALIGNMENT, &packAlignment);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    }

    if (texture && textureLayoutEquals(texture->getTextureLayout(), textureLayout)) {
        // Reuse texture
        glTextureSubImage2D(texture->getName(), 0, 0, 0, textureLayout.width, textureLayout.height,
            textureLayout.format, textureLayout.type, image.PeekData());
    } else {
        // Recreate texture
        texture = std::make_unique<glowl::Texture2D>("ImageSeriesRenderer", textureLayout, image.PeekData());
    }

    if (packAlignment) {
        glPixelStorei(GL_UNPACK_ALIGNMENT, packAlignment);
    }

    return true;
}

bool ImageDisplay2D::updateGraph(const ImageSeries::graph::GraphData2D& graph) {
    const auto& nodes = graph.getNodes();
    const auto& edges = graph.getEdges();

    // Nodes
    if (!nodes.empty()) {
        graph_node_vertices.clear();
        graph_node_radii.clear();

        graph_node_vertices.reserve(nodes.size());
        graph_node_radii.reserve(nodes.size());

        for (const auto& node : nodes) {
            graph_node_vertices.push_back(node.centerOfMass);
            graph_node_radii.push_back(node.edgeCountIn * 0.5f * std::sqrt(2.0f) + 2.0f);
        }

        std::vector<uint32_t> node_indices(graph_node_vertices.size());
        std::iota(node_indices.begin(), node_indices.end(), 0);

        std::vector<glowl::VertexLayout> node_vertexLayout = {
            glowl::VertexLayout(8, {glowl::VertexLayout::Attribute(2, GL_FLOAT, GL_FALSE, 0)})};

        try {
            node_mesh = std::make_unique<glowl::Mesh>(
                std::vector{const_cast<const void*>(static_cast<void*>(graph_node_vertices.data()))},
                std::vector{graph_node_vertices.size() * sizeof(glm::vec2)}, node_vertexLayout, node_indices.data(),
                node_indices.size() * sizeof(uint32_t), GL_UNSIGNED_INT, GL_POINTS, GL_STATIC_DRAW);
        } catch (const std::exception& ex) {
            core::utility::log::Log::DefaultLog.WriteError("ImageDisplay2D - Error creating buffer for node vertices:\n%s", ex.what());
        }

        node_mesh->bindVertexArray();

        glBindBuffer(GL_ARRAY_BUFFER, node_radius_buffer);
        glBufferData(GL_ARRAY_BUFFER, graph_node_radii.size() * sizeof(float), graph_node_radii.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

        glBindVertexArray(0);
    } else {
        node_mesh = nullptr;
    }

    // Edges
    if (!edges.empty()) {
        graph_edge_lines.clear();

        graph_edge_lines.reserve(edges.size());

        for (const auto& edge : edges) {
            const auto& from = nodes[edge.from];
            const auto& to = nodes[edge.to];

            graph_edge_lines.push_back(glm::vec4(from.centerOfMass, to.centerOfMass));
        }

        std::vector<uint32_t> edge_indices(graph_edge_lines.size() * 2);
        std::iota(edge_indices.begin(), edge_indices.end(), 0);

        std::vector<glowl::VertexLayout> edge_vertexLayout = {
            glowl::VertexLayout(8, {glowl::VertexLayout::Attribute(2, GL_FLOAT, GL_FALSE, 0)})};

        try {
            edge_mesh = std::make_unique<glowl::Mesh>(
                std::vector{const_cast<const void*>(static_cast<void*>(graph_edge_lines.data()))},
                std::vector{graph_edge_lines.size() * sizeof(glm::vec4)}, edge_vertexLayout, edge_indices.data(),
                edge_indices.size() * sizeof(uint32_t), GL_UNSIGNED_INT, GL_LINES, GL_STATIC_DRAW);
        } catch (const std::exception& ex) {
            core::utility::log::Log::DefaultLog.WriteError(
                "ImageDisplay2D - Error creating buffer for edge vertices:\n%s", ex.what());
        }
    } else {
        edge_mesh = nullptr;
    }

    return true;
}

glm::vec2 ImageDisplay2D::getImageSize() const {
    return glm::vec2(width, height);
}

bool ImageDisplay2D::render(megamol::mmstd_gl::CallRender2DGL& call) {
    auto camera = call.GetCamera();
    return renderImpl(call.GetFramebuffer(), camera.getProjectionMatrix() * camera.getViewMatrix());
}

bool ImageDisplay2D::render(megamol::mmstd_gl::CallRender3DGL& call) {
    auto camera = call.GetCamera();
    return renderImpl(call.GetFramebuffer(), camera.getProjectionMatrix() * camera.getViewMatrix());
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

    GLboolean revert_depth_test;
    glGetBooleanv(GL_DEPTH_TEST, &revert_depth_test);
    glDisable(GL_DEPTH_TEST);

    // (1) Render image
    glActiveTexture(GL_TEXTURE0);
    texture->bindTexture();

    shader->use();
    shader->setUniform("matrix", glm::scale(matrix, glm::vec3(width, height, 1.f)));
    shader->setUniform("image", 0);
    shader->setUniform("displayMode", static_cast<GLint>(getEffectiveDisplayMode()));

    mesh->draw();

    glBindTexture(GL_TEXTURE_2D, 0);

    GLboolean revert_blend;
    glGetBooleanv(GL_BLEND, &revert_blend);
    glEnable(GL_BLEND);

    // (2) Render edges
    if (edge_mesh) {
        edge_shader->use();
        edge_shader->setUniform(
            "matrix", glm::scale(glm::translate(matrix, glm::vec3(0.0, height, 0.0)), glm::vec3(1.0, -1.0, 1.0)));

        edge_mesh->draw();
    }

    // (3) Render nodes
    if (node_mesh) {
        node_shader->use();
        node_shader->setUniform(
            "matrix", glm::scale(glm::translate(matrix, glm::vec3(0.0, height, 0.0)), glm::vec3(1.0, -1.0, 1.0)));

        node_mesh->draw();
    }

    // Reset draw state
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (revert_depth_test) {
        glEnable(GL_DEPTH_TEST);
    }
    if (!revert_blend) {
        glDisable(GL_BLEND);
    }

    return true;
}

bool ImageDisplay2D::textureLayoutEquals(const glowl::TextureLayout& layout1, const glowl::TextureLayout& layout2) {
    return layout1.type == layout2.type && layout1.internal_format == layout2.internal_format &&
           layout1.levels == layout2.levels && layout1.format == layout2.format && layout1.width == layout2.width &&
           layout1.height == layout2.height;
}

} // namespace megamol::ImageSeries::GL
