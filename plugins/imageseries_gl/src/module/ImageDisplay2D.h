#pragma once

#include "mmcore_gl/utility/ShaderFactory.h"

#include "mmstd_gl/renderer/CallRender2DGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"

#include "vislib/graphics/BitmapImage.h"

#include "imageseries/graph/GraphData2D.h"

#include "glowl/FramebufferObject.hpp"
#include "glowl/GLSLProgram.hpp"
#include "glowl/Mesh.hpp"
#include "glowl/Texture2D.hpp"

#include <memory>
#include <vector>

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
    virtual ~ImageDisplay2D() noexcept;

    bool updateTexture(const vislib::graphics::BitmapImage& image);
    bool updateGraph(const ImageSeries::graph::GraphData2D& graph, float baseRadius);

    glm::vec2 getImageSize() const;

    bool render(megamol::mmstd_gl::CallRender2DGL& call, bool render_graph = true);
    bool render(megamol::mmstd_gl::CallRender3DGL& call, bool render_graph = true);

    void setDisplayMode(Mode mode);
    Mode getDisplayMode() const;

private:
    Mode getEffectiveDisplayMode() const;

    bool renderImpl(std::shared_ptr<glowl::FramebufferObject> framebuffer, const glm::mat4& matrix, bool render_graph);
    static bool textureLayoutEquals(const glowl::TextureLayout& layout1, const glowl::TextureLayout& layout2);

    std::shared_ptr<glowl::GLSLProgram> shader, edge_shader, node_shader;
    std::unique_ptr<glowl::Texture2D> texture;
    std::unique_ptr<glowl::Mesh> mesh, edge_mesh, node_mesh;

    std::vector<glm::vec2> graph_node_vertices;
    std::vector<float> graph_node_radii, graph_node_types;
    std::vector<glm::vec4> graph_edge_lines;

    GLuint node_radius_buffer, node_type_buffer;
    GLint width, height;

    Mode mode = Mode::Auto;
};

} // namespace megamol::ImageSeries::GL
