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

#include <array>
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
        Color = 0,
        TFByte = 1,
        TFWord = 2,
        CatByte = 3,
        CatWord = 4,
    };

    ImageDisplay2D(const msf::ShaderFactoryOptionsOpenGL& shaderFactoryOptions);
    virtual ~ImageDisplay2D() noexcept;

    bool updateTexture(const vislib::graphics::BitmapImage& image);
    bool updateGraph(const ImageSeries::graph::GraphData2D& graph, float baseRadius, float edgeWidth);
    bool updateTransferFunction(unsigned int texture, const std::array<float, 2>& valueRange);

    const std::array<float, 2>& getValueRange() const;

    glm::vec2 getImageSize() const;

    bool render(megamol::mmstd_gl::CallRender2DGL& call, bool render_graph = true);
    bool render(megamol::mmstd_gl::CallRender3DGL& call, bool render_graph = true);

    void setDisplayMode(Mode mode);
    Mode getDisplayMode() const;

private:
    bool renderImpl(std::shared_ptr<glowl::FramebufferObject> framebuffer, const glm::mat4& matrix, bool render_graph);
    static bool textureLayoutEquals(const glowl::TextureLayout& layout1, const glowl::TextureLayout& layout2);

    template<typename T>
    std::array<float, 2> calcValueRange(const T* arr, std::size_t size) const {
        constexpr auto minPossible = std::numeric_limits<T>::lowest();
        constexpr auto maxPossible = std::numeric_limits<T>::max();

        auto min = maxPossible;
        auto max = minPossible;

        for (std::size_t i = 0; i < size; ++i) {
            if (arr[i] != minPossible && arr[i] != maxPossible) {
                min = std::min(min, arr[i]);
                max = std::max(max, arr[i]);
            }
        }

        return std::array<float, 2>{static_cast<float>(min), static_cast<float>(max)};
    };

    std::shared_ptr<glowl::GLSLProgram> shader, edge_shader, node_shader;
    std::unique_ptr<glowl::Texture2D> texture;
    std::unique_ptr<glowl::Mesh> mesh, edge_mesh, node_mesh;

    std::vector<glm::vec2> graph_node_vertices;
    std::vector<float> graph_node_radii, graph_node_types;

    std::vector<glm::vec4> graph_edge_lines;
    std::vector<float> graph_edge_weights;
    float edgeWidth;

    GLuint node_radius_buffer, node_type_buffer, edge_weight_buffer;
    GLint width, height;

    unsigned int transferFunction;
    std::array<float, 2> valueRange, usedValueRange;

    Mode mode = Mode::Color;
};

} // namespace megamol::ImageSeries::GL
