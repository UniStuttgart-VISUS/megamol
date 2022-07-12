/*
 * TriangleMeshRenderer.cpp
 * Copyright (C) 2006-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "TriangleMeshRenderer.h"

#define SHADER_BASED

using namespace megamol;
using namespace megamol::molecularmaps;

/*
 * TriangleMeshRenderer::TriangleMeshRenderer
 */
TriangleMeshRenderer::TriangleMeshRenderer(void)
        : AbstractLocalRenderer()
        , faces(nullptr)
        , vertex_colors(nullptr)
        , vertex_normals(nullptr)
        , vertices(nullptr)
        , colorBuffer(nullptr)
        , positionBuffer(nullptr)
        , normalBuffer(nullptr)
        , shader_3(nullptr)
        , shader_4(nullptr)
        , numValuesPerColor(0)
        , vertex_array(0) {}

/*
 * TriangleMeshRenderer::~TriangleMeshRenderer
 */
TriangleMeshRenderer::~TriangleMeshRenderer(void) {
    this->Release();
}

/*
 * TriangleMeshRenderer::create
 */
bool TriangleMeshRenderer::create(void) {
#ifdef SHADER_BASED

    constexpr char vertex_code_3_c[] = R""""(
        #version 430

        layout(location = 0) in vec3 in_position;
        layout(location = 1) in vec3 in_color;
        layout(location = 2) in vec3 in_normal;

        uniform mat4 proj = mat4(1.0);
        uniform mat4 view = mat4(1.0);

        out vec4 color;
        out vec3 normal;

        void main() {
            normal = normalize(in_normal);
            color = vec4(in_color, 1.0);
            gl_Position = proj * view * glm::vec4(in_position, 1.0);
        }
        )"""";
    const std::string vertex_code_3 = vertex_code_3_c;

    constexpr char vertex_code_4_c[] = R""""(
        #version 430

        layout(location = 0) in vec3 in_position;
        layout(location = 1) in vec4 in_color;
        layout(location = 2) in vec3 in_normal;

        uniform mat4 proj = mat4(1.0);
        uniform mat4 view = mat4(1.0);

        out vec4 color;
        out vec3 normal;

        void main() {
            normal = normalize(in_normal);
            color = in_color;
            gl_Position = proj * view * glm::vec4(in_position, 1.0);
        }
    )"""";
    const std::string vertex_code_4 = vertex_code_4_c;

    constexpr char fragment_code_c[] = R""""(
        #version 430

        layout(location = 0) out vec4 out_color;

        in vec4 color;
        in vec3 normal;

        uniform mat4 proj = mat4(1.0);
        uniform mat4 view = mat4(1.0);

        uniform vec3 light_direction = vec3(0.75, -1.0, 0.0);
        uniform vec4 light_params = vec4(0.2, 0.8, 0.4, 10.0);
        uniform vec3 viewVec = vec3(0.0, 0.0, -1.0);

        vec4 localLighting(vec4 color, vec3 normal, vec3 vv, vec3 light) {
            if(length(light_direction) < 0.0001) return color;
            float ndotl = dot(normal, light);
            vec3 r = normalize(2.0 * vec3(ndotl) * normal - light);
            vec4 l = light_params;
            float alph = color.w;
            vec3 result = l.x * color.xyz + l.y * color.xyz * max(ndotl, 0.0) + l.z * vec3(pow(max(dot(r, -vv), 0.0), l.w));
            return vec4(result, alph);
        }

        void main() {
            vec3 n = normalize(normal);
            vec3 l = normalize(-light_direction);
            out_color = localLighting(color, normal, viewVec, l);
        }
    )"""";
    const std::string fragment_code = fragment_code_c;

    std::vector<std::pair<glowl::GLSLProgram::ShaderType, std::string>> shader_srcs_3;
    std::vector<std::pair<glowl::GLSLProgram::ShaderType, std::string>> shader_srcs_4;

    shader_srcs_3.push_back({glowl::GLSLProgram::ShaderType::Vertex, vertex_code_3});
    shader_srcs_3.push_back({glowl::GLSLProgram::ShaderType::Fragment, fragment_code});

    shader_srcs_4.push_back({glowl::GLSLProgram::ShaderType::Vertex, vertex_code_4});
    shader_srcs_4.push_back({glowl::GLSLProgram::ShaderType::Fragment, fragment_code});

    try {
        if (this->shader_3 != nullptr) {
            this->shader_3.reset();
        }
        this->shader_3 = std::make_shared<glowl::GLSLProgram>(shader_srcs_3);
    } catch (glowl::GLSLProgramException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[TriangleMeshRenderer] Error during shader program creation: %s. [%s, %s, line %d]\n", exc.what(),
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    try {
        if (this->shader_4 != nullptr) {
            this->shader_4.reset();
        }
        this->shader_4 = std::make_shared<glowl::GLSLProgram>(shader_srcs_4);
    } catch (glowl::GLSLProgramException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[TriangleMeshRenderer] Error during shader program creation: %s. [%s, %s, line %d]\n", exc.what(),
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

#endif
    return true;
}

/*
 * TriangleMeshRenderer::release
 */
void TriangleMeshRenderer::release(void) {}

/*
 * TriangleMeshRenderer::Render
 */
bool TriangleMeshRenderer::Render(core::view::CallRender3DGL& call, bool lighting) {

    if (this->faces == nullptr)
        return false;
    if (this->vertices == nullptr)
        return false;
    if (this->vertex_colors == nullptr)
        return false;
    if (this->vertex_normals == nullptr)
        return false;
    if (this->numValuesPerColor != 3 && this->numValuesPerColor != 4)
        return false;

#ifndef SHADER_BASED
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);

    glColorPointer(this->numValuesPerColor, GL_FLOAT, 0, this->vertex_colors->data());
    glVertexPointer(3, GL_FLOAT, 0, this->vertices->data());
    glNormalPointer(GL_FLOAT, 0, this->vertex_normals->data());
    glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(this->faces->size()), GL_UNSIGNED_INT, this->faces->data());

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
#else
    glBindVertexArray(this->vertex_array);
    std::shared_ptr<glowl::GLSLProgram> shader = nullptr;
    if (this->numValuesPerColor == 3 && this->shader_3 != nullptr) {
        shader = shader_3;
    } else if (this->numValuesPerColor == 4 && this->shader_4 != nullptr) {
        shader = shader_4;
    } else {
        glBindVertexArray(0);
        return false;
    }
    shader->use();

    auto cam = call.GetCamera();
    glm::mat4 view = cam.getViewMatrix();
    glm::mat4 proj = cam.getProjectionMatrix();
    auto view_vector = cam.getPose().direction;

    glm::vec3 light_direction = glm::vec3(0.75f, -1.0f, 0.0f);
    if (!lighting) {
        light_direction = glm::vec3(0.0f, 0.0f, 0.0f);
    }

    glUniformMatrix4fv(shader->getUniformLocation("view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(shader->getUniformLocation("proj"), 1, GL_FALSE, glm::value_ptr(proj));
    glUniform3f(shader->getUniformLocation("light_direction"), light_direction.x, light_direction.y, light_direction.z);
    glUniform3f(shader->getUniformLocation("viewVec"), view_vector.x, view_vector.y, view_vector.z);

    glDrawElements(GL_TRIANGLES, static_cast<uint32_t>(this->faces->size()), GL_UNSIGNED_INT, nullptr);

    glUseProgram(0);
    glBindVertexArray(0);

#endif

    return true;
}

/*
 * TriangleMeshRenderer::RenderWireFrame
 */
bool TriangleMeshRenderer::RenderWireFrame(core::view::CallRender3DGL& call, bool lighting) {
    GLint oldpolymode[2];
    glGetIntegerv(GL_POLYGON_MODE, oldpolymode);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    this->Render(call, lighting);
    glPolygonMode(GL_FRONT, oldpolymode[0]);
    glPolygonMode(GL_BACK, oldpolymode[1]);
    return true;
}

/*
 * TriangleMeshRenderer::update
 */
bool TriangleMeshRenderer::update(const std::vector<uint>* faces, const std::vector<float>* vertices,
    const std::vector<float>* vertex_colors, const std::vector<float>* vertex_normals, unsigned int numValuesPerColor) {

    this->faces = faces;
    this->vertices = vertices;
    this->vertex_colors = vertex_colors;
    this->vertex_normals = vertex_normals;

    if (numValuesPerColor == 3 || numValuesPerColor == 4) {
        this->numValuesPerColor = numValuesPerColor;
    } else {
        return false;
    }

#ifdef SHADER_BASED
    if (this->vertex_array == 0) {
        glGenVertexArrays(1, &this->vertex_array);
    }
    glBindVertexArray(this->vertex_array);

    if (this->positionBuffer != nullptr) {
        this->positionBuffer.reset();
    }
    if (this->colorBuffer != nullptr) {
        this->colorBuffer.reset();
    }
    if (this->normalBuffer != nullptr) {
        this->normalBuffer.reset();
    }
    if (this->faceBuffer != nullptr) {
        this->faceBuffer.reset();
    }

    this->positionBuffer = std::make_shared<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    this->colorBuffer = std::make_shared<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    this->normalBuffer = std::make_shared<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    this->faceBuffer = std::make_shared<glowl::BufferObject>(GL_ELEMENT_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    // Create and fill position buffer
    this->positionBuffer->bind();
    if (this->vertices != nullptr) {
        this->positionBuffer->rebuffer(*this->vertices);
    } else {
        this->positionBuffer->rebuffer(std::vector<float>());
    }
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Create and fill color buffer
    this->colorBuffer->bind();
    if (this->vertex_colors != nullptr) {
        this->colorBuffer->rebuffer(*this->vertex_colors);
    } else {
        this->colorBuffer->rebuffer(std::vector<float>());
    }
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, this->numValuesPerColor, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Create and fill normal buffer
    this->normalBuffer->bind();
    if (this->vertex_normals != nullptr) {
        this->normalBuffer->rebuffer(*this->vertex_normals);
    } else {
        this->normalBuffer->rebuffer(std::vector<float>());
    }
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Create and fill index buffer
    this->faceBuffer->bind();
    if (this->faces != nullptr) {
        this->faceBuffer->rebuffer(*this->faces);
    } else {
        this->faceBuffer->rebuffer(std::vector<uint32_t>());
    }
    glBindVertexArray(0);
#endif

    return true;
}
