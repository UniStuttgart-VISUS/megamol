/*
 * TriangleMeshRenderer.h
 * Copyright (C) 2006-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMMOLMAPPLG_TRIANGLEMESHRENDERER_H_INCLUDED
#define MMMOLMAPPLG_TRIANGLEMESHRENDERER_H_INCLUDED
#pragma once

#include "AbstractLocalRenderer.h"
#include "glowl/BufferObject.hpp"
#include "glowl/GLSLProgram.hpp"

#include "helper_includes/helper_math.h"

namespace megamol {
namespace molecularmaps {

class TriangleMeshRenderer : public AbstractLocalRenderer {
public:
    /** Ctor */
    TriangleMeshRenderer(void);

    /** Dtor */
    virtual ~TriangleMeshRenderer(void);

    /**
     * Initializes the renderer
     */
    virtual bool create(void);

    /**
     * Invokes the rendering calls
     */
    virtual bool Render(core_gl::view::CallRender3DGL& call, bool lighting = true);

    /**
     * Invokes the rendering calls using wireframe rendering
     */
    virtual bool RenderWireFrame(core_gl::view::CallRender3DGL& call, bool lighting = true);

    /**
     * Update function for the local data to render
     *
     * @param faces Pointer to the vector containing the face vertex indices
     * @param vertices Pointer to the vector containing the vertex positions
     * @param vertex_colors Pointer to the vector containing the vertex colors.
     * @param vertex_normals Pointer to the vector containing the vertex normals.
     * @param numValuesPerColor The number of color values per vertex. (1 for intensity, 3 for RGB, 4 for RGBA.
     * Standard: RGB)
     * @return True on success, false otherwise.
     */
    bool update(const std::vector<uint>* faces, const std::vector<float>* vertices,
        const std::vector<float>* vertex_colors, const std::vector<float>* vertex_normals,
        unsigned int numValuesPerColor = 3);

protected:
    /**
     * Frees all needed resources used by this renderer
     */
    virtual void release(void);

private:
    /** Pointer to the object holding the color vertex buffer */
    std::shared_ptr<glowl::BufferObject> colorBuffer;

    /** Pointer to the object holding the face index buffer */
    std::shared_ptr<glowl::BufferObject> faceBuffer;

    /** Pointer to the vector containing the face vertex indices */
    const std::vector<uint>* faces;

    /** Pointer to the object holding the normal vertex buffer */
    std::shared_ptr<glowl::BufferObject> normalBuffer;

    /** Number of color values per color in vertex_colors */
    unsigned int numValuesPerColor;

    /** Pointer to the object holding the position vertex buffer */
    std::shared_ptr<glowl::BufferObject> positionBuffer;

    /** Pointer to the shader program for 3-element color arrays */
    std::shared_ptr<glowl::GLSLProgram> shader_3;

    /** Pointer to the shader program for 4-element color arrays */
    std::shared_ptr<glowl::GLSLProgram> shader_4;

    /** Handle for the vertex array */
    GLuint vertex_array;

    /** Pointer to the vector containing the face vertex colors */
    const std::vector<float>* vertex_colors;

    /** Pointer to the vector containing the face vertex normals */
    const std::vector<float>* vertex_normals;

    /** Pointer to the vector containing the vertex positions */
    const std::vector<float>* vertices;
};

} /* end namespace molecularmaps */
} /* end namespace megamol */

#endif /* MMMOLMAPPLG_TRIANGLEMESHRENDERER_H_INCLUDED */
