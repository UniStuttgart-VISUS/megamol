/*
* OSPRayRenderer.h
* Copyright (C) 2009-2015 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#ifndef OSPRAY_RENDERER_H_INCLUDED
#define OSPRAY_RENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <stdint.h>
#include <vector>
#include "vislib/graphics/gl/GLSLShader.h"
#include "ospray/ospray.h"


namespace megamol {
namespace ospray {

    class OSPRayRenderer {
    protected:
    
        /**
        * helper function for rendering the OSPRay texture
        * @param GLSL shader
        * @param GL texture object
        * @param OSPRay texture
        * @param GL vertex array object
        * @param image/window width
        * @param image/window heigth
        */
        void renderTexture2D(vislib::graphics::gl::GLSLShader &shader, GLuint &texture, const uint32_t * fb, GLuint &vertexarray, int &width, int &height);

        /**
        * helper function for setting up the OSPRay screen
        * @param GL vertex array
        * @param GL vertex buffer object
        * @param GL texture object
        */
        void setupTextureScreen(GLuint &vertexarray, GLuint &vbo, GLuint &texture);

        /**
        * helper function for initializing OSPRay
        * @param OSPRay renderer object
        * @param OSPRay camera object
        * @param OSPRay world object
        */
        void setupOSPRay(OSPRenderer &renderer, OSPCamera &camera, OSPModel &world, OSPGeometry &geometry, const char * geometry_name);

        /** 
        * color transfer helper
        * @param array with gray scales
        * @param transferfunction table/texture
        * @param transferfunction table/texture size
        * @param target array (rgba)
        */
        void colorTransferGray(std::vector<float> &grayArray, float* transferTable, unsigned int &tableSize, std::vector<float> &rgbaArray);

    };

} // end namespace ospray
} // end namespace megamol

#endif /* OSPRAY_RENDERER_H_INCLUDED */