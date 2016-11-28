/*
* OSPRayRenderer.cpp
* Copyright (C) 2009-2015 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "vislib/sys/Path.h"
#include "OSPRayRenderer.h"
#include "ospray/ospray.h"
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace megamol::ospray;

void OSPRayRenderer::renderTexture2D(vislib::graphics::gl::GLSLShader &shader, GLuint &texture, const uint32_t * fb, GLuint &vertexarray, int &width, int &height) {

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, fb);

    glUniform1i(shader.ParameterLocation("tex"), 0);

    glBindVertexArray(vertexarray);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void OSPRayRenderer::setupTextureScreen(GLuint &vertexarray, GLuint &vbo, GLuint &texture) {

    // setup vertexarray
    float screenVertices[] = { 0.0f,0.0f, 1.0f,0.0f, 0.0f,1.0f, 1.0f,1.0f };

    glGenVertexArrays(1, &vertexarray);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vertexarray);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 4 * 2, screenVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);


    // setup texture
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void OSPRayRenderer::initOSPRay() {

    // init OSPRay
    // command line arguments
    int ac = 0;
    const char *av = " ";
    /* initialize ospray without commandline arguments
    instead modify ospray environment vaiables */
    ospInit(&ac, &av);
}

void OSPRayRenderer::setupOSPRay(OSPRenderer &renderer, OSPCamera &camera, OSPModel &world, OSPGeometry &geometry, const char * geometry_name, const char * renderer_name) {

    // create and setup renderer
    renderer = ospNewRenderer(renderer_name);
    camera = ospNewCamera("perspective");
    world = ospNewModel();
    ospSetObject(renderer, "model", world);
    ospSetObject(renderer, "camera", camera);
    geometry = ospNewGeometry(geometry_name);
    ospAddGeometry(world, geometry);

}

void OSPRayRenderer::colorTransferGray(std::vector<float> &grayArray, float* transferTable, unsigned int &tableSize, std::vector<float> &rgbaArray) {
  
    float gray_max = *std::max_element(grayArray.begin(), grayArray.end());
    float gray_min = *std::min_element(grayArray.begin(), grayArray.end());

    for (auto &gray : grayArray) {
        float scaled_gray;
        if ((gray_max - gray_min) <= 1e-4f) {
            scaled_gray = 0;
        } else {
            scaled_gray = (gray - gray_min) / (gray_max - gray_min);
        }
        if (transferTable == NULL && tableSize == 0) {
            for (int i = 0; i < 3; i++) {
                rgbaArray.push_back((0.3 + scaled_gray) / 1.3);
            }
            rgbaArray.push_back(1.0f);
        } else {
            float exact_tf = (tableSize - 1) * scaled_gray;
            int floor = std::floor(exact_tf);
            float tail = exact_tf - (float)floor;
            floor *= 4;
            for (int i = 0; i < 4; i++) {
                float colorFloor = transferTable[floor + i];
                float colorCeil = transferTable[floor + i + 4];
                float finalColor = colorFloor + (colorCeil - colorFloor)*(tail);
                rgbaArray.push_back(finalColor);
            }
        }
    }
}


OSPTexture2D OSPRayRenderer::TextureFromFile(vislib::TString fileName) {

    fileName = vislib::sys::Path::Resolve(fileName);

    vislib::TString ext = vislib::TString("");
    size_t pos = fileName.FindLast('.');
    if (pos != std::string::npos)
        ext = fileName.Substring(pos + 1);

    FILE *file = fopen(vislib::StringA(fileName).PeekBuffer(), "rb");
    if (!file)
        throw std::runtime_error("Could not read file");


    if (ext == vislib::TString("ppm")) {
        try {
            int rc, peekchar;

            const int LINESZ = 10000;
            char lineBuf[LINESZ + 1];

            // read format specifier:
            int format = 0;
            rc = fscanf(file, "P%i\n", &format);
            if (format != 6)
                throw std::runtime_error("Wrong PPM format.");

            // skip all comment lines
            peekchar = getc(file);
            while (peekchar == '#') {
                auto tmp = fgets(lineBuf, LINESZ, file);
                (void)tmp;
                peekchar = getc(file);
            } ungetc(peekchar, file);

            // read width and height from first non-comment line
            int width = -1, height = -1;
            rc = fscanf(file, "%i %i\n", &width, &height);
            if (rc != 2)
                throw std::runtime_error("Could not read PPM width and height.");

            // skip all comment lines
            peekchar = getc(file);
            while (peekchar == '#') {
                auto tmp = fgets(lineBuf, LINESZ, file);
                (void)tmp;
                peekchar = getc(file);
            } ungetc(peekchar, file);

            // read maxval
            int maxVal = -1;
            rc = fscanf(file, "%i", &maxVal);
            peekchar = getc(file);

            void* data;
            data = new unsigned char[width * height * 3];
            rc = fread(data, width*height * 3, 1, file);
            // flip in y, because OSPRay's textures have the origin at the lower left corner
            unsigned char *texels = (unsigned char *)data;
            for (int y = 0; y < height / 2; y++)
                for (int x = 0; x < width * 3; x++)
                    std::swap(texels[y*width * 3 + x], texels[(height - 1 - y)*width * 3 + x]);

            OSPTexture2D ret_tex = ospNewTexture2D({ width, height }, OSP_TEXTURE_RGB8, texels);
            return ret_tex;
        } catch (std::runtime_error e) {
            std::cerr << e.what() << std::endl;
        }

    } else if (ext == vislib::TString("pfm")) {
        try {
            // Note: the PFM file specification does not support comments thus we don't skip any
            // http://netpbm.sourceforge.net/doc/pfm.html
            int rc = 0;


            // read format specifier:
            // PF: color floating point image
            // Pf: grayscae floating point image
            char format[2] = { 0 };
            if (fscanf(file, "%c%c\n", &format[0], &format[1]) != 2)
                throw std::runtime_error("could not fscanf");

            if (format[0] != 'P' || (format[1] != 'F' && format[1] != 'f')) {
                throw std::runtime_error("Invalid pfm texture file, header is not PF or Pf");
            }
            int numChannels = 3;
            if (format[1] == 'f') {
                numChannels = 1;
            }

            // read width and height
            int width = -1;
            int height = -1;
            rc = fscanf(file, "%i %i\n", &width, &height);
            if (rc != 2) {
                throw std::runtime_error("Could not parse width and height in PF PFM file");
            }

            // read scale factor/endiannes
            float scaleEndian = 0.0;
            rc = fscanf(file, "%f\n", &scaleEndian);

            if (rc != 1) {
                throw std::runtime_error("Could not parse scale factor/endianness in PF PFM file");
            }
            if (scaleEndian == 0.0) {
                throw std::runtime_error("Scale factor/endianness in PF PFM file can not be 0");
            }
            if (scaleEndian > 0.0) {
                throw std::runtime_error("Could not parse PF PFM file");
            }
            float scaleFactor = std::abs(scaleEndian);

            int depth = sizeof(float);
            void* data;
            data = new float[width * height * numChannels];
            if (fread(data, sizeof(float), width * height * numChannels, file) != width * height * numChannels) {
                throw std::runtime_error("could not fread");
            }
            // flip in y, because OSPRay's textures have the origin at the lower left corner
            float *texels = (float *)data;
            for (int y = 0; y < height / 2; ++y) {
                for (int x = 0; x < width * numChannels; ++x) {
                    // Scale the pixels by the scale factor
                    texels[y * width * numChannels + x] = texels[y * width * numChannels + x] * scaleFactor;
                    texels[(height - 1 - y) * width * numChannels + x] = texels[(height - 1 - y) * width * numChannels + x] * scaleFactor;
                    std::swap(texels[y * width * numChannels + x], texels[(height - 1 - y) * width * numChannels + x]);
                }
            }
            OSPTextureFormat type = OSP_TEXTURE_R8;

            if (numChannels == 1) type = OSP_TEXTURE_R32F;
            if (numChannels == 3) type = OSP_TEXTURE_RGB32F;
            if (numChannels == 4) type = OSP_TEXTURE_RGBA32F;

            OSPTexture2D ret_tex = ospNewTexture2D({ width, height }, type, texels);
            return ret_tex;
        } catch (std::runtime_error e) {
            std::cerr << e.what() << std::endl;
        }
    } else {
        std::cerr << "File type not supported. Only PPM and PFM file formats allowed." << std::endl;
    }
}


