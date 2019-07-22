#pragma once

#include "stdafx.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "mmcore/utility/ShaderSourceFactory.h"

namespace megamol {
    namespace core {
        namespace utility {

            bool InitializeShader(megamol::core::utility::ShaderSourceFactory* factory, vislib::graphics::gl::GLSLGeometryShader& shader, const std::string& vertexName, const std::string& fragmentName, const std::string& geometryName);
            bool InitializeShader(megamol::core::utility::ShaderSourceFactory* factory, vislib::graphics::gl::GLSLShader& shader, const std::string& vertexName, const std::string& fragmentName);

            void AddShaderSnippet(megamol::core::utility::ShaderSourceFactory* factory, vislib::graphics::gl::ShaderSource& source, const std::string& snippetName, bool containsCode = false);

        }
    }
}