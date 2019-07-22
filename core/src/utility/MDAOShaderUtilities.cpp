#include "stdafx.h"
#include "mmcore/utility/MDAOShaderUtilities.h"

#include <iostream>
#include <string>


using namespace megamol::core::utility;


bool megamol::core::utility::InitializeShader(megamol::core::utility::ShaderSourceFactory* factory, vislib::graphics::gl::GLSLShader& shader, const std::string& vertexName, const std::string& fragmentName)
{
	vislib::graphics::gl::ShaderSource vert, frag;

    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        std::cerr<<"Failed to init OpenGL extensions: GLSLShader"<<std::endl;
        return false;
    }

	// Initialize our shader
	try {
		// Try to make the vertex shader
		if (!factory->MakeShaderSource(vertexName.c_str(), vert)) {
			std::cerr<<"Error loading vertex shader!"<<std::endl;
			return false;
		}
				
		// Try to make the fragment shader
		if (!factory->MakeShaderSource(fragmentName.c_str(), frag)) {
			std::cerr<<"Error loading fragment shader!"<<std::endl;
			return false;
		}
		
		// Compile and Link
        if (!shader.Compile(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile shader: Unknown error\n");
            return false;
        }
        
        if (!shader.Link()) {
			std::cerr<<"Error linking!"<<std::endl;
			return false;
		}       
		
	} catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        std::cerr<<"could not compile shader "
				 <<vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction())
				 <<": "
				 <<ce.GetMsgA()<<std::endl;
				 
		return false;
	}
	
	return true;
}


bool megamol::core::utility::InitializeShader(megamol::core::utility::ShaderSourceFactory* factory, vislib::graphics::gl::GLSLGeometryShader& shader, const std::string& vertexName, const std::string& fragmentName, const std::string& geometryName)
{
	vislib::graphics::gl::ShaderSource vert, frag, geom;

    if (!vislib::graphics::gl::GLSLGeometryShader::InitialiseExtensions()) {
        std::cerr<<"Failed to init OpenGL extensions: GLSLGeometryShader"<<std::endl;
        return false;
    }

	// Initialize our shader
	try {
		// Try to make the vertex shader
		if (!factory->MakeShaderSource(vertexName.c_str(), vert)) {
			std::cerr<<"Error loading vertex shader!"<<std::endl;
			return false;
		}
		
		// Try to make the geometry shader
		if (!factory->MakeShaderSource(geometryName.c_str(), geom)) {
			std::cerr<<"Error loading geometry shader!"<<std::endl;
			return false;
		}
		
		// Try to make the fragment shader
		if (!factory->MakeShaderSource(fragmentName.c_str(), frag)) {
			std::cerr<<"Error loading fragment shader!"<<std::endl;
			return false;
		}
		
		// Compile and Link
        if (!shader.Compile(vert.Code(), vert.Count(), geom.Code(), geom.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile shader: Unknown error\n");
            return false;
        }
        
        if (!shader.Link()) {
			std::cerr<<"Error linking!"<<std::endl;
			return false;
		}       
		
	} catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        std::cerr<<"could not compile shader "
				 <<vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction())
				 <<": "
				 <<ce.GetMsgA()<<std::endl;
				 
		return false;
	}
	
	return true;
}


void megamol::core::utility::AddShaderSnippet(megamol::core::utility::ShaderSourceFactory* factory, vislib::graphics::gl::ShaderSource& source, const std::string& snippetName, bool containsCode)
{
	if (!containsCode) {
		source.Append(factory->MakeShaderSnippet(snippetName.c_str()));	
	} else {
		vislib::graphics::gl::ShaderSource::StringSnippet* newSnippet = new vislib::graphics::gl::ShaderSource::StringSnippet(snippetName.c_str());
		source.Append(newSnippet);		
	}
}

