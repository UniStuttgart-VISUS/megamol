#include "stdafx.h"

#include "mmcore/utility/MDAOVolumeGenerator.h"
#include "mmcore/utility/MDAOShaderUtilities.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <sstream>

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include <GL/glu.h>


using namespace megamol::core::utility;

#define checkGLError { GLenum errCode = glGetError(); if (errCode != GL_NO_ERROR) std::cout<<"Error in line "<<__LINE__<<": "<<gluErrorString(errCode)<<std::endl;}

MDAOVolumeGenerator::MDAOVolumeGenerator(): fboHandle(0), volumeHandle(0), factory(nullptr), dataVersion(0)
{
	clearBuffer = nullptr;
}


MDAOVolumeGenerator::~MDAOVolumeGenerator()
{
	glDeleteTextures(1, &volumeHandle);
	volumeShader.Release();
	mipmapShader.Release();
	
	if (clearBuffer != nullptr)
		delete [] clearBuffer;
}


void MDAOVolumeGenerator::SetShaderSourceFactory(megamol::core::utility::ShaderSourceFactory* factory)
{
	this->factory = factory;
}


GLuint MDAOVolumeGenerator::GetVolumeTextureHandle()
{
	return volumeHandle;
}




bool MDAOVolumeGenerator::Init()
{
	// Generate and initialize the volume texture
	glGenTextures(1, &this->volumeHandle);
	glBindTexture(GL_TEXTURE_3D, this->volumeHandle);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER); 

	// Generate and initialize the frame buffer
	glGenFramebuffers(1, &fboHandle);
	
	GLint prevFBO;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING, &prevFBO);

	glBindFramebuffer(GL_FRAMEBUFFER, fboHandle);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, this->volumeHandle, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, prevFBO);

	// Check if we can use modern features
	computeAvailable = (::isExtAvailable("GL_ARB_compute_shader") == GL_TRUE);
	clearAvailable = (::isExtAvailable("GL_ARB_clear_texture") == GL_TRUE);
	
	std::stringstream outmsg;
	outmsg<<"Voxelization Features enabled: Compute Shader "<<computeAvailable<<", Clear Texture "<<clearAvailable<<std::endl;
	vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, outmsg.str().c_str());
	
	if (computeAvailable) {
		// Try to initialize the compute shader
		vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet> mipmapSrc;
		mipmapSrc = factory->MakeShaderSnippet("sphere_mdao_mipmap::Compute");
		try {
			mipmapShader.Compile(mipmapSrc->PeekCode());
			mipmapShader.Link();
		} catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException &ce) {
			outmsg.str("");
			outmsg<<"Could not compile volume mipmapping shader "
						<<vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction())
						<<": "
						<<ce.GetMsgA()<<std::endl;
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, outmsg.str().c_str());
			return false;
		}	
	}
	
	return InitializeShader(this->factory, this->volumeShader, "sphere_mdao_volume::vertex", "sphere_mdao_volume::fragment", "sphere_mdao_volume::geometry");
}



void MDAOVolumeGenerator::SetResolution(float resX, float resY, float resZ)
{
	if (volumeRes.Width() == resX && volumeRes.Height() == resY && volumeRes.Depth() == resZ)
		return;

	if (!clearAvailable) {
		if (clearBuffer != nullptr)
			delete [] clearBuffer;
		clearBuffer = new unsigned char[static_cast<unsigned int>(ceil(resX*resY*resZ))];
		memset(clearBuffer, 0, static_cast<unsigned int>(resX*resY*resZ));
	}
	
	volumeRes.Set(resX, resY, resZ);
	glBindTexture(GL_TEXTURE_3D, this->volumeHandle);checkGLError;
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, static_cast<GLsizei>(resX), static_cast<GLsizei>(resY), static_cast<GLsizei>(resZ), 0, GL_RED, GL_UNSIGNED_BYTE, nullptr); checkGLError
	glGenerateMipmap(GL_TEXTURE_3D); 
}


void MDAOVolumeGenerator::StartInsertion(const vislib::math::Cuboid< float >& obb, const vislib::math::Vector<float, 4> &clipDat)
{
	this->clipDat = clipDat;
	
	this->boundsSizeInverse = obb.GetSize();
	for (int i=0; i<3; ++i) {this->boundsSizeInverse[i] = 1.0f / this->boundsSizeInverse[i];}
	this->boundsMin = obb.GetLeftBottomBack();

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);checkGLError;
	glEnable(GL_BLEND);checkGLError;
	glBlendFunc(GL_ONE, GL_ONE);checkGLError;
	
	// Store old viewport
    glGetIntegerv( GL_VIEWPORT, viewport);
    glGetIntegerv( GL_FRAMEBUFFER_BINDING, &prevFBO);
	
	glBindTexture(GL_TEXTURE_3D, this->volumeHandle);checkGLError;
	glBindFramebuffer(GL_FRAMEBUFFER, this->fboHandle);checkGLError;
	
	// Set Viewport size for one slice
	glViewport(0, 0, static_cast<GLsizei>(volumeRes.GetWidth()), static_cast<GLsizei>(volumeRes.GetHeight()));
	
	volumeShader.Enable();checkGLError;
	
	volumeShader.SetParameterArray3("inBoundsMin", 1, this->boundsMin.PeekCoordinates());
	volumeShader.SetParameterArray3("inBoundsSizeInverse", 1, this->boundsSizeInverse.PeekDimension());
	volumeShader.SetParameterArray3("inVolumeSize", 1, volumeRes.PeekDimension());
	
	volumeShader.SetParameterArray4("inClipDat", 1, clipDat.PeekComponents());
}



void MDAOVolumeGenerator::ClearVolume()
{
	if (clearAvailable) {
		unsigned char clearData = 0;
		glClearTexImage(this->volumeHandle, 0, GL_RED, GL_UNSIGNED_BYTE, &clearData);checkGLError;
		return;
	} 
	
	if (clearBuffer == nullptr)
		return;

	glBindTexture(GL_TEXTURE_3D, this->volumeHandle);checkGLError;
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, static_cast<GLsizei>(volumeRes.Width()), static_cast<GLsizei>(volumeRes.Height()), static_cast<GLsizei>(volumeRes.Depth()), 0, GL_RED, GL_UNSIGNED_BYTE, clearBuffer); checkGLError;
	glBindTexture(GL_TEXTURE_3D, 0);checkGLError;
}




void MDAOVolumeGenerator::EndInsertion()
{
	volumeShader.Disable(); checkGLError;

	glBindFramebuffer(GL_FRAMEBUFFER, prevFBO);
	glDisable(GL_BLEND);

	glViewport( viewport[0], viewport[1], viewport[2], viewport[3]);
	
	++dataVersion;

/*	glBindTexture(GL_TEXTURE_3D, this->volumeHandle);
	std::vector<char> rawdata(volumeRes.Width() * volumeRes.Height() * volumeRes.Depth());

	glGetTexImage(GL_TEXTURE_3D, 0, GL_RED, GL_UNSIGNED_BYTE, rawdata.data());
	
	std::ofstream outfile("volume.raw", std::ios::binary);
	outfile.write(rawdata.data(), rawdata.size());
	outfile.close();

	std::cout<<"Wrote volume: "<<volumeRes.Width()<<" x "<<volumeRes.Height()<<" x "<<volumeRes.Depth()<<std::endl;
	
	exit(123);    */
}




void MDAOVolumeGenerator::InsertParticles(unsigned int count, float globalRadius, GLuint vertexArray)
{
	volumeShader.SetParameter("inGlobalRadius", globalRadius);checkGLError;

	glBindVertexArray(vertexArray);checkGLError;
	glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(count));checkGLError;
}


void MDAOVolumeGenerator::RecreateMipmap() {
	
	if (!computeAvailable) {
		glBindTexture(GL_TEXTURE_3D, this->volumeHandle);
		glGenerateMipmap(GL_TEXTURE_3D);
		glBindTexture(GL_TEXTURE_3D, 0);
		return;
	}
	
	// int res = volumeRes.Width();
	vislib::math::Dimension<float, 3> res = volumeRes;
	int level = 0;

	mipmapShader.Enable();
	while (res.Width()>=1.0f || res.Height()>=1.0f || res.Depth() >= 1.0f) {
		res.Scale(0.5f);
		// Bind input image
		glBindImageTexture(0, this->volumeHandle, level, GL_TRUE, 0, GL_READ_ONLY, GL_R8);checkGLError;
		// Bind output image
		glBindImageTexture(1, this->volumeHandle, level+1, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);checkGLError;

		mipmapShader.SetParameter("inputImage", 0);
		mipmapShader.SetParameter("outputImage", 1);
		glDispatchCompute(static_cast<GLuint>(ceil(res.Width()/64.0f)), static_cast<GLuint>(ceil(res.Height())), static_cast<GLuint>(ceil(res.Depth())));
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		level++;
	};
	mipmapShader.Disable();
}




const vislib::math::Dimension< float, 3 >& MDAOVolumeGenerator::GetExtents()
{
	return volumeRes;
}



unsigned int MDAOVolumeGenerator::GetDataVersion()
{
	return dataVersion;
}


