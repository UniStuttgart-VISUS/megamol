/*
* RaycastVolumeRenderer.h
*
* Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "RaycastVolumeRenderer.h"

#include <array>

#include "vislib/graphics/gl/ShaderSource.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "linmath.h"

using namespace megamol::stdplugin::volume;

RaycastVolumeRenderer::RaycastVolumeRenderer()
	: Renderer3DModule(),
	m_volumetricData_callerSlot("getData", "Connects the volume renderer with a voluemtric data source"),
	m_transferFunction_callerSlot("getTranfserFunction","Connects the volume renderer with a transfer function"),
	m_ray_step_ratio_param("ray step ratio", "")
{
	this->m_volumetricData_callerSlot.SetCompatibleCall<megamol::core::misc::VolumetricDataCallDescription>();
	this->MakeSlotAvailable(&this->m_volumetricData_callerSlot);

	this->m_transferFunction_callerSlot.SetCompatibleCall<megamol::core::view::CallGetTransferFunctionDescription>();
	this->MakeSlotAvailable(&this->m_transferFunction_callerSlot);

	auto* ep = new megamol::core::param::FloatParam(1.0);
	this->m_ray_step_ratio_param << ep;
	this->MakeSlotAvailable(&this->m_ray_step_ratio_param);
}

RaycastVolumeRenderer::~RaycastVolumeRenderer()
{
	this->Release();
}

bool RaycastVolumeRenderer::create()
{
	try
	{
		// create shader program
		m_raycast_volume_compute_shdr = std::make_unique<vislib::graphics::gl::GLSLComputeShader>();
		m_render_to_framebuffer_shdr = std::make_unique<vislib::graphics::gl::GLSLShader>();

		vislib::graphics::gl::ShaderSource compute_shader_src;
		vislib::graphics::gl::ShaderSource vertex_shader_src;
		vislib::graphics::gl::ShaderSource fragment_shader_src;

		if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::compute", compute_shader_src)) return false;
		if (!m_raycast_volume_compute_shdr->Compile(compute_shader_src.Code(), compute_shader_src.Count())) return false;
		if (!m_raycast_volume_compute_shdr->Link()) return false;

		if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::vert", vertex_shader_src)) return false;
		if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::frag", fragment_shader_src)) return false;
		if (!m_render_to_framebuffer_shdr->Compile(vertex_shader_src.Code(), vertex_shader_src.Count(), fragment_shader_src.Code(), fragment_shader_src.Count())) return false;
		if (!m_render_to_framebuffer_shdr->Link()) return false;
	}
	catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile shader (@%s): %s\n",
			vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
				ce.FailedAction()), ce.GetMsgA());
                return false;
	}
	catch (vislib::Exception e) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile shader: %s\n", e.GetMsgA());
                return false;
	}
	catch (...) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile shader: Unknown exception\n");
                return false;
	}
		

	// create render target texture
	TextureLayout render_tgt_layout(
		GL_RGBA8,
		1920,
		1080,
		1,
		GL_RGBA,
		GL_UNSIGNED_BYTE,
		1,
		{	{ GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER },
			{ GL_TEXTURE_WRAP_T,GL_CLAMP_TO_BORDER },
			{ GL_TEXTURE_WRAP_R,GL_CLAMP_TO_BORDER },
			{ GL_TEXTURE_MIN_FILTER,GL_LINEAR },
			{ GL_TEXTURE_MAG_FILTER,GL_LINEAR } },
		{} );
	m_render_target = std::make_unique<Texture2D>("raycast_volume_render_target", render_tgt_layout, nullptr);


	// create empty volume texture
	TextureLayout volume_layout(
		GL_R32F,
		1,
		1,
		1,
		GL_RED,
		GL_FLOAT,
		1,
		{ { GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER },
		{ GL_TEXTURE_WRAP_T,GL_CLAMP_TO_BORDER },
		{ GL_TEXTURE_MIN_FILTER,GL_LINEAR },
		{ GL_TEXTURE_MAG_FILTER,GL_LINEAR } },
		{});
	m_volume_texture = std::make_unique<Texture3D>("raycast_volume_texture",volume_layout,nullptr);

	// create empty transfer function texture
	TextureLayout tf(
		GL_RGBA8,
		1,
		1,
		1,
		GL_RGBA,
		GL_UNSIGNED_BYTE,
		1,
		{ { GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER },
		{ GL_TEXTURE_WRAP_T,GL_CLAMP_TO_BORDER },
		{ GL_TEXTURE_MIN_FILTER,GL_LINEAR },
		{ GL_TEXTURE_MAG_FILTER,GL_LINEAR } },
		{});
	m_transfer_function = std::make_unique<Texture2D>("raycast_volume_texture", tf, nullptr);

	return true;
}

void RaycastVolumeRenderer::release()
{
	m_raycast_volume_compute_shdr.reset(nullptr);
	m_render_target.reset(nullptr);
}

bool RaycastVolumeRenderer::GetExtents(megamol::core::Call& call)
{
	auto cr = dynamic_cast<core::view::CallRender3D*>(&call);
	auto cd = m_volumetricData_callerSlot.CallAs<megamol::core::misc::VolumetricDataCall>();

	if (cr == NULL) return false;
	if (cd == nullptr) return false;

	// TODO Do something about time/framecount ?

	if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;
	if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) return false;

	cr->SetTimeFramesCount(cd->FrameCount());
	cr->AccessBoundingBoxes() = cd->GetBoundingBoxes();
	cr->AccessBoundingBoxes().MakeScaledWorld(1.0f);

	return true;
}

bool RaycastVolumeRenderer::Render(megamol::core::Call& call)
{
	megamol::core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
	if (cr == NULL) return false;

	// manual creation of projection and view matrix
	GLfloat fovy = (cr->GetCameraParameters()->ApertureAngle() / 180.0f) * 3.14f;
	GLfloat near_clip = cr->GetCameraParameters()->NearClip();
	GLfloat far_clip = cr->GetCameraParameters()->FarClip();
	GLfloat f = 1.0f / std::tan(fovy / 2.0f);
	GLfloat nf = 1.0f / (near_clip - far_clip);
	GLfloat aspect_ratio = static_cast<GLfloat>(cr->GetViewport().AspectRatio());

	mat4x4 proj_mx;
	mat4x4_perspective(proj_mx, fovy, aspect_ratio, near_clip, far_clip);
	//std::array<GLfloat, 16> projection_matrix;
	//projection_matrix[0] = f / aspect_ratio;
	//projection_matrix[1] = 0.0f;
	//projection_matrix[2] = 0.0f;
	//projection_matrix[3] = 0.0f;
	//projection_matrix[4] = 0.0f;
	//projection_matrix[5] = f;
	//projection_matrix[6] = 0.0f;
	//projection_matrix[7] = 0.0f;
	//projection_matrix[8] = 0.0f;
	//projection_matrix[9] = 0.0f;
	//projection_matrix[10] = (far_clip + near_clip) * nf;
	//projection_matrix[11] = -1.0f;
	//projection_matrix[12] = 0.0f;
	//projection_matrix[13] = 0.0f;
	//projection_matrix[14] = (2.0f * far_clip * near_clip) * nf;
	//projection_matrix[15] = 0.0f;

	auto cam_right = cr->GetCameraParameters()->Right();
	auto cam_up = cr->GetCameraParameters()->Up();
	auto cam_front = -cr->GetCameraParameters()->Front();
	auto cam_position = cr->GetCameraParameters()->Position();

	mat4x4 view_mx;
	//std::array<GLfloat, 16> view_matrix;
	//view_matrix[0] = cam_right.X();
	//view_matrix[1] = cam_up.X();
	//view_matrix[2] = cam_front.X();
	//view_matrix[3] = 0.0f;
	//
	//view_matrix[4] = cam_right.Y();
	//view_matrix[5] = cam_up.Y();
	//view_matrix[6] = cam_front.Y();
	//view_matrix[7] = 0.0f;
	//
	//view_matrix[8] = cam_right.Z();
	//view_matrix[9] = cam_up.Z();
	//view_matrix[10] = cam_front.Z();
	//view_matrix[11] = 0.0f;
	//
	//view_matrix[12] = -(cam_position.X()*cam_right.X() + cam_position.Y()*cam_right.Y() + cam_position.Z()*cam_right.Z());
	//view_matrix[13] = -(cam_position.X()*cam_up.X() + cam_position.Y()*cam_up.Y() + cam_position.Z()*cam_up.Z());
	//view_matrix[14] = -(cam_position.X()*cam_front.X() + cam_position.Y()*cam_front.Y() + cam_position.Z()*cam_front.Z());
	//view_matrix[15] = 1.0f;
	view_mx[0][0] = cam_right.X();
	view_mx[0][1] = cam_up.X();
	view_mx[0][2] = cam_front.X();
	view_mx[0][3] = 0.0f;
	
	view_mx[1][0] = cam_right.Y();
	view_mx[1][1] = cam_up.Y();
	view_mx[1][2] = cam_front.Y();
	view_mx[1][3] = 0.0f;
	
	view_mx[2][0] = cam_right.Z();
	view_mx[2][1] = cam_up.Z();
	view_mx[2][2] = cam_front.Z();
	view_mx[2][3] = 0.0f;
	
	view_mx[3][0] = -(cam_position.X()*cam_right.X() + cam_position.Y()*cam_right.Y() + cam_position.Z()*cam_right.Z());
	view_mx[3][1] = -(cam_position.X()*cam_up.X() + cam_position.Y()*cam_up.Y() + cam_position.Z()*cam_up.Z());
	view_mx[3][2] = -(cam_position.X()*cam_front.X() + cam_position.Y()*cam_front.Y() + cam_position.Z()*cam_front.Z());
	view_mx[3][3] = 1.0f;

	if (!updateVolumeData()) return false;
	if (!updateTransferFunction()) return false;
	
	// enable raycast volume rendering program
	m_raycast_volume_compute_shdr->Enable();

	//TODO set uniform values
	mat4x4 volume_model_mx;
	mat4x4_identity(volume_model_mx);
	
	mat4x4 view_proj_mx;
	mat4x4 inv_view_proj_mx;
	mat4x4_mul(view_proj_mx, proj_mx, view_mx);
	mat4x4_invert(inv_view_proj_mx, view_proj_mx);
	glUniformMatrix4fv(m_raycast_volume_compute_shdr->ParameterLocation("camera_inv_view_proj_mx"), 1, GL_FALSE, reinterpret_cast<float*>(&inv_view_proj_mx));

	glUniformMatrix4fv(m_raycast_volume_compute_shdr->ParameterLocation("view_mx"), 1, GL_FALSE, reinterpret_cast<float*>(&view_mx));
	glUniformMatrix4fv(m_raycast_volume_compute_shdr->ParameterLocation("proj_mx"), 1, GL_FALSE, reinterpret_cast<float*>(&proj_mx));

	vec2 rt_resolution;
	rt_resolution[0] = static_cast<float>(m_render_target->getWidth());
	rt_resolution[1] = static_cast<float>(m_render_target->getHeight());
	glUniform2fv(m_raycast_volume_compute_shdr->ParameterLocation("rt_resolution"), 1, rt_resolution);

	// bbox sizes
	vec3 box_min;
	box_min[0] = m_volume_origin[0];
	box_min[1] = m_volume_origin[1];
	box_min[2] = m_volume_origin[2];
	vec3 box_max;
	box_max[0] = m_volume_origin[0] + m_volume_extents[0];
	box_max[1] = m_volume_origin[1] + m_volume_extents[1];
	box_max[2] = m_volume_origin[2] + m_volume_extents[2];
	glUniform3fv(m_raycast_volume_compute_shdr->ParameterLocation("boxMin"), 1, box_min);
	glUniform3fv(m_raycast_volume_compute_shdr->ParameterLocation("boxMax"), 1, box_max);

    glUniform3f(m_raycast_volume_compute_shdr->ParameterLocation("halfVoxelSize"), 1.0f / (2.0f * (m_volume_resolution[0] - 1))
        , 1.0f / (2.0f * (m_volume_resolution[1] - 1)), 1.0f / (2.0f * (m_volume_resolution[2] - 1)));
	glUniform1f(m_raycast_volume_compute_shdr->ParameterLocation("voxelSize"), 1.0);
	glUniform1f(m_raycast_volume_compute_shdr->ParameterLocation("rayStepRatio"), this->m_ray_step_ratio_param.Param<core::param::FloatParam>()->Value());
	glUniform1f(m_raycast_volume_compute_shdr->ParameterLocation("opacityThreshold"), 1.0);

	// bind volume texture
	glActiveTexture(GL_TEXTURE0);
	m_volume_texture->bindTexture();
	glUniform1i(m_raycast_volume_compute_shdr->ParameterLocation("volume_tx3D"), 0);
	// bind the transfer function
	glActiveTexture(GL_TEXTURE1);
	m_transfer_function->bindTexture();
	glUniform1i(m_raycast_volume_compute_shdr->ParameterLocation("transfer_function_tx2D"), 1);

	// bind image texture
	m_render_target->bindImage(0, GL_WRITE_ONLY);

	// dispatch compute
	m_raycast_volume_compute_shdr->Dispatch(
		static_cast<int>(std::ceil(rt_resolution[0] / 8.0f)),
		static_cast<int>(std::ceil(rt_resolution[1] / 8.0f)),
		1);

	m_raycast_volume_compute_shdr->Disable();

	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

	////////
	// copy image to framebuffer
	///////
	//TODO query gl state and reset to previous state?
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	m_render_to_framebuffer_shdr->Enable();

	glActiveTexture(GL_TEXTURE1);
	m_render_target->bindTexture();
	glUniform1i(m_render_to_framebuffer_shdr->ParameterLocation("src_tx2D"), 1);

	glDrawArrays(GL_TRIANGLES, 0, 6);

	m_render_to_framebuffer_shdr->Disable();


	glUseProgram(0);
	//glBindVertexArray(0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	return true;
}

bool RaycastVolumeRenderer::updateVolumeData()
{
	auto* cd = this->m_volumetricData_callerSlot.CallAs<megamol::core::misc::VolumetricDataCall>();

	if (cd == nullptr)
		return false;

	// TODO check time and frame id or whatever else
	if (this->m_volume_datahash != cd->DataHash() )
	{
		this->m_volume_datahash = cd->DataHash();
	}
	else {
		return true;
	}

	if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;
	if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) return false;
	if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_DATA)) return false;

	auto const metadata = cd->GetMetadata();

	if (!metadata->GridType == core::misc::CARTESIAN) {
		vislib::sys::Log::DefaultLog.WriteError("RaycastVolumeRenderer only works with cartesian grids (for now)");
		return false;
	}

	m_volume_origin[0] = metadata->Origin[0];
	m_volume_origin[1] = metadata->Origin[1];
	m_volume_origin[2] = metadata->Origin[2];
	m_volume_extents[0] = metadata->Extents[0];
	m_volume_extents[1] = metadata->Extents[1];
	m_volume_extents[2] = metadata->Extents[2];
    m_volume_resolution[0] = metadata->Resolution[0];
    m_volume_resolution[1] = metadata->Resolution[1];
    m_volume_resolution[2] = metadata->Resolution[2];

	GLenum internal_format;
	GLenum format;
	GLenum type;

	switch (metadata->ScalarType) {
	case core::misc::FLOATING_POINT:
		if (metadata->ScalarLength == 4) {
			internal_format = GL_R32F;
			format = GL_RED;
			type = GL_FLOAT;
		}
		else {
			vislib::sys::Log::DefaultLog.WriteError("Floating point values with a length != 4 byte are invalid.");
			return false;
		}
		break;
	case core::misc::UNSIGNED_INTEGER:
		if (metadata->ScalarLength == 1) {
			internal_format = GL_R8;
			format = GL_RED;
			type = GL_UNSIGNED_BYTE;
		}
		else if (metadata->ScalarLength == 2) {
			internal_format = GL_R16UI;
			format = GL_RED;
			type = GL_UNSIGNED_SHORT;
		}
		else {
			vislib::sys::Log::DefaultLog.WriteError("Unsigned integers with a length greater than 2 are invalid.");
			return false;
		}
		break;
	case core::misc::SIGNED_INTEGER:
		if (metadata->ScalarLength == 2) {
			internal_format = GL_R16I;
			format = GL_RED;
			type = GL_SHORT;
		}
		else {
			vislib::sys::Log::DefaultLog.WriteError("Integers with a length != 2 are invalid.");
			return false;
		}
		break;
	case core::misc::BITS:
		vislib::sys::Log::DefaultLog.WriteError("Invalid datatype.");
		return false;
		break;
	}

	auto const volumedata = cd->GetData();

	//TODO if/else data already on GPU

	// debug using dummy-data
	std::array<uint8_t, 8> debug_volume_data = { 255,0,255,0,255,255,0,0 };
	TextureLayout debug_volume_layout(
		GL_R8,
		2,
		2,
		2,
		GL_RED,
		GL_UNSIGNED_BYTE,
		1,
		{ { GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER },
		{ GL_TEXTURE_WRAP_T,GL_CLAMP_TO_BORDER },
		{ GL_TEXTURE_WRAP_R,GL_CLAMP_TO_BORDER },
		{ GL_TEXTURE_MIN_FILTER,GL_NEAREST },
		{ GL_TEXTURE_MAG_FILTER,GL_NEAREST } },
		{});

	TextureLayout volume_layout(
		internal_format,
		metadata->Resolution[0],
		metadata->Resolution[1],
		metadata->Resolution[2],
		format,
		type,
		1,
		{ { GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER },
		{ GL_TEXTURE_WRAP_T,GL_CLAMP_TO_BORDER },
		{ GL_TEXTURE_WRAP_R,GL_CLAMP_TO_BORDER },
		{ GL_TEXTURE_MIN_FILTER,GL_LINEAR },
		{ GL_TEXTURE_MAG_FILTER,GL_LINEAR } },
		{});

	m_volume_texture->reload(volume_layout, volumedata);

}

bool RaycastVolumeRenderer::updateTransferFunction()
{
	core::view::CallGetTransferFunction *ct = this->m_transferFunction_callerSlot.CallAs<core::view::CallGetTransferFunction>();
	if (ct != NULL && ((*ct)())) {
		float const* tf_tex = ct->GetTextureData();
		unsigned int tf_size = ct->TextureSize();
		auto tf_format = ct->OpenGLTextureFormat();

		GLenum internal_format = tf_format == GL_RGB ? GL_RGB32F : GL_RGBA32F;
		GLenum format = tf_format;
		GLenum type = GL_FLOAT;

		TextureLayout tf_layout(
			internal_format,
			tf_size,
			1,
			1,
			format,
			type,
			1,
			{ { GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER },
			{ GL_TEXTURE_WRAP_T,GL_CLAMP_TO_BORDER },
			{ GL_TEXTURE_MIN_FILTER,GL_LINEAR },
			{ GL_TEXTURE_MAG_FILTER,GL_LINEAR } },
			{});

		m_transfer_function->reload(tf_layout,tf_tex);
		//this->colorTransferGray(allColor, tf_tex, tex_size, processedColor, 3);
	}
	else {
		//this->colorTransferGray(allColor, NULL, 0, processedColor, 3);
	}

	return true;
}
