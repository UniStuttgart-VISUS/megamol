/*
 * VolumeSliceRenderer.cpp
 *
 * Copyright (C) 2012-2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "VolumeSliceRenderer.h"

#include "mmcore/Call.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/math/Plane.h"
#include "vislib/math/Point.h"
#include "vislib/sys/Log.h"

#include "glowl/Texture.hpp"
#include "glowl/Texture2D.hpp"
#include "glowl/Texture3D.hpp"

#include <array>
#include <cmath>

/*
 * VolumeSliceRenderer::VolumeSliceRenderer
 */
megamol::stdplugin::volume::VolumeSliceRenderer::VolumeSliceRenderer(void)
	: Renderer3DModule_2()
	, getVolSlot("getVol", "The call for data")
	, getTFSlot("gettransferfunction", "The call for Transfer function")
	, getClipPlaneSlot("getclipplane", "The call for clipping plane") {

    this->getVolSlot.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->getVolSlot);

    this->getTFSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getClipPlaneSlot.SetCompatibleCall<core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);
}


/*
 * VolumeSliceRenderer::VolumeSliceRenderer
 */
megamol::stdplugin::volume::VolumeSliceRenderer::~VolumeSliceRenderer(void) {
    this->Release();
}


/*
 * VolumeSliceRenderer::VolumeSliceRenderer
 */
bool megamol::stdplugin::volume::VolumeSliceRenderer::create(void) {
	try {
		// create shader program
		vislib::graphics::gl::ShaderSource compute_shader_src;
		vislib::graphics::gl::ShaderSource vertex_shader_src;
		vislib::graphics::gl::ShaderSource fragment_shader_src;

		if (!instance()->ShaderSourceFactory().MakeShaderSource("VolumeSliceRenderer::compute", compute_shader_src))
			return false;
		if (!this->compute_shader.Compile(compute_shader_src.Code(), compute_shader_src.Count()))
			return false;
		if (!this->compute_shader.Link()) return false;

		if (!instance()->ShaderSourceFactory().MakeShaderSource("VolumeSliceRenderer::vert", vertex_shader_src))
			return false;
		if (!instance()->ShaderSourceFactory().MakeShaderSource("VolumeSliceRenderer::frag", fragment_shader_src))
			return false;
		if (!this->render_shader.Compile(vertex_shader_src.Code(), vertex_shader_src.Count(),
			fragment_shader_src.Code(), fragment_shader_src.Count()))
			return false;
		if (!this->render_shader.Link()) return false;
	}
	catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to compile shader (@%s): %s\n",
			vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
			ce.GetMsgA());
		return false;
	}
	catch (vislib::Exception e) {
		vislib::sys::Log::DefaultLog.WriteMsg(
			vislib::sys::Log::LEVEL_ERROR, "Unable to compile shader: %s\n", e.GetMsgA());
		return false;
	}
	catch (...) {
		vislib::sys::Log::DefaultLog.WriteMsg(
			vislib::sys::Log::LEVEL_ERROR, "Unable to compile shader: Unknown exception\n");
		return false;
	}

    return true;
}


/*
 * VolumeSliceRenderer::VolumeSliceRenderer
 */
bool megamol::stdplugin::volume::VolumeSliceRenderer::GetExtents(core::view::CallRender3D_2& cr) {
    auto *vdc = this->getVolSlot.CallAs<core::misc::VolumetricDataCall>();

	vdc->SetFrameID(static_cast<unsigned int>(cr.Time()));

	if (vdc == nullptr || !(*vdc)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;
	if (vdc == nullptr || !(*vdc)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) return false;

    cr.SetTimeFramesCount(vdc->FrameCount());
    cr.AccessBoundingBoxes() = vdc->AccessBoundingBoxes();

    return true;
}


/*
 * VolumeSliceRenderer::VolumeSliceRenderer
 */
void megamol::stdplugin::volume::VolumeSliceRenderer::release(void) {
}


/*
 * VolumeSliceRenderer::VolumeSliceRenderer
 */
bool megamol::stdplugin::volume::VolumeSliceRenderer::Render(core::view::CallRender3D_2& cr) {
    // get volume data
    auto *vdc = this->getVolSlot.CallAs<core::misc::VolumetricDataCall>();
	if (vdc == nullptr || !(*vdc)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;
	if (vdc == nullptr || !(*vdc)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) return false;
    if (vdc == nullptr || !(*vdc)(core::misc::VolumetricDataCall::IDX_GET_DATA)) return false;

	auto const metadata = vdc->GetMetadata();

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

	glowl::TextureLayout volume_layout(internal_format, metadata->Resolution[0], metadata->Resolution[1],
        metadata->Resolution[2], format, type, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    glowl::Texture3D vol_tex("volume_texture", volume_layout, vdc->GetData());

	// get clip plane
	core::view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<core::view::CallClipPlane>();
	if (ccp == nullptr || !(*ccp)()) return false;

	const auto slice = ccp->GetPlane();

	// get transfer function
	core::view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<core::view::CallGetTransferFunction>();
	if (cgtf == nullptr || !(*cgtf)()) return false;

	// get camera
	core::view::Camera_2 cam;
	cr.GetCamera(cam);

	cam_type::matrix_type view, proj;
	cam.calc_matrices(view, proj);

	// create render target
	glowl::TextureLayout render_tgt_layout(GL_RGBA8, cr.GetViewport().Width(), cr.GetViewport().Height(),
		1, GL_RGBA, GL_UNSIGNED_BYTE, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
	glowl::Texture2D render_target("render_target", render_tgt_layout, nullptr);

	// compute slice
	this->compute_shader.Enable();

	glUniformMatrix4fv(
		this->compute_shader.ParameterLocation("view_mx"), 1, GL_FALSE, glm::value_ptr(static_cast<glm::mat4>(view)));
	glUniformMatrix4fv(this->compute_shader.ParameterLocation("proj_mx"), 1, GL_FALSE, glm::value_ptr(static_cast<glm::mat4>(proj)));

	std::array<float, 2> rt_resolution;
	rt_resolution[0] = static_cast<float>(render_target.getWidth());
	rt_resolution[1] = static_cast<float>(render_target.getHeight());
	glUniform2fv(this->compute_shader.ParameterLocation("rt_resolution"), 1, rt_resolution.data());
	
	glm::vec3 box_min;
	box_min[0] = vdc->GetBoundingBoxes().ObjectSpaceBBox().Left();
	box_min[1] = vdc->GetBoundingBoxes().ObjectSpaceBBox().Bottom();
	box_min[2] = vdc->GetBoundingBoxes().ObjectSpaceBBox().Back();
	glm::vec3 box_max;
	box_max[0] = vdc->GetBoundingBoxes().ObjectSpaceBBox().Right();
	box_max[1] = vdc->GetBoundingBoxes().ObjectSpaceBBox().Top();
	box_max[2] = vdc->GetBoundingBoxes().ObjectSpaceBBox().Front();
	glUniform3fv(this->compute_shader.ParameterLocation("boxMin"), 1, glm::value_ptr(box_min));
	glUniform3fv(this->compute_shader.ParameterLocation("boxMax"), 1, glm::value_ptr(box_max));

	std::array<float, 2> valueRange;
	valueRange[0] = static_cast<float>(vdc->GetMetadata()->MinValues[0]);
	valueRange[1] = static_cast<float>(vdc->GetMetadata()->MaxValues[0]);
	glUniform2fv(this->compute_shader.ParameterLocation("valRange"), 1, valueRange.data());

	std::array<float, 4> plane;
	plane[0] = slice.GetA();
	plane[1] = slice.GetB();
	plane[2] = slice.GetC();
	plane[3] = std::abs(slice.GetD());
	glUniform4fv(this->compute_shader.ParameterLocation("slice"), 1, plane.data());

	glActiveTexture(GL_TEXTURE0);
	vol_tex.bindTexture();
	glUniform1i(this->compute_shader.ParameterLocation("volume_tx3D"), 0);

	cgtf->BindConvenience(this->compute_shader, GL_TEXTURE1, 1);

	render_target.bindImage(0, GL_WRITE_ONLY);

	this->compute_shader.Dispatch(
		static_cast<int>(std::ceil(rt_resolution[0] / 8.0f)), static_cast<int>(std::ceil(rt_resolution[1] / 8.0f)), 1);

	glBindImageTexture(0, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R);

	cgtf->UnbindConvenience();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, 0);

	this->compute_shader.Disable();

	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
	
	// render
	bool state_depth_test = glIsEnabled(GL_DEPTH_TEST);
	bool state_blend = glIsEnabled(GL_BLEND);

	GLint state_blend_src_rgb, state_blend_src_alpha, state_blend_dst_rgb, state_blend_dst_alpha;
	glGetIntegerv(GL_BLEND_SRC_RGB, &state_blend_src_rgb);
	glGetIntegerv(GL_BLEND_SRC_ALPHA, &state_blend_src_alpha);
	glGetIntegerv(GL_BLEND_DST_RGB, &state_blend_dst_rgb);
	glGetIntegerv(GL_BLEND_DST_ALPHA, &state_blend_dst_alpha);

	if (state_depth_test) glDisable(GL_DEPTH_TEST);
	if (!state_blend) glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	this->render_shader.Enable();

	glActiveTexture(GL_TEXTURE0);
	render_target.bindTexture();
	glUniform1i(this->render_shader.ParameterLocation("src_tx2D"), 0);

	glDrawArrays(GL_TRIANGLES, 0, 6);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);

	this->render_shader.Disable();

	glBlendFuncSeparate(state_blend_src_rgb, state_blend_dst_rgb, state_blend_src_alpha, state_blend_dst_alpha);
	if (!state_blend) glDisable(GL_BLEND);
	if (state_depth_test) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);

    return true;
}
