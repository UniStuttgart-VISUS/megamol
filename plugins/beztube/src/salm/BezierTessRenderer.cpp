#include "stdafx.h"
#include "BezierTessRenderer.h"
#include "vislib/sys/Log.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/misc/BezierCurvesListDataCall.h"
#include "vislib/String.h"

using namespace megamol;;
using namespace megamol::beztube;
using vislib::sys::Log;
using vislib::graphics::gl::GLSLTesselationShader;
using vislib::graphics::gl::ShaderSource;
using core::misc::BezierCurvesListDataCall;

salm::BezierTessRenderer::gpuResources::gpuResources(core::utility::ShaderSourceFactory& shaderFactor) : error(true), shader() {
    // migrated from
    //  opentk_test1\opentk_test1\program.cs : OpenTK_Test1.Program.OnLoad

    //GL.Enable(EnableCap.FramebufferSrgb);
    //GL.ClearColor(0.45f, 0.04f, 0.04f, 1.0f);
    //GL.ClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    //GL.Enable(EnableCap.DepthTest);
    //GL.DepthFunc(DepthFunction.Less);

    //GL.Enable(EnableCap.CullFace);
    //GL.CullFace(CullFaceMode.Back);

    vislib::StringA definesStr(""), s;
    definesStr += "#define SHADING_STYLE 1\n";
    definesStr += "#define CAPS\n";
    definesStr += "#define USE_BACKPATCHCULLING\n";
    definesStr += "#define USE_FINE_CAPS_TESSELLATION\n";
    s.Format("#define VRING_VERTEX_COUNT %u\n", cylinderXSize);
    definesStr += s;
    s.Format("#define VRING_COUNT %u\n", cylinderYSize);
    definesStr += s;
    vislib::SmartPtr<ShaderSource::Snippet> defines(new ShaderSource::StringSnippet(definesStr));

    ShaderSource vs, tcs, tes, fs;

    if (!shaderFactor.MakeShaderSource("beztubeSalm::Tube.vs", vs)) {
        Log::DefaultLog.WriteError("Failed to load vertex shader");
        return;
    }
    const size_t snippetIdx = 1;
    vislib::StringA test = vs[snippetIdx]->PeekCode();
    test.TrimSpacesBegin();
    assert(test.StartsWith("REPLACED_BY_CODE"));
    vs.Replace(snippetIdx, defines);
    if (!shaderFactor.MakeShaderSource("beztubeSalm::Tube.tc", tcs)) {
        Log::DefaultLog.WriteError("Failed to load tessellation control shader");
        return;
    }
    test = tcs[snippetIdx]->PeekCode();
    test.TrimSpacesBegin();
    assert(test.StartsWith("REPLACED_BY_CODE"));
    tcs.Replace(snippetIdx, defines);
    if (!shaderFactor.MakeShaderSource("beztubeSalm::Tube.te", tes)) {
        Log::DefaultLog.WriteError("Failed to load tessellation evaluation shader");
        return;
    }
    test = tes[snippetIdx]->PeekCode();
    test.TrimSpacesBegin();
    assert(test.StartsWith("REPLACED_BY_CODE"));
    tes.Replace(snippetIdx, defines);
    if (!shaderFactor.MakeShaderSource("beztubeSalm::Tube.fs", fs)) {
        Log::DefaultLog.WriteError("Failed to load fragment shader");
        return;
    }
    test = fs[snippetIdx]->PeekCode();
    test.TrimSpacesBegin();
    assert(test.StartsWith("REPLACED_BY_CODE"));
    fs.Replace(snippetIdx, defines);

    if (!shader.Compile(vs.Code(), vs.Count(), tcs.Code(), tcs.Count(), tes.Code(), tes.Count(), nullptr, 0, fs.Code(), fs.Count(), false)) {
        Log::DefaultLog.WriteError("Failed to compile shader");
        return;
    }

    if (!shader.Link()) {
        Log::DefaultLog.WriteError("Failed to link shader program");
        shader.Release();
        return;
    }

    indexBuffer.CreateQuadCylinder(cylinderXSize, cylinderYSize);

    //// this is dummy data and will be overwritten
    //int tubesCount = 16;
    //int totalNodeCount = tubesCount + 1;
    //int nodeCount[] = { 4, 4, 4, 3, 2 }; // tubesCount = 16

    //var version = file.ReadInt32();
    //tubesCount = file.ReadInt32();
    //nodeCount = new int[tubesCount];
    //for (var i = 0; i < tubesCount; ++i)
    //    nodeCount[i] = file.ReadInt32();

    //totalNodeCount = file.ReadInt32();

    tubes.Allocate(SplineTubesTess::Color,
        (SplineTubesTess::TessTubeRenderSetting)(
        SplineTubesTess::Null
        | SplineTubesTess::UseCaps
        //| SplineTubesTess::ShaderBufferTypeIsSSBO
        | SplineTubesTess::UseBackPatchCulling
        | SplineTubesTess::UseFineCapsTessellation
        //| SplineTubesTess::InvertColorsForCaps
        //| SplineTubesTess::UseGeoShader
        ), GL_STREAM_DRAW, -1, nullptr, cylinderYSize, cylinderXSize);

#if 0

    for (var i = 0; i < totalNodeCount; ++i)
    {
        TubeNode node;

        node.Pos = new Vector3(file.ReadSingle(), file.ReadSingle(), file.ReadSingle());
        node.Rad = file.ReadSingle();
        node.Col = new Vector3(file.ReadSingle(), file.ReadSingle(), file.ReadSingle());

        node.PTan = new Vector3(file.ReadSingle(), file.ReadSingle(), file.ReadSingle());
        node.RTan = file.ReadSingle();
        node.CTan = new Vector3(file.ReadSingle(), file.ReadSingle(), file.ReadSingle());

        tubes.SetNode(i, node);
    }

    tubes.FixZeroTangents();
    tubes.SetCenterToOrigin();
    tubes.GenTangentFrames();
    tubes.BindShaderAndShaderBuffer();
    tubes.BindMeshData();

#endif

    perFrameData.Allocate(5);

    perFrameUBO.Allocate(ShaderBuffer::UBO, perFrameData.Float4BlockCount(), GL_STREAM_DRAW);
    perFrameUBO.BindToIndex(0);
    //tubes.Shader.BindShaderBuffer(perFrameUBO, "PerFrameBuffer");

    staticData.Allocate(1);
    staticData.Fill(0, 0, glm::vec2(640.0f, 480.0f)); // just dummy data for now

    staticUBO.Allocate(ShaderBuffer::UBO, staticData, GL_STATIC_DRAW);
    staticUBO.BindToIndex(1);
    //tubes.Shader.BindShaderBuffer(staticUBO, "StaticBuffer");

    //UpdateProjMat();

    error = false;
}

salm::BezierTessRenderer::gpuResources::~gpuResources() {
    shader.Release();
}

/****************************************************************************/

salm::BezierTessRenderer::BezierTessRenderer() : AbstractBezierRenderer(), gpuRes() {

    this->getDataSlot.SetCompatibleCall<core::misc::BezierCurvesListDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

}

salm::BezierTessRenderer::~BezierTessRenderer() {
    this->Release();
}

bool salm::BezierTessRenderer::create(void) {
    if (!AbstractBezierRenderer::create()) return false;
    if (!vislib::graphics::gl::GLSLTesselationShader::InitialiseExtensions()) {
        Log::DefaultLog.WriteError("Failed to initialize tesselation shader extensions");
        return false;
    }

    gpuRes = std::make_shared<gpuResources>(GetCoreInstance()->ShaderSourceFactory());
    if (gpuRes->Error()) {
        gpuRes.reset();
        return false;
    }

    return true;
}

void salm::BezierTessRenderer::release(void) {
    gpuRes.reset();
    shader = nullptr;
}

bool salm::BezierTessRenderer::render(core::view::CallRender3D& call) {

    // Check and update data
    ///////////////////////////////////////////////

    // Check and update viewport
    ///////////////////////////////////////////////

    //Vector3 camPos;
    //var viewMat = NewOrbitCamMat(Vector3.Zero, new Vector2(-camPhi, -camTheta), camZoom + 4f, out camPos);
    //viewProjMat = Matrix4.Transpose(viewMat) * projMat;

    // Check and update camera
    ///////////////////////////////////////////////

    //perFrameData.Fill(0, viewProjMat);
    //perFrameData.Fill(4, 0, camPos);

    //perFrameUBO.Update(perFrameData);

    // render
    ///////////////////////////////////////////////

    ////perFrameUBO.Bind();

    //tubes.UpdateAndDrawAll();

    ////tubes.DrawCaps();
    ////indexBuffer.Bind();

    ////GL.PatchParameter(PatchParameterInt.PatchVertices, 4);

    ////GL.DrawElementsInstanced(PrimitiveType.Patches, indexBuffer.Length, DrawElementsType.UnsignedInt, IntPtr.Zero, 2);

    return true;
}
