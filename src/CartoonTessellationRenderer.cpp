/*
 * CartoonTessellationRenderer.cpp
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CartoonTessellationRenderer.h"
#include "mmcore/moldyn/MolecularDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/FloatParam.h"
#include "vislib/assert.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Matrix.h"
#include <inttypes.h>
#include <stdint.h>

using namespace megamol::core;
using namespace megamol::core::moldyn;
using namespace megamol::protein;

#define RENDER_ATOMS_AS_SPHERES 1

#define MAP_BUFFER_LOCALLY
#define DEBUG_BLAHBLAH

const GLuint SSBObindingPoint = 2;
//#define NGS_THE_INSTANCE "gl_InstanceID"
#define NGS_THE_INSTANCE "gl_VertexID"

//typedef void (APIENTRY *GLDEBUGPROC)(GLenum source,GLenum type,GLuint id,GLenum severity,GLsizei length,const GLchar *message,const void *userParam);
void APIENTRY MyFunkyDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
    const GLchar* message, const GLvoid* userParam) {
        const char *sourceText, *typeText, *severityText;
        switch(source) {
            case GL_DEBUG_SOURCE_API:
                sourceText = "API";
                break;
            case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
                sourceText = "Window System";
                break;
            case GL_DEBUG_SOURCE_SHADER_COMPILER:
                sourceText = "Shader Compiler";
                break;
            case GL_DEBUG_SOURCE_THIRD_PARTY:
                sourceText = "Third Party";
                break;
            case GL_DEBUG_SOURCE_APPLICATION:
                sourceText = "Application";
                break;
            case GL_DEBUG_SOURCE_OTHER:
                sourceText = "Other";
                break;
            default:
                sourceText = "Unknown";
                break;
        }
        switch(type) {
            case GL_DEBUG_TYPE_ERROR:
                typeText = "Error";
                break;
            case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
                typeText = "Deprecated Behavior";
                break;
            case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
                typeText = "Undefined Behavior";
                break;
            case GL_DEBUG_TYPE_PORTABILITY:
                typeText = "Portability";
                break;
            case GL_DEBUG_TYPE_PERFORMANCE:
                typeText = "Performance";
                break;
            case GL_DEBUG_TYPE_OTHER:
                typeText = "Other";
                break;
            case GL_DEBUG_TYPE_MARKER:
                typeText = "Marker";
                break;
            default:
                typeText = "Unknown";
                break;
        }
        switch(severity) {
            case GL_DEBUG_SEVERITY_HIGH:
                severityText = "High";
                break;
            case GL_DEBUG_SEVERITY_MEDIUM:
                severityText = "Medium";
                break;
            case GL_DEBUG_SEVERITY_LOW:
                severityText = "Low";
                break;
            case GL_DEBUG_SEVERITY_NOTIFICATION:
                severityText = "Notification";
                break;
            default:
                severityText = "Unknown";
                break;
        }
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[%s %s] (%s %u) %s\n", sourceText, severityText, typeText, id, message);
}


/*
 * moldyn::CartoonTessellationRenderer::CartoonTessellationRenderer
 */
CartoonTessellationRenderer::CartoonTessellationRenderer(void) : Renderer3DModule(),
    getDataSlot("getdata", "Connects to the data source"),
    fences(), currBuf(0), bufSize(32 * 1024 * 1024), numBuffers(3),
    scalingParam("scaling", "scaling factor for particle radii"),
    // this variant should not need the fence
    singleBufferCreationBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT),
    singleBufferMappingBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT),
    theShaders() {

    this->getDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->scalingParam << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->scalingParam);
    fences.resize(numBuffers);
}


/*
 * moldyn::CartoonTessellationRenderer::~CartoonTessellationRenderer
 */
CartoonTessellationRenderer::~CartoonTessellationRenderer(void) {
    this->Release();
}

void CartoonTessellationRenderer::queueSignal(GLsync& syncObj) {
    if (syncObj) {
        glDeleteSync(syncObj);
    }
    syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

void CartoonTessellationRenderer::waitSignal(GLsync& syncObj) {
    if (syncObj) {
        while (1) {
            GLenum wait = glClientWaitSync(syncObj, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                return;
            }
        }
    }
}


/*
 * moldyn::SimpleSphereRenderer::create
 */
bool CartoonTessellationRenderer::create(void) {
    using namespace vislib::sys;
    using namespace vislib::graphics::gl;
#ifdef DEBUG_BLAHBLAH
    glDebugMessageCallback(MyFunkyDebugCallback, NULL);
#endif
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions())
        return false;
    if (!vislib::graphics::gl::GLSLTesselationShader::InitialiseExtensions())
        return false;

    //vislib::graphics::gl::ShaderSource vert, frag;
    vert = new ShaderSource();
    tessCont = new ShaderSource();
    tessEval = new ShaderSource();
    geom = new ShaderSource();
    frag = new ShaderSource();
    if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellation::vertex", *vert)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellation::tesscontrol", *tessCont)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellation::tesseval", *tessEval)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellation::geometry", *geom)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellation::fragment", *frag)) {
        return false;
    }

    // Load sphere shader
    ShaderSource vertSrc;
    ShaderSource fragSrc;
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
        "protein::std::sphereVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
        "protein::std::sphereFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    try {
        if (!this->sphereShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    }
    catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create sphere shader: %s\n", e.GetMsgA());
        return false;
    }

    glGenVertexArrays(1, &this->vertArray);
    glBindVertexArray(this->vertArray);
    glGenBuffers(1, &this->theSingleBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSingleBuffer);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, this->bufSize * this->numBuffers, nullptr, singleBufferCreationBits);
    this->theSingleMappedMem = glMapNamedBufferRangeEXT(this->theSingleBuffer, 0, this->bufSize * this->numBuffers, singleBufferMappingBits);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindVertexArray(0);

    return true;
}

bool CartoonTessellationRenderer::makeColorString(MolecularDataCall &mol, std::string &code, std::string &declaration) {
    //declaration = "    float colorIndex;\n";
    //code = "    theColIdx = theBuffer[" NGS_THE_INSTANCE " + instanceOffset].colorIndex; \n";
    declaration = "";
    code = "    theColor = gl_Color; \n";
    //declaration = "    vec4 color;\n";
    //code = "    theColor = theBuffer[" NGS_THE_INSTANCE " + instanceOffset].color;\n";
    return true;
}

bool CartoonTessellationRenderer::makeVertexString(MolecularDataCall &mol, std::string &code, std::string &declaration)  {
    declaration = "    float posX; float posY; float posZ; float posR;\n";
    //declaration = "    float posX; float posY; float posZ;\n";
    code = "    gl_Position = vec4(theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posX,\n"
           "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posY,\n"
           "                 theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posZ, 1.0); \n"
           "    rad = theBuffer[" NGS_THE_INSTANCE " + instanceOffset].posR;";
            //"    rad = CONSTRAD;";
    return true;
}

std::shared_ptr<GLSLShader> CartoonTessellationRenderer::makeShader(vislib::SmartPtr<ShaderSource> vert, vislib::SmartPtr<ShaderSource> frag) {
    std::shared_ptr<GLSLShader> sh = std::make_shared<GLSLShader>(GLSLShader());
    try {
        if (!sh->Create(vert->Code(), vert->Count(), frag->Code(), frag->Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile sphere shader: Unknown error\n");
            return false;
        }

    }
    catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()), ce.GetMsgA());
        return false;
    }
    catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: %s\n", e.GetMsgA());
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: Unknown exception\n");
        return false;
    }
    return sh;
}

std::shared_ptr<GLSLTesselationShader> CartoonTessellationRenderer::makeShader(vislib::SmartPtr<ShaderSource> vert, 
    vislib::SmartPtr<ShaderSource> tessCont, vislib::SmartPtr<ShaderSource> tessEval, 
    vislib::SmartPtr<ShaderSource> geom, vislib::SmartPtr<ShaderSource> frag) {

    std::shared_ptr<GLSLTesselationShader> sh = std::make_shared<GLSLTesselationShader>(GLSLTesselationShader());
    try {
        // compile the shader
        if (!sh->Compile(vert->Code(), vert->Count(),
            tessCont->Code(), tessCont->Count(),
            tessEval->Code(), tessEval->Count(),
            geom->Code(), geom->Count(),
            frag->Code(), frag->Count())) {
            throw vislib::Exception("Could not compile cartoon shader. ", __FILE__, __LINE__);
        }
        // link the shader
        if (!sh->Link()){
            throw vislib::Exception("Could not link cartoon shader", __FILE__, __LINE__);
        }
    }
    catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile tessellation shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()), ce.GetMsgA());
        return false;
    }
    catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile tessellation shader: %s\n", e.GetMsgA());
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile tessellation shader: Unknown exception\n");
        return false;
    }
    return sh;
}

std::shared_ptr<vislib::graphics::gl::GLSLShader> CartoonTessellationRenderer::generateShader(MolecularDataCall &mol) {
    int c = 0;
    int p = 1;

    shaderMap::iterator i = theShaders.find({ c, p });
    if (i == theShaders.end()) {
        unsigned int colBytes, vertBytes, colStride, vertStride;
        this->getBytesAndStride(mol, colBytes, vertBytes, colStride, vertStride);

        vislib::SmartPtr<ShaderSource> v2 = new ShaderSource(*vert);
        vislib::SmartPtr<ShaderSource::Snippet> codeSnip, declarationSnip;
        std::string vertCode, colCode, vertDecl, colDecl;
        makeVertexString(mol, vertCode, vertDecl);
        makeColorString(mol, colCode, colDecl);
        std::string decl = "\nstruct SphereParams {\n";
        decl += vertDecl;
        decl += colDecl;
        decl += "};\n";

        decl += "layout(packed, binding = " + std::to_string(SSBObindingPoint) + ") buffer shader_data {\n"
            "    SphereParams theBuffer[];\n"
            "};\n";
        std::string code = "\n";
        code += colCode;
        code += vertCode;
        declarationSnip = new ShaderSource::StringSnippet(decl.c_str());
        codeSnip = new ShaderSource::StringSnippet(code.c_str());
        //v2->Insert(3, declarationSnip);
        //v2->Insert(5, codeSnip);
        std::string s(v2->WholeCode());

        vislib::SmartPtr<ShaderSource> vss(v2);
        theShaders.emplace(std::make_pair(std::make_pair(c, p), makeShader(v2, tessCont, tessEval, geom, frag)));
        i = theShaders.find({ c, p });

    }
    return i->second;
}


/*
 * moldyn::SimpleSphereRenderer::release
 */
void CartoonTessellationRenderer::release(void) {
    glUnmapNamedBufferEXT(this->theSingleBuffer);
    for (auto &x : fences) {
        if (x) {
            glDeleteSync(x);
        }
    }
    //this->sphereShader.Release();
    // TODO release all shaders!
    glDeleteVertexArrays(1, &this->vertArray);
    glDeleteBuffers(1, &this->theSingleBuffer);
}


void CartoonTessellationRenderer::getBytesAndStride(MolecularDataCall &mol, unsigned int &colBytes, unsigned int &vertBytes,
    unsigned int &colStride, unsigned int &vertStride) {
    vertBytes = 0; colBytes = 0;
    //colBytes = vislib::math::Max(colBytes, 3 * 4U);
    vertBytes = vislib::math::Max(vertBytes, 4 * 4U);

    colStride = 0;
    colStride = colStride < colBytes ? colBytes : colStride;
    vertStride = 0;
    vertStride = vertStride < vertBytes ? vertBytes : vertStride;
}


/*
* GetCapabilities
*/
bool CartoonTessellationRenderer::GetCapabilities(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        view::CallRender3D::CAP_RENDER
        | view::CallRender3D::CAP_LIGHTING
        | view::CallRender3D::CAP_ANIMATION
        );

    return true;
}


/*
* GetExtents
*/
bool CartoonTessellationRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    MolecularDataCall *mol = this->getDataSlot.CallAs<MolecularDataCall>();
    if ((mol != NULL) && ((*mol)(MolecularDataCall::CallForGetExtent))) {
        cr->SetTimeFramesCount(mol->FrameCount());
        cr->AccessBoundingBoxes() = mol->AccessBoundingBoxes();

        float scaling = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        }
        else {
            scaling = 1.0f;
        }
        cr->AccessBoundingBoxes().MakeScaledWorld(scaling);

    }
    else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }

    return true;
}

/*
 *  getData
 */
MolecularDataCall* CartoonTessellationRenderer::getData(unsigned int t, float& outScaling) {
    MolecularDataCall *mol = this->getDataSlot.CallAs<MolecularDataCall>();
    outScaling = 1.0f;
    if (mol != NULL) {
        mol->SetFrameID(t);
        if (!(*mol)(MolecularDataCall::CallForGetExtent)) return NULL;

        // calculate scaling
        outScaling = mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (outScaling > 0.0000001) {
            outScaling = 10.0f / outScaling;
        }
        else {
            outScaling = 1.0f;
        }

        mol->SetFrameID(t);
        if (!(*mol)(MolecularDataCall::CallForGetData)) return NULL;

        return mol;
    }
    else {
        return NULL;
    }
}

/*
 * moldyn::SimpleSphereRenderer::Render
 */
bool CartoonTessellationRenderer::Render(Call& call) {
#ifdef DEBUG_BLAHBLAH
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    float scaling = 1.0f;
    MolecularDataCall *mol = this->getData(static_cast<unsigned int>(cr->Time()), scaling);
    if (mol == NULL) return false;

//	timer.BeginFrame();

    float clipDat[4];
    float clipCol[4];
    clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
    clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
    clipCol[3] = 1.0f;
    
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    float viewportStuff[4];
    ::glGetFloatv(GL_VIEWPORT, viewportStuff);
    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glScalef(scaling, scaling, scaling);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, theSingleBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer);

    // this is the apex of suck and must die
    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrix(&modelViewMatrix_column[0]);
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);
    // Compute modelviewprojection matrix
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrixInv = modelViewMatrix;
    modelViewMatrixInv.Invert();
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrix = projMatrix * modelViewMatrix;
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrixInv = modelViewProjMatrix;
    modelViewProjMatrixInv.Invert();
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrixTransp = modelViewProjMatrix;
    modelViewProjMatrixTransp.Transpose();
    // end suck

    // copy data
    if (this->positions.Count() != mol->MoleculeCount()) {
        this->positions.SetCount(mol->MoleculeCount());
        this->splinePoints.SetCount(mol->MoleculeCount());
    }
    unsigned int firstResIdx = 0;
    unsigned int lastResIdx = 0;
    unsigned int firstAtomIdx = 0;
    unsigned int lastAtomIdx = 0;
    unsigned int atomTypeIdx = 0;
    for (unsigned int molIdx = 0; molIdx < mol->MoleculeCount(); molIdx++){
        this->positions[molIdx].Clear();
        this->positions[molIdx].AssertCapacity(mol->Molecules()[molIdx].ResidueCount() * 4);

        firstResIdx = mol->Molecules()[molIdx].FirstResidueIndex();
        lastResIdx = firstResIdx + mol->Molecules()[molIdx].ResidueCount();
        for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++){
            firstAtomIdx = mol->Residues()[resIdx]->FirstAtomIndex();
            lastAtomIdx = firstAtomIdx + mol->Residues()[resIdx]->AtomCount();
            for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx; atomIdx++){
                unsigned int atomTypeIdx = mol->AtomTypeIndices()[atomIdx];
                if (mol->AtomTypes()[atomTypeIdx].Name().Equals("CA")){
                    this->positions[molIdx].Add(mol->AtomPositions()[3 * atomIdx]);
                    this->positions[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 1]);
                    this->positions[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 2]);
                    this->positions[molIdx].Add(1.0f);
                }
            }
        }

        this->splinePoints[molIdx].Clear();
        this->splinePoints[molIdx].AssertCapacity(this->positions[molIdx].Count() * 3);

        for (unsigned int i = 0; i < (this->positions[molIdx].Count() / 4) - 3; i++) {
            this->splinePoints[molIdx].Add(this->positions[molIdx][4 * i + 0]);
            this->splinePoints[molIdx].Add(this->positions[molIdx][4 * i + 1]);
            this->splinePoints[molIdx].Add(this->positions[molIdx][4 * i + 2]);
            this->splinePoints[molIdx].Add(this->positions[molIdx][4 * i + 3]);
            //this->splinePoints[molIdx].Add(this->positions[molIdx][4 * (i + 1) + 0]);
            //this->splinePoints[molIdx].Add(this->positions[molIdx][4 * (i + 1) + 1]);
            //this->splinePoints[molIdx].Add(this->positions[molIdx][4 * (i + 1) + 2]);
            //this->splinePoints[molIdx].Add(this->positions[molIdx][4 * (i + 1) + 3]);
            //this->splinePoints[molIdx].Add(this->positions[molIdx][4 * (i + 2) + 0]);
            //this->splinePoints[molIdx].Add(this->positions[molIdx][4 * (i + 2) + 1]);
            //this->splinePoints[molIdx].Add(this->positions[molIdx][4 * (i + 2) + 2]);
            //this->splinePoints[molIdx].Add(this->positions[molIdx][4 * (i + 2) + 3]);
            //this->splinePoints[molIdx].Add(this->positions[molIdx][4 * (i + 3) + 0]);
            //this->splinePoints[molIdx].Add(this->positions[molIdx][4 * (i + 3) + 1]);
            //this->splinePoints[molIdx].Add(this->positions[molIdx][4 * (i + 3) + 2]);
            //this->splinePoints[molIdx].Add(this->positions[molIdx][4 * (i + 3) + 3]);
        }
    }

    //currBuf = 0;
    for (unsigned int i = 0; i < this->splinePoints.Count(); i++) {
        newShader = this->generateShader(*mol);

        newShader->Enable();
        glColor4f(1.0f / this->splinePoints.Count() * (i + 1), 0.75f, 0.25f, 1.0f);
        colIdxAttribLoc = glGetAttribLocationARB(*this->newShader, "colIdx");
        glUniform4fv(newShader->ParameterLocation("viewAttr"), 1, viewportStuff);
        glUniform3fv(newShader->ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
        glUniform3fv(newShader->ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
        glUniform3fv(newShader->ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
        glUniform4fv(newShader->ParameterLocation("clipDat"), 1, clipDat);
        glUniform4fv(newShader->ParameterLocation("clipCol"), 1, clipCol);
        glUniformMatrix4fv(newShader->ParameterLocation("MVinv"), 1, GL_FALSE, modelViewMatrixInv.PeekComponents());
        glUniformMatrix4fv(newShader->ParameterLocation("MVP"), 1, GL_FALSE, modelViewProjMatrix.PeekComponents());
        glUniformMatrix4fv(newShader->ParameterLocation("MVPinv"), 1, GL_FALSE, modelViewProjMatrixInv.PeekComponents());
        glUniformMatrix4fv(newShader->ParameterLocation("MVPtransp"), 1, GL_FALSE, modelViewProjMatrixTransp.PeekComponents());
        glUniform1f(newShader->ParameterLocation("scaling"), this->scalingParam.Param<param::FloatParam>()->Value());
        float minC = 0.0f, maxC = 0.0f;
        unsigned int colTabSize = 0;
        glUniform4f(this->newShader->ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));

        unsigned int colBytes, vertBytes, colStride, vertStride;
        this->getBytesAndStride(*mol, colBytes, vertBytes, colStride, vertStride);
        
        UINT64 numVerts, vertCounter;
        numVerts = this->bufSize / vertStride;
        const char *currVert = (const char *)(this->splinePoints[i].PeekElements());
        const char *currCol = 0;
        vertCounter = 0;
        while (vertCounter < this->splinePoints[i].Count() / 4) {
            void *mem = static_cast<char*>(this->theSingleMappedMem) + bufSize * currBuf;
            const char *whence = currVert;
            UINT64 vertsThisTime = vislib::math::Min(this->splinePoints[i].Count() / 4 - vertCounter, numVerts);
            this->waitSignal(fences[currBuf]);
            memcpy(mem, whence, vertsThisTime * vertStride);
            glFlushMappedNamedBufferRangeEXT(theSingleBuffer, bufSize * currBuf, vertsThisTime * vertStride);
            glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);

            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer, bufSize * currBuf, bufSize);
            glPatchParameteri(GL_PATCH_VERTICES, 1);
            glDrawArrays(GL_PATCHES, 0, vertsThisTime);
            this->queueSignal(fences[currBuf]);

            currBuf = (currBuf + 1) % this->numBuffers;
            vertCounter += vertsThisTime;
            currVert += vertsThisTime * vertStride;
            currCol += vertsThisTime * colStride;
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisable(GL_TEXTURE_1D);
        newShader->Disable();
    }

#if RENDER_ATOMS_AS_SPHERES
    glEnable(GL_BLEND);
    glColor4f(0.5f, 0.5f, 0.5f, 0.5f);
    // enable sphere shader
    this->sphereShader.Enable();
    // set shader variables
    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
    // set vertex and color pointers and draw them
    glBegin(GL_POINTS);
    for (unsigned int i = 0; i < this->positions.Count(); i++) {
        for (unsigned int j = 0; j < this->positions[i].Count() / 4; j++) {
            glColor4f(0.75f, 0.5f, 0.1f, 1.0f);
            glVertex4f(this->positions[i][4 * j], this->positions[i][4 * j + 1], this->positions[i][4 * j + 2], 0.3f);
        }
    }
    for (unsigned int i = 0; i < mol->AtomCount(); i++) {
        unsigned int atomTypeIdx = mol->AtomTypeIndices()[i];
        if (mol->AtomTypes()[atomTypeIdx].Name().Equals("CA")){
            glColor4f(0.5f, 0.75f, 0.1f, 0.5f);
            glVertex4f(mol->AtomPositions()[3 * i], mol->AtomPositions()[3 * i + 1], mol->AtomPositions()[3 * i + 2], 1.0f);
        }
        else {
            glColor4f(0.5f, 0.5f, 0.5f, 0.5f);
            glVertex4f(mol->AtomPositions()[3 * i], mol->AtomPositions()[3 * i + 1], mol->AtomPositions()[3 * i + 2], 0.5f);
        }

    }
    glEnd();
    // disable sphere shader
    this->sphereShader.Disable();
#endif

    mol->Unlock();

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
#ifdef DEBUG_BLAHBLAH
    glDisable(GL_DEBUG_OUTPUT);
    glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif

//	timer.EndFrame();

    return true;
}
