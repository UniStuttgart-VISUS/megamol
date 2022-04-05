#include "QuickSurf.h"

#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"

using namespace megamol::protein_calls;
using namespace megamol::protein_cuda;

QuickSurf::QuickSurf()
        : core_gl::view::Renderer3DModuleGL()
        , dataInSlot_("dataIn", "Connects this module to the data provider.")
        , lightInSlot_("lightIn", "Connects this module to the light information.")
        , qs_qualityParam_(
              "quicksurf::quality", "QuickSurf quality parameter. 0 is minimum quality, 3 maximimum quality.")
        , qs_radScaleParam_(
              "quicksurf::radiusScale", "Scales the influence radius of particles when computing QuickSurf.")
        , qs_gridSpacingParam_("quicksurf::gridSpacing", "Steers the inter-voxel distance of the QuickSurf grid.")
        , qs_isovalueParam_("quicksurf::isovalue",
              "Isovalue of the QuickSurf surface. Only used for the surface visualization and not the volume output.")
        , colorTableFileParam_("color::colorTableFilename", "Path to the file containing a custom color table")
        , coloringMode0Param_("color::coloringMode0", "The first coloring mode")
        , coloringMode1Param_("color::coloringMode1", "The second coloring mode")
        , coloringModeWeightParam_("color::colorWeighting", "Weighting factor between the two coloring modes")
        , minGradColorParam_("color::minGradColor", "The color for the minimum value for gradient coloring")
        , midGradColorParam_("color::midGradColor", "The color for the middle value for gradient coloring")
        , maxGradColorParam_("color::maxGradColor", "The color for the maximum value for gradient coloring")
        , cudaqsurf_(nullptr)
        , numVoxels_(128, 128, 128)
        , xAxis_(1.0, 0.0, 0.0)
        , yAxis_(0.0, 1.0, 0.0)
        , zAxis_(0.0, 0.0, 1.0)
        , gridOrigin_(0.0, 0.0, 0.0)
        , vertBuffers_{}
        , vaoHandles_{}
        , meshShader_(nullptr)
        , tableFromFile_(false)
        , atomColorTable_{}
        , fileColorTable_{}
        , rainbowColorTable_{}
        , currentColoringMode0_(ProteinColor::ColoringMode::CHAIN)
        , currentColoringMode1_(ProteinColor::ColoringMode::ELEMENT)
        , refreshColors_(true)
        , lastHash_(0)
        , lastFrame_(0) {

    dataInSlot_.SetCompatibleCall<MolecularDataCallDescription>();
    dataInSlot_.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->dataInSlot_);

    lightInSlot_.SetCompatibleCall<core::view::light::CallLightDescription>();
    lightInSlot_.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&lightInSlot_);

    // TODO switch this to an enum
    qs_qualityParam_.SetParameter(new core::param::IntParam(2, 0, 3));
    this->MakeSlotAvailable(&qs_qualityParam_);

    qs_radScaleParam_.SetParameter(new core::param::FloatParam(1.0, 0.0));
    this->MakeSlotAvailable(&qs_radScaleParam_);

    qs_gridSpacingParam_.SetParameter(new core::param::FloatParam(1.0, 0.0));
    this->MakeSlotAvailable(&qs_gridSpacingParam_);

    qs_isovalueParam_.SetParameter(new core::param::FloatParam(0.5, 0.0));
    this->MakeSlotAvailable(&qs_isovalueParam_);

    std::string filename("colors.txt");
    tableFromFile_ = ProteinColor::ReadColorTableFromFile(filename, fileColorTable_);
    colorTableFileParam_.SetParameter(
        new core::param::FilePathParam(filename, core::param::FilePathParam::FilePathFlags_::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&colorTableFileParam_);

    currentColoringMode0_ = ProteinColor::ColoringMode::CHAIN;
    currentColoringMode1_ = ProteinColor::ColoringMode::ELEMENT;
    core::param::EnumParam* cm0 = new core::param::EnumParam(static_cast<int>(currentColoringMode0_));
    core::param::EnumParam* cm1 = new core::param::EnumParam(static_cast<int>(currentColoringMode1_));
    ProteinColor::ColoringMode cMode;
    for (uint32_t cCnt = 0; cCnt < static_cast<uint32_t>(ProteinColor::ColoringMode::MODE_COUNT); ++cCnt) {
        cMode = static_cast<ProteinColor::ColoringMode>(cCnt);
        cm0->SetTypePair(static_cast<int>(cMode), ProteinColor::GetName(cMode).c_str());
        cm1->SetTypePair(static_cast<int>(cMode), ProteinColor::GetName(cMode).c_str());
    }
    coloringMode0Param_ << cm0;
    coloringMode1Param_ << cm1;
    this->MakeSlotAvailable(&coloringMode0Param_);
    this->MakeSlotAvailable(&coloringMode1Param_);

    coloringModeWeightParam_.SetParameter(new core::param::FloatParam(0.5f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&coloringModeWeightParam_);

    minGradColorParam_.SetParameter(new core::param::ColorParam("#146496"));
    this->MakeSlotAvailable(&minGradColorParam_);

    midGradColorParam_.SetParameter(new core::param::ColorParam("#f0f0f0"));
    this->MakeSlotAvailable(&midGradColorParam_);

    maxGradColorParam_.SetParameter(new core::param::ColorParam("#ae3b32"));
    this->MakeSlotAvailable(&maxGradColorParam_);

    auto defparams = deferredProvider_.getUsedParamSlots();
    for (const auto& param : defparams) {
        this->MakeSlotAvailable(param);
    }

    ProteinColor::MakeRainbowColorTable(100, rainbowColorTable_);
}

QuickSurf::~QuickSurf() {
    this->Release();
}

bool QuickSurf::create(void) {

    try {
        auto const shdr_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

        meshShader_ = core::utility::make_glowl_shader("quicksurf_mesh", shdr_options,
            std::filesystem::path("protein_cuda/quicksurf/qsurf_mesh.vert.glsl"),
            std::filesystem::path("protein_cuda/quicksurf/qsurf_mesh.frag.glsl"));

    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "[QuickSurf] %s", ex.what());
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "[QuickSurf] Unable to compile shader: Unknown exception: %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "[QuickSurf] Unable to compile shader: Unknown exception.");
    }

    deferredProvider_.setup(this->GetCoreInstance());
    return true;
}

void QuickSurf::release(void) {
    // TODO
}

bool QuickSurf::GetExtents(core_gl::view::CallRender3DGL& call) {
    // TODO multiple possible input calls
    MolecularDataCall* mol = dataInSlot_.CallAs<MolecularDataCall>();
    if (mol == nullptr) {
        return false;
    }
    mol->SetCalltime(call.Time());
    mol->SetFrameID(static_cast<int>(call.Time()));
    if (!(*mol)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }

    call.AccessBoundingBoxes() = mol->AccessBoundingBoxes();
    call.SetTimeFramesCount(mol->FrameCount());

    return true;
}

bool QuickSurf::Render(core_gl::view::CallRender3DGL& call) {
    auto call_fbo = call.GetFramebuffer();
    deferredProvider_.setFramebufferExtents(call_fbo->getWidth(), call_fbo->getHeight());

    call_fbo->bind();

    bool externalfbo = false;
    if (call_fbo->getNumColorAttachments() == 3) {
        externalfbo = true;
    } else {
        deferredProvider_.bindDeferredFramebufferToDraw();
    }

    // TODO multiple possible input calls
    MolecularDataCall* mol = dataInSlot_.CallAs<MolecularDataCall>();
    if (mol == nullptr) {
        return false;
    }

    mol->SetCalltime(call.Time());
    mol->SetFrameID(static_cast<int>(call.Time()));

    if (!(*mol)(MolecularDataCall::CallForGetData)) {
        return false;
    }

    if (cudaqsurf_ == nullptr) {
        cudaqsurf_ = std::make_unique<CUDAQuickSurf>();
    }

    if (HandleParameters(*mol)) {
        if (refreshColors_) {
            std::vector<glm::vec3> smallColorTable = {
                glm::make_vec3(minGradColorParam_.Param<core::param::ColorParam>()->Value().data()),
                glm::make_vec3(midGradColorParam_.Param<core::param::ColorParam>()->Value().data()),
                glm::make_vec3(maxGradColorParam_.Param<core::param::ColorParam>()->Value().data())};
            float weight = coloringModeWeightParam_.Param<core::param::FloatParam>()->Value();
            ProteinColor::MakeWeightedColorTable(*mol, currentColoringMode0_, currentColoringMode1_, weight,
                1.0 - weight, atomColorTable_, smallColorTable, fileColorTable_, rainbowColorTable_, nullptr, nullptr,
                true);
            refreshColors_ = false;
        }

        // TODO transparency
        auto ret = this->calculateSurface(*mol, qs_qualityParam_.Param<core::param::IntParam>()->Value(),
            qs_radScaleParam_.Param<core::param::FloatParam>()->Value(),
            qs_gridSpacingParam_.Param<core::param::FloatParam>()->Value(),
            qs_isovalueParam_.Param<core::param::FloatParam>()->Value(), false);
        if (!ret) {
            core::utility::log::Log::DefaultLog.WriteError("The QuickSurf gaussian surface could not be calculated");
            return false;
        }
        // clear all vaos
        if (vaoHandles_.size() > 0) {
            glDeleteVertexArrays(vaoHandles_.size(), vaoHandles_.data());
            vaoHandles_.clear();
        }

        /** Creating new vaos each time is very ugly, but we have to do it like that unless Quicksurf
         *  stops to create new buffers every frame. This cannot really be avoided as it is not clear
         *  beforehand how many chunks will be produced. */
        for (auto& buf : vertBuffers_) {
            GLuint vaoHandle = 0;

            // generate new vao with all buffers bound
            glGenVertexArrays(1, &vaoHandle);
            glBindVertexArray(vaoHandle);

            buf.positionBuffer_->bind();
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

            buf.normalBuffer_->bind();
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

            buf.colorBuffer_->bind();
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

            glBindVertexArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            glDisableVertexAttribArray(0);
            glDisableVertexAttribArray(1);
            glDisableVertexAttribArray(2);

            std::vector<glm::vec3> data(buf.vertCount_ / 3, glm::vec3(0.0));
            glGetNamedBufferSubData(buf.colorBuffer_->getName(), 0, buf.vertCount_, data.data());

            vaoHandles_.push_back(vaoHandle);
        }
    }

    auto& cam = call.GetCamera();
    auto view = cam.getViewMatrix();
    auto proj = cam.getProjectionMatrix();
    auto mvp = proj * view;

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    /** render each vao seperately */
    int i = 0;
    for (const auto& vao : vaoHandles_) {
        glBindVertexArray(vao);

        meshShader_->use();
        meshShader_->setUniform("mvp", mvp);
        meshShader_->setUniform("alpha", 1.0f);

        glDrawArrays(GL_TRIANGLES, 0, vertBuffers_[i].vertCount_);

        glUseProgram(0);
        glBindVertexArray(0);
        ++i;
    }

    glDisable(GL_DEPTH_TEST);

    if (externalfbo) {
        call_fbo->bind();
    } else {
        deferredProvider_.resetToPreviousFramebuffer();
        deferredProvider_.draw(call, this->lightInSlot_.CallAs<core::view::light::CallLight>());
    }

    return true;
}

bool QuickSurf::HandleParameters(protein_calls::MolecularDataCall& mdc) {
    bool retval = false;
    if (lastHash_ != mdc.DataHash()) {
        lastHash_ = mdc.DataHash();
        retval = true;
    }
    if (lastFrame_ != mdc.FrameID()) {
        lastFrame_ = mdc.FrameID();
        retval = true;
    }
    if (qs_qualityParam_.IsDirty()) {
        qs_qualityParam_.ResetDirty();
        retval = true;
    }
    if (qs_radScaleParam_.IsDirty()) {
        qs_radScaleParam_.ResetDirty();
        retval = true;
    }
    if (qs_gridSpacingParam_.IsDirty()) {
        qs_gridSpacingParam_.ResetDirty();
        retval = true;
    }
    if (qs_isovalueParam_.IsDirty()) {
        qs_isovalueParam_.ResetDirty();
        retval = true;
    }
    if (colorTableFileParam_.IsDirty()) {
        auto filename = colorTableFileParam_.Param<core::param::FilePathParam>()->Value();
        tableFromFile_ = ProteinColor::ReadColorTableFromFile(filename, fileColorTable_);
        colorTableFileParam_.ResetDirty();
        retval = true;
        refreshColors_ = true;
    }
    if (coloringMode0Param_.IsDirty()) {
        currentColoringMode0_ =
            static_cast<ProteinColor::ColoringMode>(coloringMode0Param_.Param<core::param::EnumParam>()->Value());
        coloringMode0Param_.ResetDirty();
        retval = true;
        refreshColors_ = true;
    }
    if (coloringMode1Param_.IsDirty()) {
        currentColoringMode1_ =
            static_cast<ProteinColor::ColoringMode>(coloringMode1Param_.Param<core::param::EnumParam>()->Value());
        coloringMode1Param_.ResetDirty();
        retval = true;
        refreshColors_ = true;
    }
    if (coloringModeWeightParam_.IsDirty()) {
        coloringModeWeightParam_.ResetDirty();
        retval = true;
        refreshColors_ = true;
    }
    if (minGradColorParam_.IsDirty()) {
        minGradColorParam_.ResetDirty();
        retval = true;
        refreshColors_ = true;
    }
    if (midGradColorParam_.IsDirty()) {
        midGradColorParam_.ResetDirty();
        retval = true;
        refreshColors_ = true;
    }
    if (maxGradColorParam_.IsDirty()) {
        maxGradColorParam_.ResetDirty();
        retval = true;
        refreshColors_ = true;
    }

    return retval;
}

bool QuickSurf::calculateSurface(
    MolecularDataCall& mdc, int quality, float radscale, float gridspacing, float isoval, bool sortTriangles) {

    if (mdc.AtomCount() == 0) {
        return false;
    }
    if (mdc.AtomTypeCount() == 0) {
        return false;
    }

    /** Calculate positional and radial bounds */
    glm::vec3 firstpos = glm::make_vec3(mdc.AtomPositions());
    float firstrad = mdc.AtomTypes()[mdc.AtomTypeIndices()[0]].Radius();
    glm::vec2 bounds_r{firstrad};
    for (size_t i = 0; i < mdc.AtomCount(); ++i) {
        glm::vec3 pos = glm::make_vec3(&mdc.AtomPositions()[i * 3]);
        float rad = mdc.AtomTypes()[mdc.AtomTypeIndices()[i]].Radius();

        bounds_r.x = (rad < bounds_r.x) ? rad : bounds_r.x;
        bounds_r.y = (rad > bounds_r.y) ? rad : bounds_r.y;
    }

    

    // crude estimate of the grid padding we require to prevent the
    // resulting isosurface from being clipped
    float gridpadding = radscale * bounds_r.y * 1.5f;
    float padrad = gridpadding;
    float pi = std::acos(-1); // simple trick to get an accurate pi value without c++ 20
    padrad = static_cast<float>(0.4 * sqrt(4.0 / 3.0 * pi * padrad * padrad * padrad));
    gridpadding = std::max(gridpadding, padrad);

    // we ignore the padding and the radscale. For us, the following is enough
    auto& bbox = mdc.AccessBoundingBoxes().ObjectSpaceBBox();
    glm::vec3 mincoord(bbox.Left(), bbox.Bottom(), bbox.Back());
    glm::vec3 maxcoord(bbox.Right(), bbox.Top(), bbox.Front());

    xAxis_.x = maxcoord.x - mincoord.x;
    yAxis_.y = maxcoord.y - mincoord.y;
    zAxis_.z = maxcoord.z - mincoord.z;

    numVoxels_.x = static_cast<int32_t>(std::ceil(xAxis_.x / gridspacing));
    numVoxels_.y = static_cast<int32_t>(std::ceil(yAxis_.y / gridspacing));
    numVoxels_.z = static_cast<int32_t>(std::ceil(zAxis_.z / gridspacing));

    gridOrigin_ = mincoord;

    std::vector<glm::vec4> xyzr(mdc.AtomCount()); // positions
    std::vector<glm::vec4> colors(mdc.AtomCount());

    for (size_t i = 0; i < mdc.AtomCount(); ++i) {
        // position
        glm::vec3 pos = glm::make_vec3(&mdc.AtomPositions()[i * 3]);
        xyzr[i].x = pos.x - gridOrigin_.x;
        xyzr[i].y = pos.y - gridOrigin_.y;
        xyzr[i].z = pos.z - gridOrigin_.z;
        xyzr[i].w = mdc.AtomTypes()[mdc.AtomTypeIndices()[i]].Radius();

        // color
        glm::vec3 col = atomColorTable_[i];
        colors[i].r = col.r;
        colors[i].g = col.g;
        colors[i].b = col.b;
        colors[i].a = 1.0; // TODO do transparency stuff
    }

    float gausslim = 2.0f; // low quality, value = 0
    if (quality == 1) {
        gausslim = 2.5f; // medium quality, value = 1
    } else if (quality == 2) {
        gausslim = 3.0f; // high quality, value = 2
    } else if (quality == 3) {
        gausslim = 4.0f; //ultra quality, value = 3
    }

    int retval =
        cudaqsurf_->calc_surf(mdc.AtomCount(), &xyzr[0].x, &colors[0].x, true, CUDAQuickSurf::VolTexFormat::RGB3F,
            &gridOrigin_.x, &numVoxels_.x, bounds_r.y, radscale, gridspacing, isoval, gausslim, vertBuffers_);

    return true;
}
