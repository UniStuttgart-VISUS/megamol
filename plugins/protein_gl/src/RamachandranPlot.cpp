/**
 * MegaMol
 * Copyright (c) 2017-2022, MegaMol Dev Team
 * All rights reserved.
 */
#include "RamachandranPlot.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore_gl/utility/ShaderFactory.h"

using namespace megamol;
using namespace megamol::protein_gl;

RamachandranPlot::RamachandranPlot()
        : mmstd_gl::Renderer2DModuleGL()
        , moleculeInSlot_("moleculeIn", "Input of the molecular data containing the necessary amino acid information")
        , showBoundingBoxParam_("showBoundingBox", "Steers the display of the bounding box of the graph")
        , pointSizeParam_("pointSize", "Steers the size of the displayed points")
        , pointColorParam_("color::pointColor", "")
        , sureHelixColor_("color::area::sureHelixColor", "Steers the color of the sure helix assignment area")
        , unsureHelixColor_("color::area::unsureHelixColor", "Steers the color of the unsure helix assignment area")
        , sureSheetColor_("color::area::sureSheetColor", "Steers the color of the sure sheet assignment area")
        , unsureSheetColor_("color::area::unsureSheetColor", "Steers the color of the unsure sheet assignment area")
        , sureHelixPointColor_(
              "color::point::sureHelixPointColor", "Steers the color of the sure helix assignment points")
        , unsureHelixPointColor_(
              "color::point::unsureHelixPointColor", "Steers the color of the unsure helix assignment points")
        , sureSheetPointColor_(
              "color::point::sureSheetPointColor", "Steers the color of the sure sheet assignment points")
        , unsureSheetPointColor_(
              "color::point::unsureSheetPointColor", "Steers the color of the unsure sheet assignment points")
        , boundingBoxColor_("color::boundingBoxColor", "Color of the bounding box")
        , passthroughShader_(nullptr)
        , positionBuffer_(nullptr)
        , vao_(0)
        , font_(core::utility::SDFFont::PRESET_EVOLVENTA_SANS, core::utility::SDFFont::RENDERMODE_FILL) {

    moleculeInSlot_.SetCompatibleCall<protein_calls::MolecularDataCallDescription>();
    this->MakeSlotAvailable(&moleculeInSlot_);

    showBoundingBoxParam_.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&showBoundingBoxParam_);

    pointSizeParam_.SetParameter(new core::param::FloatParam(10.0f, 0.01f, 1000.0f));
    this->MakeSlotAvailable(&pointSizeParam_);

    pointColorParam_.SetParameter(new core::param::ColorParam(0.0f, 0.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&pointColorParam_);

    sureHelixColor_.SetParameter(new core::param::ColorParam(0.890196f, 0.10196f, 0.109804f, 1.0f));
    this->MakeSlotAvailable(&sureHelixColor_);

    unsureHelixColor_.SetParameter(
        new core::param::ColorParam(0.5f * 0.890196f, 0.5f * 0.10196f, 0.5f * 0.109804f, 1.0f));
    this->MakeSlotAvailable(&unsureHelixColor_);

    sureSheetColor_.SetParameter(new core::param::ColorParam(0.1216f, 0.470588f, 0.70588f, 1.0f));
    this->MakeSlotAvailable(&sureSheetColor_);

    unsureSheetColor_.SetParameter(
        new core::param::ColorParam(0.5f * 0.1216f, 0.5f * 0.470588f, 0.5f * 0.70588f, 1.0f));
    this->MakeSlotAvailable(&unsureSheetColor_);

    sureHelixPointColor_.SetParameter(new core::param::ColorParam(1.0f, 1.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&sureHelixPointColor_);

    unsureHelixPointColor_.SetParameter(new core::param::ColorParam(0.5f, 0.5f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&unsureHelixPointColor_);

    sureSheetPointColor_.SetParameter(new core::param::ColorParam(0.0f, 1.0f, 1.0f, 1.0f));
    this->MakeSlotAvailable(&sureSheetPointColor_);

    unsureSheetPointColor_.SetParameter(new core::param::ColorParam(0.0f, 0.5f, 0.5f, 1.0f));
    this->MakeSlotAvailable(&unsureSheetPointColor_);

    boundingBoxColor_.SetParameter(new core::param::ColorParam(0.8f, 0.8f, 0.8f, 1.0f));
    this->MakeSlotAvailable(&boundingBoxColor_);

    // these bounds do not change so we can set them here
    bounds_.SetBoundingBox(-180.0f, -180.0f, 0.0f, 180.0f, 180.0f, 0.0f);
    bounds_.SetClipBox(-180.0f, -180.0f, 0.0f, 180.0f, 180.0f, 0.0f);
}

RamachandranPlot::~RamachandranPlot() {
    this->Release();
}

bool RamachandranPlot::create() {
    // be careful here, the point order is important to be able to draw all of this as GL_TRIANGLE_FAN

    // light sheets
    semiSheetPolygons_.clear();
    // lower left
    std::vector<glm::vec2> poly;
    poly.emplace_back(glm::vec2(-180.0f, -163.8f));
    poly.emplace_back(glm::vec2(-75.6f, -163.8f));
    poly.emplace_back(glm::vec2(-46.9f, -180.0f));
    poly.emplace_back(glm::vec2(-180.0f, -180.0f));
    semiSheetPolygons_.push_back(poly);

    // upper left light
    poly.clear();
    poly.emplace_back(glm::vec2(-180.0f, 42.9f));
    poly.emplace_back(glm::vec2(-140.8f, 16.1f));
    poly.emplace_back(glm::vec2(-86.0f, 16.1f));
    poly.emplace_back(glm::vec2(-74.3f, 45.6f));
    poly.emplace_back(glm::vec2(-74.3f, 72.5f));
    poly.emplace_back(glm::vec2(-44.3f, 102.0f));
    poly.emplace_back(glm::vec2(-44.3f, 161.1f));
    poly.emplace_back(glm::vec2(-46.9f, 179.9f));
    poly.emplace_back(glm::vec2(-180.0f, 180.0f));
    semiSheetPolygons_.push_back(poly);

    // sheet central
    sureSheetPolygons_.clear();
    poly.clear();
    poly.emplace_back(glm::vec2(-70.4f, 91.3f));
    poly.emplace_back(glm::vec2(-54.7f, 112.8f));
    poly.emplace_back(glm::vec2(-54.7f, 173.2f));
    poly.emplace_back(glm::vec2(-136.9f, 173.2f));
    poly.emplace_back(glm::vec2(-136.9f, 155.8f));
    poly.emplace_back(glm::vec2(-156.5f, 135.6f));
    poly.emplace_back(glm::vec2(-156.5f, 91.3f));
    sureSheetPolygons_.push_back(poly);

    // light helices
    semiHelixPolygons_.clear();
    // upper right (left-handed helices)
    poly.clear();
    poly.emplace_back(glm::vec2(62.6f, 14.7f));
    poly.emplace_back(glm::vec2(62.6f, 96.7f));
    poly.emplace_back(glm::vec2(45.6f, 79.2f));
    poly.emplace_back(glm::vec2(45.6f, 26.8f));
    semiHelixPolygons_.push_back(poly);

    // central left (this polygon cannot be drawn as single triangle fan)
    // the commented out coordinates are the original vertices of the polygon
    poly.clear();
    // poly.emplace_back(glm::vec2(-180.0f, -71.1f)); // 0
    // poly.emplace_back(glm::vec2(-180.0f, -34.9f)); // 1
    // poly.emplace_back(glm::vec2(-164.3f, -42.9f)); // 2
    // poly.emplace_back(glm::vec2(-133.0f, -42.9f)); // 3
    // poly.emplace_back(glm::vec2(-109.4f, -32.2f)); // 4
    // poly.emplace_back(glm::vec2(-106.9f, -21.4f)); // 5
    // poly.emplace_back(glm::vec2(-44.3f, -21.4f)); // 6
    // poly.emplace_back(glm::vec2(-44.3f, -71.1f)); // 7
    poly.emplace_back(glm::vec2(-180.0f, -71.1f)); // 0
    poly.emplace_back(glm::vec2(-180.0f, -34.9f)); // 1
    poly.emplace_back(glm::vec2(-164.3f, -42.9f)); // 2
    poly.emplace_back(glm::vec2(-44.3f, -71.1f));  // 7
    semiHelixPolygons_.push_back(poly);
    poly.clear();
    poly.emplace_back(glm::vec2(-44.3f, -71.1f));  // 7
    poly.emplace_back(glm::vec2(-164.3f, -42.9f)); // 2
    poly.emplace_back(glm::vec2(-133.0f, -42.9f)); // 3
    poly.emplace_back(glm::vec2(-109.4f, -32.2f)); // 4
    poly.emplace_back(glm::vec2(-106.9f, -21.4f)); // 5
    poly.emplace_back(glm::vec2(-44.3f, -21.4f));  // 6
    semiHelixPolygons_.push_back(poly);

    // helix central
    sureHelixPolygons_.clear();
    poly.clear();
    poly.emplace_back(glm::vec2(-54.7f, -60.4f));
    poly.emplace_back(glm::vec2(-54.7f, -40.2f));
    poly.emplace_back(glm::vec2(-100.4f, -40.2f));
    poly.emplace_back(glm::vec2(-123.9f, -51.0f));
    poly.emplace_back(glm::vec2(-156.5f, -51.0f));
    poly.emplace_back(glm::vec2(-156.5f, -60.4f));
    sureHelixPolygons_.push_back(poly);

    // setup font
    if (!font_.Initialise(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>())) {
        return false;
    }
    font_.SetBatchDrawMode(true);

    // create shaders
    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());
    try {
        passthroughShader_ = core::utility::make_glowl_shader("passthrough", shader_options,
            "protein_gl/render2d/passthrough.vert.glsl", "protein_gl/render2d/passthrough.frag.glsl");
        stippleShader_ = core::utility::make_glowl_shader("stipple", shader_options,
            "protein_gl/render2d/linestipple.vert.glsl", "protein_gl/render2d/linestipple.frag.glsl");
    } catch (std::exception& e) {
        core::utility::log::Log::DefaultLog.WriteError(("RamachandranPlot: " + std::string(e.what())).c_str());
        return false;
    }

    // create buffers and VAOs
    positionBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    colorBuffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);

    positionBuffer_->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    colorBuffer_->bind();
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    return true;
}

void RamachandranPlot::release() {
    // intentionally empty
}

bool RamachandranPlot::GetExtents(mmstd_gl::CallRender2DGL& call) {
    call.AccessBoundingBoxes() = bounds_;

    const auto mol = moleculeInSlot_.CallAs<protein_calls::MolecularDataCall>();
    if (mol != nullptr && ((*mol)(protein_calls::MolecularDataCall::CallForGetExtent))) {
        call.SetTimeFramesCount(mol->FrameCount());
    } else {
        call.SetTimeFramesCount(1);
    }

    return true;
}

bool RamachandranPlot::Render(mmstd_gl::CallRender2DGL& call) {

    protein_calls::MolecularDataCall* mol = moleculeInSlot_.CallAs<protein_calls::MolecularDataCall>();
    if (mol == nullptr) {
        return false;
    }

    mol->SetFrameID(static_cast<unsigned int>(call.Time()), true);
    if (!(*mol)(protein_calls::MolecularDataCall::CallForGetExtent)) {
        return false;
    }
    mol->SetFrameID(static_cast<unsigned int>(call.Time()), true);
    if (!(*mol)(protein_calls::MolecularDataCall::CallForGetData)) {
        return false;
    }

    computeDihedralAngles(*mol);
    computePolygonPositions();
    colorBuffer_->rebuffer(pointColors_);

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    core::view::Camera cam = call.GetCamera();
    const auto view = cam.getViewMatrix();
    const auto proj = cam.getProjectionMatrix();
    const auto mvp = proj * view;

    const glm::vec3 semiSheetColor = glm::make_vec3(unsureSheetColor_.Param<core::param::ColorParam>()->Value().data());
    const glm::vec3 sureSheetColor = glm::make_vec3(sureSheetColor_.Param<core::param::ColorParam>()->Value().data());
    const glm::vec3 semiHelixColor = glm::make_vec3(unsureHelixColor_.Param<core::param::ColorParam>()->Value().data());
    const glm::vec3 sureHelixColor = glm::make_vec3(sureHelixColor_.Param<core::param::ColorParam>()->Value().data());
    const glm::vec3 bbColor = glm::make_vec3(boundingBoxColor_.Param<core::param::ColorParam>()->Value().data());
    const glm::vec3 vertexColor = glm::make_vec3(pointColorParam_.Param<core::param::ColorParam>()->Value().data());
    glBindVertexArray(vao_);
    passthroughShader_->use();
    passthroughShader_->setUniform("global_color", semiSheetColor);
    passthroughShader_->setUniform("use_per_vertex_color", false);
    passthroughShader_->setUniform("mvp", mvp);
    for (auto& poly : semiSheetPolygons_) {
        positionBuffer_->rebuffer(poly);
        glDrawArrays(GL_TRIANGLE_FAN, 0, poly.size());
    }
    passthroughShader_->setUniform("global_color", sureSheetColor);
    for (auto& poly : sureSheetPolygons_) {
        positionBuffer_->rebuffer(poly);
        glDrawArrays(GL_TRIANGLE_FAN, 0, poly.size());
    }
    passthroughShader_->setUniform("global_color", semiHelixColor);
    for (auto& poly : semiHelixPolygons_) {
        positionBuffer_->rebuffer(poly);
        glDrawArrays(GL_TRIANGLE_FAN, 0, poly.size());
    }
    passthroughShader_->setUniform("global_color", sureHelixColor);
    for (auto& poly : sureHelixPolygons_) {
        positionBuffer_->rebuffer(poly);
        glDrawArrays(GL_TRIANGLE_FAN, 0, poly.size());
    }

    // draw coordinate system using the stipple shader
    const std::vector<glm::vec2> coordinateSysVec = {
        glm::vec2(0.0f, -180.0f), glm::vec2(0.0f, 180.0f), glm::vec2(-180.0f, 0.0f), glm::vec2(180.0f, 0.0f)};
    stippleShader_->use();
    stippleShader_->setUniform("global_color", bbColor);
    stippleShader_->setUniform("mvp", mvp);
    stippleShader_->setUniform("factor", 2.0f);
    stippleShader_->setUniform("viewport", glm::vec2(call.GetViewResolution()));
    positionBuffer_->rebuffer(coordinateSysVec);
    glDrawArrays(GL_LINES, 0, coordinateSysVec.size());

    // draw vertices
    passthroughShader_->use();
    passthroughShader_->setUniform("use_per_vertex_color", true);
    passthroughShader_->setUniform("mvp", mvp);
    glPointSize(this->pointSizeParam_.Param<core::param::FloatParam>()->Value());
    positionBuffer_->rebuffer(angles_);
    glDrawArrays(GL_POINTS, 0, angles_.size());

    // TODO draw selected point in different color

    // draw bounding box
    if (showBoundingBoxParam_.Param<core::param::BoolParam>()->Value()) {
        const std::vector<glm::vec2> bbvec = {glm::vec2(-180.0f, -180.0f), glm::vec2(-180.0f, 180.0f),
            glm::vec2(180.0f, 180.0f), glm::vec2(180.0f, -180.0f)};
        passthroughShader_->setUniform("global_color", bbColor);
        passthroughShader_->setUniform("use_per_vertex_color", false);
        positionBuffer_->rebuffer(bbvec);
        glDrawArrays(GL_LINE_LOOP, 0, bbvec.size());
    }

    // draw legend
    auto bbcol = boundingBoxColor_.Param<core::param::ColorParam>()->Value();
    font_.ClearBatchDrawCache();
    font_.SetSmoothMode(true);
    constexpr float font_size = 20.0f;
    // names
    font_.DrawString(
        mvp, bbcol.data(), 0.0f, -200.0f, font_size, false, "phi", core::utility::SDFFont::ALIGN_CENTER_TOP);
    font_.DrawString(
        mvp, bbcol.data(), -200.0f, 0.0f, font_size, false, "psi", core::utility::SDFFont::ALIGN_RIGHT_MIDDLE);
    // zeros
    font_.DrawString(
        mvp, bbcol.data(), 0.0f, -180.0f, font_size, false, "0°", core::utility::SDFFont::ALIGN_CENTER_TOP);
    font_.DrawString(
        mvp, bbcol.data(), -180.0f, 0.0f, font_size, false, "0°", core::utility::SDFFont::ALIGN_RIGHT_MIDDLE);
    // 180
    font_.DrawString(
        mvp, bbcol.data(), 180.0f, -180.0f, font_size, false, "180°", core::utility::SDFFont::ALIGN_RIGHT_TOP);
    font_.DrawString(
        mvp, bbcol.data(), -180.0f, 180.0f, font_size, false, "180°", core::utility::SDFFont::ALIGN_RIGHT_TOP);
    // -180
    font_.DrawString(
        mvp, bbcol.data(), -180.0f, -180.0f, font_size, false, "-180°", core::utility::SDFFont::ALIGN_RIGHT_BOTTOM);
    font_.DrawString(
        mvp, bbcol.data(), -180.0f, -180.0f, font_size, false, "-180°", core::utility::SDFFont::ALIGN_LEFT_TOP);
    font_.BatchDrawString(mvp);

    glUseProgram(0);
    glBindVertexArray(0);

    //glEnable(GL_DEPTH_TEST);

    return true;
}

bool RamachandranPlot::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    // TODO selection
    return false;
}

bool RamachandranPlot::OnMouseMove(double x, double y) {
    // Nothing to see here...
    return false;
}

void RamachandranPlot::computeDihedralAngles(protein_calls::MolecularDataCall& mol) {
    angles_.clear();

    unsigned int firstAAIdx = 0;
    unsigned int lastAAIdx = 0;

    int molCount = mol.MoleculeCount();
    angles_.resize(mol.ResidueCount());
    pointStates_.resize(mol.ResidueCount(), protein_calls::RamachandranDataCall::PointState::NONE);

    // the 5 needed positions for the dihedral angles
    glm::vec3 prevCPos;
    glm::vec3 NPos;
    glm::vec3 CaPos;
    glm::vec3 CPos;
    glm::vec3 nextNPos;

    for (int molIdx = 0; molIdx < molCount; molIdx++) {
        protein_calls::MolecularDataCall::Molecule chain = mol.Molecules()[molIdx];

        if (mol.Residues()[chain.FirstResidueIndex()]->Identifier() !=
            protein_calls::MolecularDataCall::Residue::AMINOACID) {
            continue;
        }

        firstAAIdx = chain.FirstResidueIndex();
        lastAAIdx = firstAAIdx + chain.ResidueCount();

        for (unsigned int aaIdx = firstAAIdx; aaIdx < lastAAIdx; aaIdx++) {

            protein_calls::MolecularDataCall::AminoAcid* acid = nullptr;

            if (mol.Residues()[aaIdx]->Identifier() == protein_calls::MolecularDataCall::Residue::AMINOACID) {
                acid = (protein_calls::MolecularDataCall::AminoAcid*) (mol.Residues()[aaIdx]);
            } else {
                angles_[aaIdx].x = -1000.0f;
                angles_[aaIdx].y = -1000.0f;
                continue;
            }

            float phi = 0.0f;
            float psi = 0.0f;

            // get all relevant atom positions of the current amino acid
            NPos = glm::make_vec3(&mol.AtomPositions()[acid->NIndex() * 3]);
            CaPos = glm::make_vec3(&mol.AtomPositions()[acid->CAlphaIndex() * 3]);
            CPos = glm::make_vec3(&mol.AtomPositions()[acid->CCarbIndex() * 3]);

            // get all relevant atom positions of the last and the next amino acid

            // is this the first residue?
            if (aaIdx == chain.FirstResidueIndex()) {
                // we have no prevCPos
                phi = -1000.0f;
            } else {
                acid = (protein_calls::MolecularDataCall::AminoAcid*) (mol.Residues()[aaIdx - 1]);
                if (acid != nullptr) {
                    prevCPos = glm::make_vec3(&mol.AtomPositions()[acid->CCarbIndex() * 3]);
                } else {
                    phi = -1000.0f;
                }
            }

            if (aaIdx == chain.FirstResidueIndex() + chain.ResidueCount() - 1) {
                // we have no nextNPos
                psi = -1000.0f;
            } else {
                acid = (protein_calls::MolecularDataCall::AminoAcid*) (mol.Residues()[aaIdx + 1]);
                if (acid != nullptr) {
                    nextNPos = glm::make_vec3(&mol.AtomPositions()[acid->NIndex() * 3]);
                } else {
                    psi = -1000.0f;
                }
            }

            // if nothing speaks against it, compute the angles
            if (phi > -500.0f) {
                phi = dihedralAngle(prevCPos, NPos, CaPos, CPos);
            }
            if (psi > -500.0f) {
                psi = dihedralAngle(NPos, CaPos, CPos, nextNPos);
            }

            angles_[aaIdx].x = phi;
            angles_[aaIdx].y = psi;
        }
    }
}

void RamachandranPlot::computePolygonPositions() {
    pointStates_ = std::vector<protein_calls::RamachandranDataCall::PointState>(
        pointStates_.size(), protein_calls::RamachandranDataCall::PointState::NONE);
    const glm::vec3 sureHelixColor =
        glm::make_vec3(sureHelixPointColor_.Param<core::param::ColorParam>()->Value().data());
    const glm::vec3 sureSheetColor =
        glm::make_vec3(sureSheetPointColor_.Param<core::param::ColorParam>()->Value().data());
    const glm::vec3 unsureHelixColor =
        glm::make_vec3(unsureHelixPointColor_.Param<core::param::ColorParam>()->Value().data());
    const glm::vec3 unsureSheetColor =
        glm::make_vec3(unsureSheetPointColor_.Param<core::param::ColorParam>()->Value().data());
    const glm::vec3 globalColor = glm::make_vec3(pointColorParam_.Param<core::param::ColorParam>()->Value().data());
    pointColors_ = std::vector<glm::vec3>(pointStates_.size(), globalColor);
    for (int i = 0; i < static_cast<int>(angles_.size()); ++i) {
        const auto& pos = angles_[i];

        if (pos.x < -500.0f || pos.y < -500.0f)
            continue;

        float posx = pos.x;
        float posy = pos.y;

        bool semiHelixState = false;
        bool semiSheetState = false;
        bool trueHelixState = false;
        bool trueSheetState = false;

        // test against sheets
        for (auto& semiSheet : semiSheetPolygons_) {
            semiSheetState = locateInPolygon(semiSheet, pos) ? true : semiSheetState;
        }

        for (auto& sureSheet : sureSheetPolygons_) {
            trueSheetState = locateInPolygon(sureSheet, pos) ? true : trueSheetState;
        }

        // test against helices
        for (auto& semiHelix : semiHelixPolygons_) {
            semiHelixState = locateInPolygon(semiHelix, pos) ? true : semiHelixState;
        }

        for (auto& sureHelix : sureHelixPolygons_) {
            trueHelixState = locateInPolygon(sureHelix, pos) ? true : trueHelixState;
        }

        if (semiSheetState) {
            pointStates_[i] = protein_calls::RamachandranDataCall::PointState::UNSURE_BETA;
            pointColors_[i] = unsureSheetColor;
        }
        if (trueSheetState) {
            pointStates_[i] = protein_calls::RamachandranDataCall::PointState::SURE_BETA;
            pointColors_[i] = sureSheetColor;
        }
        if (semiHelixState) {
            pointStates_[i] = protein_calls::RamachandranDataCall::PointState::UNSURE_ALPHA;
            pointColors_[i] = unsureHelixColor;
        }
        if (trueHelixState) {
            pointStates_[i] = protein_calls::RamachandranDataCall::PointState::SURE_ALPHA;
            pointColors_[i] = sureHelixColor;
        }
    }
}

bool RamachandranPlot::locateInPolygon(const std::vector<glm::vec2>& polyVector, const glm::vec2 inputPos) const {
    const int nvert = static_cast<int>(polyVector.size());
    int i, j, c = 0;
    const float testx = inputPos.x;
    const float testy = inputPos.y;
    for (i = 0, j = nvert - 1; i < nvert; j = i++) {
        const float vertxi = polyVector[i].x;
        const float vertxj = polyVector[j].x;
        const float vertyi = polyVector[i].y;
        const float vertyj = polyVector[j].y;

        if (((vertyi > testy) != (vertyj > testy)) &&
            (testx < (vertxj - vertxi) * (testy - vertyi) / (vertyj - vertyi) + vertxi)) {
            c = !c;
        }
    }
    return (c != 0);
}

float RamachandranPlot::dihedralAngle(
    const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, const glm::vec3& v4) const {
    /*
     *  Code from https://github.com/biopython/biopython/blob/9fa26d5efb38e7398f82f4b4e028ca84b7eeaa58/Bio/PDB/Vector.py
     */
    const auto ab = v1 - v2;
    const auto cb = v3 - v2;
    const auto db = v4 - v3;
    const auto u = glm::cross(ab, cb);
    const auto v = glm::cross(db, cb);
    const auto w = glm::cross(u, v);
    float result = this->angle(u, v);
    if (this->angle(cb, w) > 0.001f) {
        result = -result;
    }
    return glm::degrees(result);
}

float RamachandranPlot::angle(const glm::vec3& v1, const glm::vec3& v2) const {
    const auto n1 = glm::length(v1);
    const auto n2 = glm::length(v2);
    auto c = glm::dot(v1, v2) / (n1 * n2);
    c = glm::clamp(c, -1.0f, 1.0f);
    c = std::acos(c);
    return c;
}
