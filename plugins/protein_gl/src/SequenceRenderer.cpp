#include "SequenceRenderer.h"
#include "stdafx.h"

#include "RuntimeConfig.h"
#include "SequenceRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/utility/FileUtils.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore_gl/utility/RenderUtils.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "protein_calls/ProteinColor.h"
#include "protein_calls/ProteinHelpers.h"
#include "vislib/math/Rectangle.h"
#include "vislib/sys/BufferedFile.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"
#include <GL/glu.h>
#include <math.h>


using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_gl;
using namespace vislib_gl::graphics::gl;
using namespace megamol::protein_calls;
using megamol::core::utility::log::Log;

/*
 * SequenceRenderer::SequenceRenderer (CTOR)
 */
SequenceRenderer::SequenceRenderer(void)
        : core_gl::view::Renderer2DModuleGL()
        , dataCallerSlot("getData", "Connects the sequence diagram rendering with data storage.")
        , bindingSiteCallerSlot("getBindingSites", "Connects the sequence diagram rendering with binding site storage.")
        , resSelectionCallerSlot(
              "getResSelection", "Connects the sequence diagram rendering with residue selection storage.")
        , resCountPerRowParam("ResiduesPerRow", "The number of residues per row")
        , colorTableFileParam("ColorTableFilename", "The filename of the color table.")
        , toggleKeyParam("ToggleKeyDrawing", "Turns the drawing of the binding site key/legend on and off.")
        , clearResSelectionParam(
              "clearResidueSelection", "Clears the current selection (everything will be deselected).")
        , dataPrepared(false)
        , texture_shader_(nullptr)
        , passthrough_shader_(nullptr)
        , position_buffer_(nullptr)
        , color_buffer_(nullptr)
        , texposition_buffer_(nullptr)
        , pass_vao_(0)
        , tex_vao_(0)
        , atomCount(0)
        , bindingSiteCount(0)
        , resCount(0)
        , resCols(0)
        , resRows(0)
        , rowHeight(3.0f)
#ifndef USE_SIMPLE_FONT
        , theFont(FontInfo_Verdana)
#endif // USE_SIMPLE_FONT
        , font_(core::utility::SDFFont::PRESET_ROBOTO_SANS, utility::SDFFont::RENDERMODE_FILL)
        , marker_textures_{}
        , resSelectionCall(nullptr)
        , leftMouseDown(false) {

    // molecular data caller slot
    this->dataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);

    // binding site data caller slot
    this->bindingSiteCallerSlot.SetCompatibleCall<BindingSiteCallDescription>();
    this->MakeSlotAvailable(&this->bindingSiteCallerSlot);

    // residue selection caller slot
    this->resSelectionCallerSlot.SetCompatibleCall<ResidueSelectionCallDescription>();
    this->MakeSlotAvailable(&this->resSelectionCallerSlot);

    // param slot for number of residues per row
    this->resCountPerRowParam.SetParameter(new param::IntParam(50, 1));
    this->MakeSlotAvailable(&this->resCountPerRowParam);

    // fill color table with default values and set the filename param
    this->colorTableFileParam.SetParameter(
        new param::FilePathParam("colors.txt", param::FilePathParam::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&this->colorTableFileParam);
    ProteinColor::ReadColorTableFromFile(
        this->colorTableFileParam.Param<param::FilePathParam>()->Value(), this->colorTable);

    // param slot for key toggling
    this->toggleKeyParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->toggleKeyParam);

    // param slot for key toggling
    this->clearResSelectionParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_C));
    this->MakeSlotAvailable(&this->clearResSelectionParam);
}

/*
 * SequenceRenderer::~SequenceRenderer (DTOR)
 */
SequenceRenderer::~SequenceRenderer(void) {
    this->Release();
}

std::vector<std::string> SequenceRenderer::requested_lifetime_resources() {
    auto lr = Module::requested_lifetime_resources();
    lr.emplace_back("RuntimeConfig");
    return lr;
}


/*
 * SequenceRenderer::create
 */
bool SequenceRenderer::create() {
    this->LoadTexture("stride-white.png");
    this->LoadTexture("stride-coil.png");
    this->LoadTexture("stride-sheet.png");
    this->LoadTexture("stride-arrow.png");
    this->LoadTexture("stride-helix-left.png");
    this->LoadTexture("stride-helix2.png");
    this->LoadTexture("stride-helix-right.png");

    try {
        auto const shdr_options = msf::ShaderFactoryOptionsOpenGL(instance()->GetShaderPaths());
        passthrough_shader_ = core::utility::make_glowl_shader("passthr", shdr_options,
            std::filesystem::path("protein_gl/render2d/passthrough.vert.glsl"),
            std::filesystem::path("protein_gl/render2d/passthrough.frag.glsl"));

        texture_shader_ = core::utility::make_glowl_shader("texture", shdr_options,
            std::filesystem::path("protein_gl/render2d/bwtexture.vert.glsl"),
            std::filesystem::path("protein_gl/render2d/bwtexture.frag.glsl"));

    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "[SequenceRenderer] %s", ex.what());
        return false;
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "[SequenceRenderer] Unable to compile shader: Unknown exception: %s", ex.what());
        return false;
    }

    position_buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    color_buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    texposition_buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &pass_vao_);
    glBindVertexArray(pass_vao_);

    position_buffer_->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    color_buffer_->bind();
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glGenVertexArrays(1, &tex_vao_);
    glBindVertexArray(tex_vao_);

    texposition_buffer_->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);

    if (!font_.Initialise(GetCoreInstance())) {
        core::utility::log::Log::DefaultLog.WriteError(
            "[SequenceRenderer]: The font rendering could not be initialized");
        return false;
    }
    font_.SetBatchDrawMode(true);

    return true;
}

/*
 * SequenceRenderer::release
 */
void SequenceRenderer::release() {}

bool SequenceRenderer::GetExtents(core_gl::view::CallRender2DGL& call) {
    // check molecular data
    MolecularDataCall* mol = this->dataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == nullptr)
        return false;
    if (!(*mol)(MolecularDataCall::CallForGetData))
        return false;

    // get pointer to BindingSiteCall
    BindingSiteCall* bs = this->bindingSiteCallerSlot.CallAs<BindingSiteCall>();
    if (bs != nullptr) {
        // execute the call
        if (!(*bs)(BindingSiteCall::CallForGetData)) {
            bs = nullptr;
        }
    }

    // check whether the number of residues per row or the number of atoms was changed
    if (this->resCountPerRowParam.IsDirty() || this->atomCount != mol->AtomCount()) {
        this->dataPrepared = false;
    }

    // check whether the number of binding sites has changed
    if (bs && this->bindingSiteCount != bs->GetBindingSiteCount()) {
        this->dataPrepared = false;
    }

    // prepare the data
    if (!this->dataPrepared) {
        unsigned int oldRowHeight = (unsigned int)this->rowHeight;
        this->dataPrepared = this->PrepareData(mol, bs);
        if (oldRowHeight != this->rowHeight) {
            this->dataPrepared = this->PrepareData(mol, bs);
        }

        if (this->dataPrepared) {
            // the data has been prepared for the current number of residues per row
            this->resCountPerRowParam.ResetDirty();
            // store the number of atoms for which the data was prepared
            this->atomCount = mol->AtomCount();
            // store the number of binding sites
            bs ? this->bindingSiteCount = bs->GetBindingSiteCount() : this->bindingSiteCount = 0;
        }
    }

    // set the bounding box
    // call.SetBoundingBox( 0.0f, 0.0f, static_cast<float>(this->resCols), static_cast<float>(this->resRows));
    call.AccessBoundingBoxes().SetBoundingBox(
        0.0f, -static_cast<float>(this->resRows) * this->rowHeight, 0, static_cast<float>(this->resCols), 0.0f, 0);
    const auto bb = call.AccessBoundingBoxes().BoundingBox();
    call.AccessBoundingBoxes().SetClipBox(
        bb.Left() - 1000.0f, bb.Bottom() - 1000.0f, bb.Back(), bb.Right() + 1000.0f, bb.Top() + 1000.0f, bb.Front());

    return true;
}


/*
 * SequenceRenderer::Render
 */
bool SequenceRenderer::Render(core_gl::view::CallRender2DGL& call) {
    // get pointer to MolecularDataCall
    MolecularDataCall* mol = this->dataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == nullptr)
        return false;
    // execute the call
    if (!(*mol)(MolecularDataCall::CallForGetData))
        return false;

    cam_ = call.GetCamera();
    screen_ = call.GetViewResolution();

    std::vector<glm::vec2> positions;
    std::vector<glm::vec3> colors;
    std::vector<glm::vec4> texture_data;

    font_.ClearBatchDrawCache();
    font_.SetSmoothMode(true);

    const auto proj = call.GetCamera().getProjectionMatrix();
    const auto view = call.GetCamera().getViewMatrix();
    const auto mvp = proj * view;

    this->resSelectionCall = this->resSelectionCallerSlot.CallAs<ResidueSelectionCall>();
    if (this->resSelectionCall != nullptr) {
        (*this->resSelectionCall)(ResidueSelectionCall::CallForGetSelection);
        if (this->clearResSelectionParam.IsDirty()) {
            this->resSelectionCall->GetSelectionPointer()->Clear();
            this->clearResSelectionParam.ResetDirty();
        }
        unsigned int cnt = 0;
        vislib::Array<ResidueSelectionCall::Residue>* resSelPtr = resSelectionCall->GetSelectionPointer();
        for (unsigned int i = 0; i < aminoAcidIndexStrings.size(); i++) {
            for (unsigned int j = 0; j < aminoAcidIndexStrings[i].size(); j++) {
                this->selection[cnt] = false;
                if (resSelPtr) {
                    // loop over selections and try to match wit the current amino acid
                    for (unsigned int k = 0; k < resSelPtr->Count(); k++) {
                        if ((*resSelPtr)[k].chainID == aminoAcidIndexStrings[i][j].first &&
                            (*resSelPtr)[k].resNum == aminoAcidIndexStrings[i][j].second) {
                            this->selection[cnt] = true;
                        }
                    }
                }
                cnt++;
            }
        }
    }

    // read and update the color table, if necessary
    if (this->colorTableFileParam.IsDirty()) {
        ProteinColor::ReadColorTableFromFile(
            this->colorTableFileParam.Param<param::FilePathParam>()->Value(), this->colorTable);
        this->colorTableFileParam.ResetDirty();
    }

    if (this->dataPrepared) {
        glDisable(GL_CULL_FACE);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // get the text color (inverse background color)
        float bgColor[4];
        float fgColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
        for (unsigned int i = 0; i < 4; i++) {
            fgColor[i] -= bgColor[i];
        }

        glDisable(GL_DEPTH_TEST);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glLoadMatrixf(glm::value_ptr(proj));
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glLoadMatrixf(glm::value_ptr(view));

        // temporary variables and constants
        const float eps = 0.0f;
        // draw tiles for structure
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_BLEND);
        glActiveTexture(GL_TEXTURE0);
        for (unsigned int i = 0; i < this->resIndex.size(); i++) {
            marker_textures_[0]->bindTexture();
            if (this->resSecStructType[i] == MolecularDataCall::SecStructure::TYPE_HELIX) {
                glColor3f(1.0f, 0.0f, 0.0f);
                if (i > 0 && this->resSecStructType[i - 1] != this->resSecStructType[i]) {
                    marker_textures_[4]->bindTexture();
                } else if ((i + 1) < this->resIndex.size() &&
                           this->resSecStructType[i + 1] != this->resSecStructType[i]) {
                    marker_textures_[6]->bindTexture();
                } else {
                    marker_textures_[5]->bindTexture();
                }
            } else if (this->resSecStructType[i] == MolecularDataCall::SecStructure::TYPE_SHEET) {
                glColor3f(0.0f, 0.0f, 1.0f);
                if ((i + 1) < this->resIndex.size() && this->resSecStructType[i + 1] != this->resSecStructType[i]) {
                    marker_textures_[3]->bindTexture();
                } else {
                    marker_textures_[2]->bindTexture();
                }
            } else if (this->resSecStructType[i] == MolecularDataCall::SecStructure::TYPE_TURN) {
                glColor3f(1.0f, 1.0f, 0.0f);
                marker_textures_[1]->bindTexture();
            } else { // TYPE_COIL
                glColor3f(0.5f, 0.5f, 0.5f);
                marker_textures_[1]->bindTexture();
            }
            glBegin(GL_QUADS);
            glTexCoord2f(eps, eps);
            glVertex2f(this->vertices[i].x, -this->vertices[i].y);
            glTexCoord2f(eps, 1.0f - eps);
            glVertex2f(this->vertices[i].x, -this->vertices[i].y - 1.0f);
            glTexCoord2f(1.0f - eps, 1.0f - eps);
            glVertex2f(this->vertices[i].x + 1.0f, -this->vertices[i].y - 1.0f);
            glTexCoord2f(1.0f - eps, eps);
            glVertex2f(this->vertices[i].x + 1.0f, -this->vertices[i].y);
            glEnd();
        }
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_2D);

        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();

        positions.clear();
        colors.clear();
        positions.reserve(4 * bsIndices.size() + 4 * chainVertices.size());
        colors.reserve(4 * bsIndices.size() + 4 * chainVertices.size());

        // draw tiles for binding sites
        for (unsigned int i = 0; i < this->bsIndices.size(); i++) {
            positions.emplace_back(glm::vec2(this->bsVertices[i].x + 0.1f, -this->bsVertices[i].y));
            positions.emplace_back(glm::vec2(this->bsVertices[i].x + 0.1f, -this->bsVertices[i].y - 0.4f));
            positions.emplace_back(glm::vec2(this->bsVertices[i].x + 0.9f, -this->bsVertices[i].y - 0.4f));
            positions.emplace_back(glm::vec2(this->bsVertices[i].x + 0.9f, -this->bsVertices[i].y));
            std::vector<glm::vec3> cols(4, this->bsColors[this->bsIndices[i]]);
            colors.insert(colors.end(), cols.begin(), cols.end());
        }

        // draw tiles for chains
        for (unsigned int i = 0; i < this->chainVertices.size(); i++) {
            positions.emplace_back(glm::vec2(this->chainVertices[i].x, -this->chainVertices[i].y - 0.2f));
            positions.emplace_back(glm::vec2(this->chainVertices[i].x, -this->chainVertices[i].y - 0.3f));
            positions.emplace_back(glm::vec2(this->chainVertices[i].x + 1.0f, -this->chainVertices[i].y - 0.3f));
            positions.emplace_back(glm::vec2(this->chainVertices[i].x + 1.0f, -this->chainVertices[i].y - 0.2f));
            std::vector<glm::vec3> cols(4, this->chainColors[i]);
            colors.insert(colors.end(), cols.begin(), cols.end());
        }

        position_buffer_->rebuffer(positions);
        color_buffer_->rebuffer(colors);
        glBindVertexArray(pass_vao_);
        passthrough_shader_->use();
        passthrough_shader_->setUniform("mvp", mvp);
        passthrough_shader_->setUniform("use_per_vertex_color", true);

        glDrawArrays(GL_QUADS, 0, positions.size());

        glUseProgram(0);
        glBindVertexArray(0);

        // draw chain separators
        position_buffer_->rebuffer(this->chainSeparatorVertices);
        glBindVertexArray(pass_vao_);
        passthrough_shader_->use();
        passthrough_shader_->setUniform("mvp", mvp);
        passthrough_shader_->setUniform("global_color", glm::make_vec3(fgColor));
        passthrough_shader_->setUniform("use_per_vertex_color", false);
        glDisableVertexAttribArray(1); // we use the global color so we disable this temporally

        glDrawArrays(GL_LINES, 0, this->chainSeparatorVertices.size());

        glEnableVertexAttribArray(1);
        glUseProgram(0);
        glBindVertexArray(0);

        fgColor[3] = 1.0f;

        // labeling
        for (unsigned int i = 0; i < this->aminoAcidStrings.size(); i++) {
            for (unsigned int j = 0; j < (unsigned int)this->aminoAcidStrings[i].size(); j++) {
                // draw the one-letter amino acid code
                font_.DrawString(mvp, fgColor, static_cast<float>(j) + 0.5f,
                    -(static_cast<float>(i) * this->rowHeight + 1.0f), 1.0f, false,
                    this->aminoAcidStrings[i].substr(j, 1).c_str(), core::utility::SDFFont::ALIGN_CENTER_TOP);
                // draw the chain name and amino acid index
                std::string str(1, this->aminoAcidIndexStrings[i][j].first);
                str.append(" " + std::to_string(this->aminoAcidIndexStrings[i][j].second));
                font_.DrawString(mvp, fgColor, static_cast<float>(j) + 0.5f,
                    -(static_cast<float>(i) * this->rowHeight + 2.0f), 0.35f, false, str.c_str(),
                    core::utility::SDFFont::ALIGN_CENTER_TOP);
            }
        }

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);

        // draw legend / key
        if (this->toggleKeyParam.Param<param::BoolParam>()->Value()) {
            float fontSize = 1.0f;

            // draw binding site legend
            if (!this->bindingSiteNames.empty()) {
                font_.DrawString(mvp, fgColor, static_cast<float>(this->resCols) + 1.0f, -1.0f, fontSize, false,
                    "Binding Sites: ", core::utility::SDFFont::ALIGN_LEFT_TOP);
                for (unsigned int i = 0; i < this->bindingSiteNames.size(); i++) {
                    // draw the binding site names
                    font_.DrawString(mvp, &this->bsColors[i].x, static_cast<float>(this->resCols) + 1.0f,
                        -(static_cast<float>(i) * 2.0f + 2.0f), fontSize, false, this->bindingSiteNames[i].c_str(),
                        core::utility::SDFFont::ALIGN_LEFT_TOP);
                    font_.DrawString(view, proj, &this->bsColors[i].x, static_cast<float>(this->resCols) + 1.0f,
                        -(static_cast<float>(i) * 2.0f + 3.0f), fontSize * 0.5f, false,
                        this->bindingSiteDescription[i].c_str(), core::utility::SDFFont::ALIGN_LEFT_TOP);
                }
            }
            // draw diagram legend
            fontSize = 1.0f;
            font_.DrawString(
                mvp, fgColor, -0.5f, -0.5f, fontSize, false, "STRIDE", core::utility::SDFFont::ALIGN_RIGHT_MIDDLE);
            font_.DrawString(
                mvp, fgColor, -0.5f, -1.5f, fontSize, false, "Amino Acid", core::utility::SDFFont::ALIGN_RIGHT_MIDDLE);
            fontSize = 0.5f;
            font_.DrawString(
                mvp, fgColor, -0.5f, -2.25f, fontSize, false, "PDB", core::utility::SDFFont::ALIGN_RIGHT_MIDDLE);
            font_.DrawString(
                mvp, fgColor, -0.5f, -2.75f, fontSize, false, "Chain", core::utility::SDFFont::ALIGN_RIGHT_MIDDLE);
            if (!this->bindingSiteNames.empty()) {
                font_.DrawString(mvp, fgColor, -0.5f, -this->rowHeight, fontSize, false, "Binding Sites",
                    core::utility::SDFFont::ALIGN_RIGHT_BOTTOM);
            }
        }

        font_.BatchDrawString(mvp);

        // draw overlays for selected amino acids
        glm::vec2 pos;
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
        fgColor[3] = 0.3f;

        positions.clear();
        for (unsigned int i = 0; i < this->selection.size(); i++) {
            if (this->selection[i]) {
                pos = {static_cast<float>(i % this->resCols), floorf(static_cast<float>(i) / this->resCols) + 1.0f};
                positions.emplace_back(glm::vec2(pos.x, -this->rowHeight * (pos.y - 1.0f)));
                positions.emplace_back(glm::vec2(pos.x, -this->rowHeight * (pos.y) + 0.5f));
                positions.emplace_back(glm::vec2(pos.x + 1.0f, -this->rowHeight * (pos.y) + 0.5f));
                positions.emplace_back(glm::vec2(pos.x + 1.0f, -this->rowHeight * (pos.y - 1.0f)));
            }
        }
        position_buffer_->rebuffer(positions);

        glBindVertexArray(pass_vao_);
        passthrough_shader_->use();
        passthrough_shader_->setUniform("mvp", mvp);
        passthrough_shader_->setUniform("global_color", glm::make_vec3(fgColor));
        passthrough_shader_->setUniform("global_alpha", fgColor[3]);
        glDisableVertexAttribArray(1); // we use the global color so we disable this temporally

        glDrawArrays(GL_QUADS, 0, positions.size());

        glEnableVertexAttribArray(1);
        glUseProgram(0);
        glBindVertexArray(0);

        glDisable(GL_BLEND);

        // draw frames for selected amino acids
        fgColor[3] = 1.0f;
        positions.clear();

        for (unsigned int i = 0; i < this->selection.size(); i++) {
            if (this->selection[i]) {
                pos = {static_cast<float>(i % this->resCols), floorf(static_cast<float>(i) / this->resCols) + 1.0f};
                // left (only draw if left neighbor ist not selected)
                if (i == 0 || !this->selection[i - 1]) {
                    positions.emplace_back(glm::vec2(pos.x, -this->rowHeight * (pos.y - 1.0f)));
                    positions.emplace_back(glm::vec2(pos.x, -this->rowHeight * (pos.y) + 0.5f));
                }
                // bottom
                positions.emplace_back(glm::vec2(pos.x, -this->rowHeight * (pos.y) + 0.5f));
                positions.emplace_back(glm::vec2(pos.x + 1.0f, -this->rowHeight * (pos.y) + 0.5f));
                // right (only draw if right neighbor ist not selected)
                if (i == (this->selection.size() - 1) || !this->selection[i + 1]) {
                    positions.emplace_back(glm::vec2(pos.x + 1.0f, -this->rowHeight * (pos.y) + 0.5f));
                    positions.emplace_back(glm::vec2(pos.x + 1.0f, -this->rowHeight * (pos.y - 1.0f)));
                }
                // top
                positions.emplace_back(glm::vec2(pos.x + 1.0f, -this->rowHeight * (pos.y - 1.0f)));
                positions.emplace_back(glm::vec2(pos.x, -this->rowHeight * (pos.y - 1.0f)));
            }
        }
        position_buffer_->rebuffer(positions);

        glBindVertexArray(pass_vao_);
        passthrough_shader_->use();
        passthrough_shader_->setUniform("mvp", mvp);
        passthrough_shader_->setUniform("global_color", glm::make_vec3(fgColor));
        passthrough_shader_->setUniform("global_alpha", 1.0f);
        glDisableVertexAttribArray(1); // we use the global color so we disable this temporally

        glDrawArrays(GL_LINES, 0, positions.size());

        glEnableVertexAttribArray(1);
        glUseProgram(0);
        glBindVertexArray(0);

        // render mouse hover
        if (this->mousePos.x > -1.0f && this->mousePos.x < static_cast<float>(this->resCols) &&
            this->mousePos.y > 0.0f && this->mousePos.y < static_cast<float>(this->resRows + 1) &&
            this->mousePosResIdx > -1 && this->mousePosResIdx < (int)this->resCount) {
            positions = {glm::vec2(this->mousePos.x, -this->rowHeight * (this->mousePos.y - 1.0f)),
                glm::vec2(this->mousePos.x, -this->rowHeight * (this->mousePos.y) + 0.5f),
                glm::vec2(this->mousePos.x + 1.0f, -this->rowHeight * (this->mousePos.y) + 0.5f),
                glm::vec2(this->mousePos.x + 1.0f, -this->rowHeight * (this->mousePos.y - 1.0f)),
                glm::vec2(this->mousePos.x, -this->rowHeight * (this->mousePos.y - 1.0f))};
            position_buffer_->rebuffer(positions);

            glBindVertexArray(pass_vao_);
            passthrough_shader_->use();
            passthrough_shader_->setUniform("mvp", mvp);
            passthrough_shader_->setUniform("global_color", glm::vec3(1.0f, 0.75f, 0.0f));
            glDisableVertexAttribArray(1); // we use the global color so we disable this temporally

            glDrawArrays(GL_LINE_STRIP, 0, 5);

            glEnableVertexAttribArray(1);
            glUseProgram(0);
            glBindVertexArray(0);
        }
    } // dataPrepared

    return true;
}


/*
 * Callback for mouse events (move, press, and release)
 */
bool SequenceRenderer::MouseEvent(float x, float y, view::MouseFlags flags) {
    bool consumeEvent = false;

    auto cam_pose = cam_.getPose();
    view::Camera::OrthographicParameters cam_intrin;
    try {
        cam_intrin = cam_.get<core::view::Camera::OrthographicParameters>();
    } catch (...) { return consumeEvent; }
    double world_x = ((x * 2.0 / static_cast<double>(screen_.x)) - 1.0);
    double world_y = 1.0 - (y * 2.0 / static_cast<double>(screen_.y));
    world_x = world_x * 0.5 * cam_intrin.frustrum_height * cam_intrin.aspect + cam_pose.position.x;
    world_y = world_y * 0.5 * cam_intrin.frustrum_height + cam_pose.position.y;

    this->mousePos = glm::vec2(floorf(world_x), fabsf(floorf(world_y / this->rowHeight)));
    this->mousePosResIdx = static_cast<int>(this->mousePos.x + (this->resCols * (this->mousePos.y - 1)));
    // do nothing else if mouse is outside bounding box
    if (this->mousePos.x < 0.0f || this->mousePos.x > this->resCols || this->mousePos.y < 0.0f ||
        this->mousePos.y > this->resRows) {
        return consumeEvent;
    }

    // left click
    if (flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) {
        if (flags & view::MOUSEFLAG_MODKEY_ALT_DOWN) {
            if (this->mousePosResIdx > -1 && this->mousePosResIdx < (int)this->resCount) {
                if (!this->leftMouseDown) {
                    this->initialClickSelection = !this->selection[this->mousePosResIdx];
                }
                this->selection[this->mousePosResIdx] = this->initialClickSelection;
            }
            consumeEvent = true;
        } else {
            if (this->mousePosResIdx > -1 && this->mousePosResIdx < (int)this->resCount) {
                this->selection[this->mousePosResIdx] = !this->selection[this->mousePosResIdx];
            }
        }
        this->leftMouseDown = true;
    } else {
        this->leftMouseDown = false;
    }

    // propagate selection to selection module
    if (this->resSelectionCall != nullptr) {
        vislib::Array<ResidueSelectionCall::Residue> selectedRes;
        for (unsigned int i = 0; i < this->selection.size(); i++) {
            if (this->selection[i]) {
                unsigned int x = i % this->resCols;
                unsigned int y = i / this->resCols;
                selectedRes.Add(ResidueSelectionCall::Residue());
                selectedRes.Last().chainID = this->aminoAcidIndexStrings[y][x].first;
                selectedRes.Last().resNum = this->aminoAcidIndexStrings[y][x].second;
                selectedRes.Last().id = -1;
            }
        }
        this->resSelectionCall->SetSelectionPointer(&selectedRes);
        (*this->resSelectionCall)(ResidueSelectionCall::CallForSetSelection);
    }

    return consumeEvent;
}


/*
 *
 */
bool SequenceRenderer::PrepareData(MolecularDataCall* mol, BindingSiteCall* bs) {
    if (!mol)
        return false;

    // temporary variables
    unsigned int firstStruct;
    unsigned int firstMol;
    vislib::StringA tmpStr;

    // initialization
    unsigned int oldResCount = this->resCount;
    this->resCount = 0;
    this->vertices.clear();
    this->vertices.reserve(mol->ResidueCount());
    this->chainVertices.clear();
    this->chainVertices.reserve(mol->ResidueCount());
    this->chainSeparatorVertices.clear();
    this->chainSeparatorVertices.reserve(mol->ChainCount() * 4);
    this->chainColors.clear();
    this->chainColors.reserve(mol->ResidueCount());
    this->bsVertices.clear();
    this->bsVertices.reserve(mol->ResidueCount());
    this->bsIndices.clear();
    this->bsIndices.reserve(mol->ResidueCount());
    this->bsColors.clear();
    this->resIndex.clear();
    this->resIndex.reserve(mol->ResidueCount());
    this->resCols = static_cast<unsigned int>(this->resCountPerRowParam.Param<param::IntParam>()->Value());
    this->aminoAcidStrings.clear();
    this->aminoAcidStrings.reserve(mol->ResidueCount() / this->resCols + 1);
    this->aminoAcidIndexStrings.clear();
    this->aminoAcidIndexStrings.reserve(mol->ResidueCount() / this->resCols + 1);
    this->bindingSiteDescription.clear();
    this->bindingSiteNames.clear();
    if (bs) {
        this->bindingSiteDescription.resize(bs->GetBindingSiteCount());
        this->bindingSiteNames.reserve(bs->GetBindingSiteCount());
        this->bsColors.resize(bs->GetBindingSiteCount());
        // copy binding site names
        for (unsigned int i = 0; i < bs->GetBindingSiteCount(); i++) {
            this->bindingSiteDescription[i].clear();
            this->bindingSiteNames.emplace_back(bs->GetBindingSiteName(i).PeekBuffer());
            this->bsColors[i] = glm::make_vec3(bs->GetBindingSiteColor(i).PeekComponents());
        }
    }
    unsigned int currentRow = 0;
    unsigned int maxNumBindingSitesPerRes = 0;

    // count residues
    for (unsigned int cCnt = 0; cCnt < mol->ChainCount(); cCnt++) {
        firstMol = mol->Chains()[cCnt].FirstMoleculeIndex();
        for (unsigned int mCnt = firstMol; mCnt < firstMol + mol->Chains()[cCnt].MoleculeCount(); mCnt++) {
            firstStruct = mol->Molecules()[mCnt].FirstSecStructIndex();
            for (unsigned int sCnt = 0; sCnt < mol->Molecules()[mCnt].SecStructCount(); sCnt++) {
                for (unsigned int rCnt = 0; rCnt < mol->SecondaryStructures()[firstStruct + sCnt].AminoAcidCount();
                     rCnt++) {
                    // store the original index of the current residue
                    this->resIndex.emplace_back(
                        mol->SecondaryStructures()[firstStruct + sCnt].FirstAminoAcidIndex() + rCnt);
                    // store the secondary structure element type of the current residue
                    this->resSecStructType.emplace_back(mol->SecondaryStructures()[firstStruct + sCnt].Type());
                    // compute the position of the residue icon
                    if (this->resCount == 0) {
                        // structure vertices
                        this->vertices.emplace_back(glm::vec2(0.0f));
                        this->aminoAcidStrings.emplace_back("");
                        this->aminoAcidIndexStrings.emplace_back(std::vector<std::pair<char, int>>());
                        // chain tile vertices and colors
                        this->chainVertices.emplace_back(glm::vec2(0.0f, 2.5f));
                        this->chainColors.emplace_back(this->colorTable[cCnt]);
                    } else {
                        if (this->resCount % static_cast<unsigned int>(
                                                 this->resCountPerRowParam.Param<param::IntParam>()->Value()) !=
                            0) {
                            // structure vertices
                            this->vertices.emplace_back(glm::vec2(
                                this->vertices[this->resCount - 1].x + 1.0f, this->vertices[this->resCount - 1].y));
                            // chain tile vertices and colors
                            this->chainVertices.emplace_back(glm::vec2(this->chainVertices[this->resCount - 1].x + 1.0f,
                                this->chainVertices[this->resCount - 1].y));
                            this->chainColors.emplace_back(this->colorTable[cCnt]);
                        } else {
                            // structure vertices
                            this->vertices.emplace_back(
                                glm::vec2(0.0f, this->vertices[this->resCount - 1].y + this->rowHeight));
                            currentRow++;
                            this->aminoAcidStrings.emplace_back("");
                            // this->aminoAcidIndexStrings.Add(vislib::Array<vislib::StringA>());
                            this->aminoAcidIndexStrings.emplace_back(std::vector<std::pair<char, int>>());
                            // chain tile vertices and colors
                            this->chainVertices.emplace_back(
                                glm::vec2(0.0f, this->chainVertices[this->resCount - 1].y + this->rowHeight));
                            this->chainColors.emplace_back(this->colorTable[cCnt]);
                        }
                    }
                    // store the amino acid name
                    this->aminoAcidStrings[currentRow] += protein_calls::GetAminoAcidOneLetterCode(
                        mol->ResidueTypeNames()[mol->Residues()[this->resIndex.back()]->Type()].PeekBuffer());
                    // store the chain name and residue index
                    this->aminoAcidIndexStrings[currentRow].emplace_back(std::pair<char, int>(
                        mol->Chains()[cCnt].Name(), mol->Residues()[this->resIndex.back()]->OriginalResIndex()));

                    // try to match binding sites
                    if (bs) {
                        vislib::Pair<char, unsigned int> bsRes;
                        unsigned int numBS = 0;
                        // loop over all binding sites
                        for (unsigned int bsCnt = 0; bsCnt < bs->GetBindingSiteCount(); bsCnt++) {
                            for (unsigned int bsResCnt = 0; bsResCnt < bs->GetBindingSite(bsCnt)->Count(); bsResCnt++) {
                                bsRes = bs->GetBindingSite(bsCnt)->operator[](bsResCnt);
                                if (mol->Chains()[cCnt].Name() == bsRes.First() &&
                                    mol->Residues()[this->resIndex.back()]->OriginalResIndex() == bsRes.Second() &&
                                    mol->ResidueTypeNames()[mol->Residues()[this->resIndex.back()]->Type()] ==
                                        bs->GetBindingSiteResNames(bsCnt)->operator[](bsResCnt)) {
                                    this->bsVertices.emplace_back(glm::vec2(this->vertices[this->vertices.size() - 1].x,
                                        this->vertices[this->vertices.size() - 1].y + 3.0f + numBS * 0.5f));
                                    this->bsIndices.emplace_back(bsCnt);
                                    if (!this->bindingSiteDescription[bsCnt].empty()) {
                                        this->bindingSiteDescription[bsCnt].append(", ");
                                    }
                                    this->bindingSiteDescription[bsCnt].append(tmpStr);
                                    numBS++;
                                    maxNumBindingSitesPerRes = vislib::math::Max(maxNumBindingSitesPerRes, numBS);
                                }
                            }
                        }
                    }

                    this->resCount++;
                } // residues
            }     // secondary structures
        }         // molecules
        // add chain separator vertices
        this->chainSeparatorVertices.emplace_back(glm::vec2(
            this->vertices[this->vertices.size() - 1].x + 1.0f, -this->vertices[this->vertices.size() - 1].y));
        this->chainSeparatorVertices.emplace_back(glm::vec2(this->vertices[this->vertices.size() - 1].x + 1.0f,
            -(this->chainVertices[this->chainVertices.size() - 1].y + 0.5f)));
    } // chains

    // check if residue count changed and adjust selection array
    if (oldResCount != this->resCount) {
        this->selection.clear();
        this->selection.resize(this->resCount, false);
    }

    // loop over all binding sites and add binding site descriptions
    if (bs) {
        for (unsigned int bsCnt = 0; bsCnt < bs->GetBindingSiteCount(); bsCnt++) {
            this->bindingSiteDescription[bsCnt] = "\n" + this->bindingSiteDescription[bsCnt];
            this->bindingSiteDescription[bsCnt] =
                bs->GetBindingSiteDescription(bsCnt).PeekBuffer() + this->bindingSiteDescription[bsCnt];
        }
    }

    this->rowHeight = 3.0f + maxNumBindingSitesPerRes * 0.5f + 0.5f;

    // set the number of columns
    this->resCols = vislib::math::Min(
        this->resCount, static_cast<unsigned int>(this->resCountPerRowParam.Param<param::IntParam>()->Value()));
    this->resCountPerRowParam.Param<param::IntParam>()->SetValue(this->resCols);
    // compute the number of rows
    this->resRows =
        static_cast<unsigned int>(ceilf(static_cast<float>(this->resCount) / static_cast<float>(this->resCols)));

    return true;
}

bool SequenceRenderer::LoadTexture(const std::string filename) {
    // find the texture filepath
    std::string texture_filepath;
    const auto resource_directories =
        frontend_resources.get<megamol::frontend_resources::RuntimeConfig>().resource_directories;
    for (const auto& cur_dir : resource_directories) {
        auto found_filepath = core::utility::FileUtils::SearchFileRecursive(cur_dir, filename);
        if (!found_filepath.empty()) {
            texture_filepath = found_filepath;
        }
    }
    // load the actual texture
    marker_textures_.emplace_back(nullptr);
    if (core_gl::utility::RenderUtils::LoadTextureFromFile(marker_textures_.back(), texture_filepath)) {
        return true;
    } else {
        marker_textures_.pop_back();
    }
    return false;
}
