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
        , atomCount(0)
        , bindingSiteCount(0)
        , resCount(0)
        , resCols(0)
        , resRows(0)
        , rowHeight(3.0f)
#ifndef USE_SIMPLE_FONT
        , theFont(FontInfo_Verdana)
#endif // USE_SIMPLE_FONT
        , markerTextures_{}
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
    lr.push_back("RuntimeConfig");
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

    return true;
}

/*
 * SequenceRenderer::release
 */
void SequenceRenderer::release() {}

bool SequenceRenderer::GetExtents(core_gl::view::CallRender2DGL& call) {
    // check molecular data
    MolecularDataCall* mol = this->dataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == NULL)
        return false;
    if (!(*mol)(MolecularDataCall::CallForGetData))
        return false;

    // get pointer to BindingSiteCall
    BindingSiteCall* bs = this->bindingSiteCallerSlot.CallAs<BindingSiteCall>();
    if (bs != NULL) {
        // execute the call
        if (!(*bs)(BindingSiteCall::CallForGetData)) {
            bs = NULL;
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

    return true;
}


/*
 * SequenceRenderer::Render
 */
bool SequenceRenderer::Render(core_gl::view::CallRender2DGL& call) {
    // get pointer to MolecularDataCall
    MolecularDataCall* mol = this->dataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == NULL)
        return false;
    // execute the call
    if (!(*mol)(MolecularDataCall::CallForGetData))
        return false;

    auto proj = call.GetCamera().getProjectionMatrix();
    auto view = call.GetCamera().getViewMatrix();

    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glLoadMatrixf(glm::value_ptr(proj));
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glLoadMatrixf(glm::value_ptr(view));

    this->resSelectionCall = this->resSelectionCallerSlot.CallAs<ResidueSelectionCall>();
    if (this->resSelectionCall != NULL) {
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

    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // get the text color (inverse background color)
    float bgColor[4];
    float fgColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
    for (unsigned int i = 0; i < 4; i++) {
        fgColor[i] -= bgColor[i];
    }

    if (this->dataPrepared) {
        // temporary variables and constants
        const float eps = 0.0f;
        vislib::StringA tmpStr;
        // draw tiles for structure
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_BLEND);
        glActiveTexture(GL_TEXTURE0);
        for (unsigned int i = 0; i < this->resIndex.size(); i++) {
            markerTextures_[0]->bindTexture();
            if (this->resSecStructType[i] == MolecularDataCall::SecStructure::TYPE_HELIX) {
                glColor3f(1.0f, 0.0f, 0.0f);
                if (i > 0 && this->resSecStructType[i - 1] != this->resSecStructType[i]) {
                    markerTextures_[4]->bindTexture();
                } else if ((i + 1) < this->resIndex.size() &&
                           this->resSecStructType[i + 1] != this->resSecStructType[i]) {
                    markerTextures_[6]->bindTexture();
                } else {
                    markerTextures_[5]->bindTexture();
                }
            } else if (this->resSecStructType[i] == MolecularDataCall::SecStructure::TYPE_SHEET) {
                glColor3f(0.0f, 0.0f, 1.0f);
                if ((i + 1) < this->resIndex.size() && this->resSecStructType[i + 1] != this->resSecStructType[i]) {
                    markerTextures_[3]->bindTexture();
                } else {
                    markerTextures_[2]->bindTexture();
                }
            } else if (this->resSecStructType[i] == MolecularDataCall::SecStructure::TYPE_TURN) {
                glColor3f(1.0f, 1.0f, 0.0f);
                markerTextures_[1]->bindTexture();
            } else { // TYPE_COIL
                glColor3f(0.5f, 0.5f, 0.5f);
                markerTextures_[1]->bindTexture();
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

        // draw tiles for binding sites
        glBegin(GL_QUADS);
        for (unsigned int i = 0; i < this->bsIndices.size(); i++) {
            // glColor3fv( this->colorTable[this->bsIndices[i] + mol->ChainCount()].PeekComponents());
            glColor3fv(&this->bsColors[this->bsIndices[i]].x);
            glVertex2f(this->bsVertices[i].x + 0.1f, -this->bsVertices[i].y);
            glVertex2f(this->bsVertices[i].x + 0.1f, -this->bsVertices[i].y - 0.4f);
            glVertex2f(this->bsVertices[i].x + 0.9f, -this->bsVertices[i].y - 0.4f);
            glVertex2f(this->bsVertices[i].x + 0.9f, -this->bsVertices[i].y);
        }
        glEnd();
        // draw tiles for chains
        glBegin(GL_QUADS);
        for (unsigned int i = 0; i < this->chainVertices.size(); i++) {
            glColor3fv((&this->chainColors[i].x));
            glVertex2f(this->chainVertices[i].x, -this->chainVertices[i].y - 0.2f);
            glVertex2f(this->chainVertices[i].x, -this->chainVertices[i].y - 0.3f);
            glVertex2f(this->chainVertices[i].x + 1.0f, -this->chainVertices[i].y - 0.3f);
            glVertex2f(this->chainVertices[i].x + 1.0f, -this->chainVertices[i].y - 0.2f);
        }
        glEnd();

        // draw chain separators
        glColor3fv(fgColor);
        glEnable(GL_VERTEX_ARRAY);
        glVertexPointer(2, GL_FLOAT, 0, this->chainSeparatorVertices.data());
        glDrawArrays(GL_LINES, 0, (GLsizei)this->chainSeparatorVertices.size());
        glDisable(GL_VERTEX_ARRAY);

        // labeling
        glColor3fv(fgColor);
        if (theFont.Initialise()) {
            for (unsigned int i = 0; i < this->aminoAcidStrings.size(); i++) {
                for (unsigned int j = 0; j < (unsigned int)this->aminoAcidStrings[i].size(); j++) {
                    // draw the one-letter amino acid code
                    theFont.DrawString(static_cast<float>(j), -(static_cast<float>(i) * this->rowHeight), 1.0f, -1.0f,
                        1.0f, true, this->aminoAcidStrings[i].substr(j, 1).c_str(),
                        vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);
                    // draw the chain name and amino acid index
                    // theFont.DrawString( static_cast<float>(j), -( static_cast<float>(i) * this->rowHeight +
                    // this->rowHeight - 0.5f), 1.0f, 0.5f,
                    tmpStr.Format(
                        "%c %i", this->aminoAcidIndexStrings[i][j].first, this->aminoAcidIndexStrings[i][j].second);
                    theFont.DrawString(static_cast<float>(j), -(static_cast<float>(i) * this->rowHeight + 2.5f), 1.0f,
                        0.5f, 0.35f, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE);
                    // 0.35f, true, this->aminoAcidIndexStrings[i][j],
                    // vislib::graphics::AbstractFont::ALIGN_CENTER_MIDDLE);
                }
            }
        }

        // draw legend / key
        if (this->toggleKeyParam.Param<param::BoolParam>()->Value()) {
            glColor3fv(fgColor);
            float wordlength;
            float fontSize = 1.0f;
            if (theFont.Initialise()) {
                // draw binding site legend
                if (!this->bindingSiteNames.empty()) {
                    tmpStr = "Binding Sites: ";
                    wordlength = theFont.LineWidth(fontSize, tmpStr);
                    theFont.DrawString(static_cast<float>(this->resCols) + 1.0f, -1.0f, wordlength, 1.0f, fontSize,
                        true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                    for (unsigned int i = 0; i < this->bindingSiteNames.size(); i++) {
                        // draw the binding site names
                        glColor3fv(&this->bsColors[i].x);
                        theFont.DrawString(static_cast<float>(this->resCols) + 1.0f,
                            -(static_cast<float>(i) * 2.0f + 2.0f), wordlength, 1.0f, fontSize, true,
                            this->bindingSiteNames[i].c_str(), vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                        wordlength = theFont.LineWidth(fontSize, this->bindingSiteDescription[i].c_str());
                        theFont.DrawString(static_cast<float>(this->resCols) + 1.0f,
                            -(static_cast<float>(i) * 2.0f + 3.0f), wordlength, 1.0f, fontSize * 0.5f, true,
                            this->bindingSiteDescription[i].c_str(), vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                    }
                }
                // draw diagram legend
                glColor3fv(fgColor);
                fontSize = 0.5f;
                tmpStr = "STRIDE";
                wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                theFont.DrawString(-wordlength, -1.0f, wordlength, 1.0f, fontSize, true, tmpStr,
                    vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                tmpStr = "Amino Acid";
                wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                theFont.DrawString(-wordlength, -2.0f, wordlength, 1.0f, fontSize, true, tmpStr,
                    vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                tmpStr = "PDB";
                wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                theFont.DrawString(-wordlength, -2.5f, wordlength, 0.5f, fontSize, true, tmpStr,
                    vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                tmpStr = "Chain";
                wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                theFont.DrawString(-wordlength, -3.0f, wordlength, 0.5f, fontSize, true, tmpStr,
                    vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                if (!this->bindingSiteNames.empty()) {
                    tmpStr = "Binding Sites";
                    wordlength = theFont.LineWidth(fontSize, tmpStr) + fontSize;
                    theFont.DrawString(-wordlength, -this->rowHeight, wordlength, this->rowHeight - 3.0f, fontSize,
                        true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_MIDDLE);
                }
            }
        }

        glDisable(GL_DEPTH_TEST);
        // draw overlays for selected amino acids
        vislib::math::Vector<float, 2> pos;
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
        fgColor[3] = 0.3f;
        glColor4fv(fgColor);
        for (unsigned int i = 0; i < this->selection.size(); i++) {
            if (this->selection[i]) {
                pos.Set(static_cast<float>(i % this->resCols), floorf(static_cast<float>(i) / this->resCols) + 1.0f);
                glBegin(GL_QUADS);
                glVertex2f(pos.X(), -this->rowHeight * (pos.Y() - 1.0f));
                glVertex2f(pos.X(), -this->rowHeight * (pos.Y()) + 0.5f);
                glVertex2f(pos.X() + 1.0f, -this->rowHeight * (pos.Y()) + 0.5f);
                glVertex2f(pos.X() + 1.0f, -this->rowHeight * (pos.Y() - 1.0f));
                glEnd();
            }
        }
        glDisable(GL_BLEND);

        // draw frames for selected amino acids
        fgColor[3] = 1.0f;
        glColor3fv(fgColor);
        glBegin(GL_LINES);
        for (unsigned int i = 0; i < this->selection.size(); i++) {
            if (this->selection[i]) {
                pos.Set(static_cast<float>(i % this->resCols), floorf(static_cast<float>(i) / this->resCols) + 1.0f);
                // left (only draw if left neighbor ist not selected)
                if (i == 0 || !this->selection[i - 1]) {
                    glVertex2f(pos.X(), -this->rowHeight * (pos.Y() - 1.0f));
                    glVertex2f(pos.X(), -this->rowHeight * (pos.Y()) + 0.5f);
                }
                // bottom
                glVertex2f(pos.X(), -this->rowHeight * (pos.Y()) + 0.5f);
                glVertex2f(pos.X() + 1.0f, -this->rowHeight * (pos.Y()) + 0.5f);
                // right (only draw if right neighbor ist not selected)
                if (i == (this->selection.size() - 1) || !this->selection[i + 1]) {
                    glVertex2f(pos.X() + 1.0f, -this->rowHeight * (pos.Y()) + 0.5f);
                    glVertex2f(pos.X() + 1.0f, -this->rowHeight * (pos.Y() - 1.0f));
                }
                // top
                glVertex2f(pos.X() + 1.0f, -this->rowHeight * (pos.Y() - 1.0f));
                glVertex2f(pos.X(), -this->rowHeight * (pos.Y() - 1.0f));
            }
        }
        glEnd();

        // render mouse hover
        if (this->mousePos.x > -1.0f && this->mousePos.x < static_cast<float>(this->resCols) &&
            this->mousePos.y > 0.0f && this->mousePos.y < static_cast<float>(this->resRows + 1) &&
            this->mousePosResIdx > -1 && this->mousePosResIdx < (int)this->resCount) {
            glColor3f(1.0f, 0.75f, 0.0f);
            glBegin(GL_LINE_STRIP);
            glVertex2f(this->mousePos.x, -this->rowHeight * (this->mousePos.y - 1.0f));
            glVertex2f(this->mousePos.x, -this->rowHeight * (this->mousePos.y) + 0.5f);
            glVertex2f(this->mousePos.x + 1.0f, -this->rowHeight * (this->mousePos.y) + 0.5f);
            glVertex2f(this->mousePos.x + 1.0f, -this->rowHeight * (this->mousePos.y - 1.0f));
            glVertex2f(this->mousePos.x, -this->rowHeight * (this->mousePos.y - 1.0f));
            glEnd();
        }
        glEnable(GL_DEPTH_TEST);

    } // dataPrepared

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    return true;
}


/*
 * Callback for mouse events (move, press, and release)
 */
bool SequenceRenderer::MouseEvent(float x, float y, view::MouseFlags flags) {
    bool consumeEvent = false;
    this->mousePos = glm::vec2(floorf(x), fabsf(floorf(y / this->rowHeight)));
    this->mousePosResIdx = static_cast<int>(this->mousePos.x + (this->resCols * (this->mousePos.y - 1)));
    // do nothing else if mouse is outside bounding box
    if (this->mousePos.x < 0.0f || this->mousePos.x > this->resCols || this->mousePos.y < 0.0f ||
        this->mousePos.y > this->resRows) {
        return consumeEvent;
    }

    // left click
    if (flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) {
        if (flags & view::MOUSEFLAG_MODKEY_ALT_DOWN) {
            if (!this->leftMouseDown) {
                this->initialClickSelection = !this->selection[this->mousePosResIdx];
            }
            this->selection[this->mousePosResIdx] = this->initialClickSelection;
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
    if (this->resSelectionCall != NULL) {
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
            this->bindingSiteNames.push_back(bs->GetBindingSiteName(i).PeekBuffer());
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
                    this->resIndex.push_back(
                        mol->SecondaryStructures()[firstStruct + sCnt].FirstAminoAcidIndex() + rCnt);
                    // store the secondary structure element type of the current residue
                    this->resSecStructType.push_back(mol->SecondaryStructures()[firstStruct + sCnt].Type());
                    // compute the position of the residue icon
                    if (this->resCount == 0) {
                        // structure vertices
                        this->vertices.push_back(glm::vec2(0.0f));
                        this->aminoAcidStrings.push_back("");
                        this->aminoAcidIndexStrings.push_back(std::vector<std::pair<char, int>>());
                        // chain tile vertices and colors
                        this->chainVertices.push_back(glm::vec2(0.0f, 2.5f));
                        this->chainColors.push_back(this->colorTable[cCnt]);
                    } else {
                        if (this->resCount % static_cast<unsigned int>(
                                                 this->resCountPerRowParam.Param<param::IntParam>()->Value()) !=
                            0) {
                            // structure vertices
                            this->vertices.push_back(glm::vec2(
                                this->vertices[this->resCount - 1].x + 1.0f, this->vertices[this->resCount - 1].y));
                            // chain tile vertices and colors
                            this->chainVertices.push_back(glm::vec2(this->chainVertices[this->resCount - 1].x + 1.0f,
                                this->chainVertices[this->resCount - 1].y));
                            this->chainColors.push_back(this->colorTable[cCnt]);
                        } else {
                            // structure vertices
                            this->vertices.push_back(
                                glm::vec2(0.0f, this->vertices[this->resCount - 1].y + this->rowHeight));
                            currentRow++;
                            this->aminoAcidStrings.push_back("");
                            // this->aminoAcidIndexStrings.Add(vislib::Array<vislib::StringA>());
                            this->aminoAcidIndexStrings.push_back(std::vector<std::pair<char, int>>());
                            // chain tile vertices and colors
                            this->chainVertices.push_back(
                                glm::vec2(0.0f, this->chainVertices[this->resCount - 1].y + this->rowHeight));
                            this->chainColors.push_back(this->colorTable[cCnt]);
                        }
                    }
                    // store the amino acid name
                    this->aminoAcidStrings[currentRow] += protein_calls::GetAminoAcidOneLetterCode(
                        mol->ResidueTypeNames()[mol->Residues()[this->resIndex.back()]->Type()].PeekBuffer());
                    // store the chain name and residue index
                    this->aminoAcidIndexStrings[currentRow].push_back(std::pair<char, int>(
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
                                    this->bsVertices.push_back(glm::vec2(this->vertices[this->vertices.size() - 1].x,
                                        this->vertices[this->vertices.size() - 1].y + 3.0f + numBS * 0.5f));
                                    this->bsIndices.push_back(bsCnt);
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
        this->chainSeparatorVertices.push_back(glm::vec2(
            this->vertices[this->vertices.size() - 1].x + 1.0f, -this->vertices[this->vertices.size() - 1].y));
        this->chainSeparatorVertices.push_back(glm::vec2(this->vertices[this->vertices.size() - 1].x + 1.0f,
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
    markerTextures_.push_back(nullptr);
    if (core_gl::utility::RenderUtils::LoadTextureFromFile(markerTextures_.back(), texture_filepath)) {
        return true;
    } else {
        markerTextures_.pop_back();
    }
    return false;
}
