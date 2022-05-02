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
        , markerTextures(0)
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
        for (unsigned int i = 0; i < aminoAcidIndexStrings.Count(); i++) {
            for (unsigned int j = 0; j < aminoAcidIndexStrings[i].Count(); j++) {
                this->selection[cnt] = false;
                if (resSelPtr) {
                    // loop over selections and try to match wit the current amino acid
                    for (unsigned int k = 0; k < resSelPtr->Count(); k++) {
                        if ((*resSelPtr)[k].chainID == aminoAcidIndexStrings[i][j].First() &&
                            (*resSelPtr)[k].resNum == aminoAcidIndexStrings[i][j].Second()) {
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
        for (unsigned int i = 0; i < this->resIndex.Count(); i++) {
            //markerTextures[0]->Bind();
            markerTextures_[0]->bindTexture();
            if (this->resSecStructType[i] == MolecularDataCall::SecStructure::TYPE_HELIX) {
                glColor3f(1.0f, 0.0f, 0.0f);
                if (i > 0 && this->resSecStructType[i - 1] != this->resSecStructType[i]) {
                    //markerTextures[4]->Bind();
                    markerTextures_[4]->bindTexture();
                } else if ((i + 1) < this->resIndex.Count() &&
                           this->resSecStructType[i + 1] != this->resSecStructType[i]) {
                    //markerTextures[6]->Bind();
                    markerTextures_[6]->bindTexture();
                } else {
                    //markerTextures[5]->Bind();
                    markerTextures_[5]->bindTexture();
                }
            } else if (this->resSecStructType[i] == MolecularDataCall::SecStructure::TYPE_SHEET) {
                glColor3f(0.0f, 0.0f, 1.0f);
                if ((i + 1) < this->resIndex.Count() && this->resSecStructType[i + 1] != this->resSecStructType[i]) {
                    //markerTextures[3]->Bind();
                    markerTextures_[3]->bindTexture();
                } else {
                    //markerTextures[2]->Bind();
                    markerTextures_[2]->bindTexture();
                }
            } else if (this->resSecStructType[i] == MolecularDataCall::SecStructure::TYPE_TURN) {
                glColor3f(1.0f, 1.0f, 0.0f);
                //markerTextures[1]->Bind();
                markerTextures_[1]->bindTexture();
            } else { // TYPE_COIL
                glColor3f(0.5f, 0.5f, 0.5f);
                //markerTextures[1]->Bind();
                markerTextures_[1]->bindTexture();
            }
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
            glBegin(GL_QUADS);
            glTexCoord2f(eps, eps);
            glVertex2f(this->vertices[2 * i], -this->vertices[2 * i + 1]);
            glTexCoord2f(eps, 1.0f - eps);
            glVertex2f(this->vertices[2 * i], -this->vertices[2 * i + 1] - 1.0f);
            glTexCoord2f(1.0f - eps, 1.0f - eps);
            glVertex2f(this->vertices[2 * i] + 1.0f, -this->vertices[2 * i + 1] - 1.0f);
            glTexCoord2f(1.0f - eps, eps);
            glVertex2f(this->vertices[2 * i] + 1.0f, -this->vertices[2 * i + 1]);
            glEnd();
        }
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_2D);
        // draw tiles for binding sites
        glBegin(GL_QUADS);
        for (unsigned int i = 0; i < this->bsIndices.Count(); i++) {
            // glColor3fv( this->colorTable[this->bsIndices[i] + mol->ChainCount()].PeekComponents());
            glColor3fv(this->bsColors[this->bsIndices[i]].PeekComponents());
            glVertex2f(this->bsVertices[2 * i] + 0.1f, -this->bsVertices[2 * i + 1]);
            glVertex2f(this->bsVertices[2 * i] + 0.1f, -this->bsVertices[2 * i + 1] - 0.4f);
            glVertex2f(this->bsVertices[2 * i] + 0.9f, -this->bsVertices[2 * i + 1] - 0.4f);
            glVertex2f(this->bsVertices[2 * i] + 0.9f, -this->bsVertices[2 * i + 1]);
        }
        glEnd();
        // draw tiles for chains
        glBegin(GL_QUADS);
        for (unsigned int i = 0; i < this->chainVertices.Count() / 2; i++) {
            glColor3fv((&this->chainColors[i * 3]));
            glVertex2f(this->chainVertices[2 * i], -this->chainVertices[2 * i + 1] - 0.2f);
            glVertex2f(this->chainVertices[2 * i], -this->chainVertices[2 * i + 1] - 0.3f);
            glVertex2f(this->chainVertices[2 * i] + 1.0f, -this->chainVertices[2 * i + 1] - 0.3f);
            glVertex2f(this->chainVertices[2 * i] + 1.0f, -this->chainVertices[2 * i + 1] - 0.2f);
        }
        glEnd();

        // draw chain separators
        glColor3fv(fgColor);
        glEnable(GL_VERTEX_ARRAY);
        glVertexPointer(2, GL_FLOAT, 0, this->chainSeparatorVertices.PeekElements());
        glDrawArrays(GL_LINES, 0, (GLsizei)this->chainSeparatorVertices.Count() / 2 - 2);
        glDisable(GL_VERTEX_ARRAY);

        // labeling
        glColor3fv(fgColor);
        if (theFont.Initialise()) {
            for (unsigned int i = 0; i < this->aminoAcidStrings.Count(); i++) {
                for (unsigned int j = 0; j < (unsigned int)this->aminoAcidStrings[i].Length(); j++) {
                    // draw the one-letter amino acid code
                    theFont.DrawString(static_cast<float>(j), -(static_cast<float>(i) * this->rowHeight), 1.0f, -1.0f,
                        1.0f, true, this->aminoAcidStrings[i].Substring(j, 1),
                        vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);
                    // draw the chain name and amino acid index
                    // theFont.DrawString( static_cast<float>(j), -( static_cast<float>(i) * this->rowHeight +
                    // this->rowHeight - 0.5f), 1.0f, 0.5f,
                    tmpStr.Format(
                        "%c %i", this->aminoAcidIndexStrings[i][j].First(), this->aminoAcidIndexStrings[i][j].Second());
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
                if (!this->bindingSiteNames.IsEmpty()) {
                    tmpStr = "Binding Sites: ";
                    wordlength = theFont.LineWidth(fontSize, tmpStr);
                    theFont.DrawString(static_cast<float>(this->resCols) + 1.0f, -1.0f, wordlength, 1.0f, fontSize,
                        true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                    for (unsigned int i = 0; i < this->bindingSiteNames.Count(); i++) {
                        // draw the binding site names
                        glColor3fv(this->bsColors[i].PeekComponents());
                        theFont.DrawString(static_cast<float>(this->resCols) + 1.0f,
                            -(static_cast<float>(i) * 2.0f + 2.0f), wordlength, 1.0f, fontSize, true,
                            this->bindingSiteNames[i], vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
                        wordlength = theFont.LineWidth(fontSize, this->bindingSiteDescription[i]);
                        theFont.DrawString(static_cast<float>(this->resCols) + 1.0f,
                            -(static_cast<float>(i) * 2.0f + 3.0f), wordlength, 1.0f, fontSize * 0.5f, true,
                            this->bindingSiteDescription[i], vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
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
                if (!this->bindingSiteNames.IsEmpty()) {
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
        for (unsigned int i = 0; i < this->selection.Count(); i++) {
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
        for (unsigned int i = 0; i < this->selection.Count(); i++) {
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
                if (i == (this->selection.Count() - 1) || !this->selection[i + 1]) {
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
        if (this->mousePos.X() > -1.0f && this->mousePos.X() < static_cast<float>(this->resCols) &&
            this->mousePos.Y() > 0.0f && this->mousePos.Y() < static_cast<float>(this->resRows + 1) &&
            this->mousePosResIdx > -1 && this->mousePosResIdx < (int)this->resCount) {
            glColor3f(1.0f, 0.75f, 0.0f);
            glBegin(GL_LINE_STRIP);
            glVertex2f(this->mousePos.X(), -this->rowHeight * (this->mousePos.Y() - 1.0f));
            glVertex2f(this->mousePos.X(), -this->rowHeight * (this->mousePos.Y()) + 0.5f);
            glVertex2f(this->mousePos.X() + 1.0f, -this->rowHeight * (this->mousePos.Y()) + 0.5f);
            glVertex2f(this->mousePos.X() + 1.0f, -this->rowHeight * (this->mousePos.Y() - 1.0f));
            glVertex2f(this->mousePos.X(), -this->rowHeight * (this->mousePos.Y() - 1.0f));
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
    this->mousePos.Set(floorf(x), fabsf(floorf(y / this->rowHeight)));
    this->mousePosResIdx = static_cast<int>(this->mousePos.X() + (this->resCols * (this->mousePos.Y() - 1)));
    // do nothing else if mouse is outside bounding box
    if (this->mousePos.X() < 0.0f || this->mousePos.X() > this->resCols || this->mousePos.Y() < 0.0f ||
        this->mousePos.Y() > this->resRows) {
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
        for (unsigned int i = 0; i < this->selection.Count(); i++) {
            if (this->selection[i]) {
                unsigned int x = i % this->resCols;
                unsigned int y = i / this->resCols;
                selectedRes.Add(ResidueSelectionCall::Residue());
                selectedRes.Last().chainID = this->aminoAcidIndexStrings[y][x].First();
                selectedRes.Last().resNum = this->aminoAcidIndexStrings[y][x].Second();
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
    this->vertices.Clear();
    this->vertices.AssertCapacity(mol->ResidueCount() * 2);
    this->chainVertices.Clear();
    this->chainVertices.AssertCapacity(mol->ResidueCount() * 2);
    this->chainSeparatorVertices.Clear();
    this->chainSeparatorVertices.AssertCapacity(mol->ChainCount() * 4);
    this->chainColors.Clear();
    this->chainColors.AssertCapacity(mol->ResidueCount() * 3);
    this->bsVertices.Clear();
    this->bsVertices.AssertCapacity(mol->ResidueCount() * 2);
    this->bsIndices.Clear();
    this->bsIndices.AssertCapacity(mol->ResidueCount());
    this->bsColors.Clear();
    this->resIndex.Clear();
    this->resIndex.AssertCapacity(mol->ResidueCount());
    this->resCols = static_cast<unsigned int>(this->resCountPerRowParam.Param<param::IntParam>()->Value());
    this->aminoAcidStrings.Clear();
    this->aminoAcidStrings.AssertCapacity(mol->ResidueCount() / this->resCols + 1);
    this->aminoAcidIndexStrings.Clear();
    this->aminoAcidIndexStrings.AssertCapacity(mol->ResidueCount() / this->resCols + 1);
    this->bindingSiteDescription.Clear();
    this->bindingSiteNames.Clear();
    if (bs) {
        this->bindingSiteDescription.SetCount(bs->GetBindingSiteCount());
        this->bindingSiteNames.AssertCapacity(bs->GetBindingSiteCount());
        this->bsColors.SetCount(bs->GetBindingSiteCount());
        // copy binding site names
        for (unsigned int i = 0; i < bs->GetBindingSiteCount(); i++) {
            this->bindingSiteDescription[i].Clear();
            this->bindingSiteNames.Add(bs->GetBindingSiteName(i));
            this->bsColors[i] = bs->GetBindingSiteColor(i);
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
                    this->resIndex.Add(mol->SecondaryStructures()[firstStruct + sCnt].FirstAminoAcidIndex() + rCnt);
                    // store the secondary structure element type of the current residue
                    this->resSecStructType.Add(mol->SecondaryStructures()[firstStruct + sCnt].Type());
                    // compute the position of the residue icon
                    if (this->resCount == 0) {
                        // structure vertices
                        this->vertices.Add(0.0f);
                        this->vertices.Add(0.0f);
                        this->aminoAcidStrings.Add("");
                        // this->aminoAcidIndexStrings.Add(vislib::Array<vislib::StringA>());
                        this->aminoAcidIndexStrings.Add(vislib::Array<vislib::Pair<char, int>>());
                        // chain tile vertices and colors
                        this->chainVertices.Add(0.0f);
                        // this->chainVertices.Add( this->rowHeight - 0.5f);
                        this->chainVertices.Add(2.5f);
                        this->chainColors.Add(this->colorTable[cCnt].x);
                        this->chainColors.Add(this->colorTable[cCnt].y);
                        this->chainColors.Add(this->colorTable[cCnt].z);
                    } else {
                        if (this->resCount % static_cast<unsigned int>(
                                                 this->resCountPerRowParam.Param<param::IntParam>()->Value()) !=
                            0) {
                            // structure vertices
                            this->vertices.Add(this->vertices[this->resCount * 2 - 2] + 1.0f);
                            this->vertices.Add(this->vertices[this->resCount * 2 - 1]);
                            // chain tile vertices and colors
                            this->chainVertices.Add(this->chainVertices[this->resCount * 2 - 2] + 1.0f);
                            this->chainVertices.Add(this->chainVertices[this->resCount * 2 - 1]);
                            this->chainColors.Add(this->colorTable[cCnt].x);
                            this->chainColors.Add(this->colorTable[cCnt].y);
                            this->chainColors.Add(this->colorTable[cCnt].z);
                        } else {
                            // structure vertices
                            this->vertices.Add(0.0f);
                            this->vertices.Add(this->vertices[this->resCount * 2 - 1] + this->rowHeight);
                            currentRow++;
                            this->aminoAcidStrings.Add("");
                            // this->aminoAcidIndexStrings.Add(vislib::Array<vislib::StringA>());
                            this->aminoAcidIndexStrings.Add(vislib::Array<vislib::Pair<char, int>>());
                            // chain tile vertices and colors
                            this->chainVertices.Add(0.0f);
                            this->chainVertices.Add(this->chainVertices[this->resCount * 2 - 1] + this->rowHeight);
                            this->chainColors.Add(this->colorTable[cCnt].x);
                            this->chainColors.Add(this->colorTable[cCnt].y);
                            this->chainColors.Add(this->colorTable[cCnt].z);
                        }
                    }
                    // store the amino acid name
                    this->aminoAcidStrings[currentRow].Append(this->GetAminoAcidOneLetterCode(
                        mol->ResidueTypeNames()[mol->Residues()[this->resIndex.Last()]->Type()]));
                    // store the chain name and residue index
                    // tmpStr.Format("%c %i", mol->Chains()[cCnt].Name(),
                    // mol->Residues()[this->resIndex.Last()]->OriginalResIndex());
                    // this->aminoAcidIndexStrings[currentRow].Add( tmpStr);
                    this->aminoAcidIndexStrings[currentRow].Add(vislib::Pair<char, int>(
                        mol->Chains()[cCnt].Name(), mol->Residues()[this->resIndex.Last()]->OriginalResIndex()));

                    // try to match binding sites
                    if (bs) {
                        vislib::Pair<char, unsigned int> bsRes;
                        unsigned int numBS = 0;
                        // loop over all binding sites
                        for (unsigned int bsCnt = 0; bsCnt < bs->GetBindingSiteCount(); bsCnt++) {
                            for (unsigned int bsResCnt = 0; bsResCnt < bs->GetBindingSite(bsCnt)->Count(); bsResCnt++) {
                                bsRes = bs->GetBindingSite(bsCnt)->operator[](bsResCnt);
                                if (mol->Chains()[cCnt].Name() == bsRes.First() &&
                                    mol->Residues()[this->resIndex.Last()]->OriginalResIndex() == bsRes.Second() &&
                                    mol->ResidueTypeNames()[mol->Residues()[this->resIndex.Last()]->Type()] ==
                                        bs->GetBindingSiteResNames(bsCnt)->operator[](bsResCnt)) {
                                    // this->aminoAcidIndexStrings[currentRow].Last().Append(" *");
                                    this->bsVertices.Add(this->vertices[this->vertices.Count() - 2]);
                                    this->bsVertices.Add(
                                        this->vertices[this->vertices.Count() - 1] + 3.0f + numBS * 0.5f);
                                    this->bsIndices.Add(bsCnt);
                                    if (!this->bindingSiteDescription[bsCnt].IsEmpty()) {
                                        this->bindingSiteDescription[bsCnt].Append(", ");
                                    }
                                    this->bindingSiteDescription[bsCnt].Append(tmpStr);
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
        this->chainSeparatorVertices.Add(this->vertices[this->vertices.Count() - 2] + 1.0f);
        this->chainSeparatorVertices.Add(-this->vertices[this->vertices.Count() - 1]);
        this->chainSeparatorVertices.Add(this->vertices[this->vertices.Count() - 2] + 1.0f);
        this->chainSeparatorVertices.Add(-(this->chainVertices[this->chainVertices.Count() - 1] + 0.5f));
    } // chains

    // check if residue count changed and adjust selection array
    if (oldResCount != this->resCount) {
        this->selection.SetCount(this->resCount);
        for (unsigned int i = 0; i < this->selection.Count(); i++) {
            this->selection[i] = false;
        }
    }

    // loop over all binding sites and add binding site descriptions
    if (bs) {
        for (unsigned int bsCnt = 0; bsCnt < bs->GetBindingSiteCount(); bsCnt++) {
            this->bindingSiteDescription[bsCnt].Prepend("\n");
            this->bindingSiteDescription[bsCnt].Prepend(bs->GetBindingSiteDescription(bsCnt));
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

/*
 * Check if the residue is an amino acid.
 */
char SequenceRenderer::GetAminoAcidOneLetterCode(vislib::StringA resName) {
    if (resName.Equals("ALA"))
        return 'A';
    else if (resName.Equals("ARG"))
        return 'R';
    else if (resName.Equals("ASN"))
        return 'N';
    else if (resName.Equals("ASP"))
        return 'D';
    else if (resName.Equals("CYS"))
        return 'C';
    else if (resName.Equals("GLN"))
        return 'Q';
    else if (resName.Equals("GLU"))
        return 'E';
    else if (resName.Equals("GLY"))
        return 'G';
    else if (resName.Equals("HIS"))
        return 'H';
    else if (resName.Equals("ILE"))
        return 'I';
    else if (resName.Equals("LEU"))
        return 'L';
    else if (resName.Equals("LYS"))
        return 'K';
    else if (resName.Equals("MET"))
        return 'M';
    else if (resName.Equals("PHE"))
        return 'F';
    else if (resName.Equals("PRO"))
        return 'P';
    else if (resName.Equals("SER"))
        return 'S';
    else if (resName.Equals("THR"))
        return 'T';
    else if (resName.Equals("TRP"))
        return 'W';
    else if (resName.Equals("TYR"))
        return 'Y';
    else if (resName.Equals("VAL"))
        return 'V';
    else if (resName.Equals("ASH"))
        return 'D';
    else if (resName.Equals("CYX"))
        return 'C';
    else if (resName.Equals("CYM"))
        return 'C';
    else if (resName.Equals("GLH"))
        return 'E';
    else if (resName.Equals("HID"))
        return 'H';
    else if (resName.Equals("HIE"))
        return 'H';
    else if (resName.Equals("HIP"))
        return 'H';
    else if (resName.Equals("MSE"))
        return 'M';
    else if (resName.Equals("LYN"))
        return 'K';
    else if (resName.Equals("TYM"))
        return 'Y';
    else
        return '?';
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
