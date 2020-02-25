/*
 * ClusterRenderer.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/view/Renderer2DModule.h"

#include "vislib/sys/Log.h"
#include "vislib\math\ForceDirected.h"

#include <filesystem>

#include "CallClusterPosition.h"
#include "ClusterHierarchieRenderer.h"
#include "ClusterRenderer.h"
#include "TextureLoader.h"

#define LINECOLOR 0.0, 0.0, 1.0
#define BLACK 0, 0, 0
#define PICSCALING 0.2


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::MolSurfMapCluster;


/*
 * ClusterRenderer::ClusterRenderer
 */
ClusterRenderer::ClusterRenderer(void)
    : view::Renderer2DModule()
    , clusterDataSlot("inData", "The input data slot for sphere data.")
    , getPosition("getPosition", "Returns the aktual Rendered-Root-Node from clustering")
    , setPosition("setPosition", "Set the aktual position-root-node from clustering")


    , theFont(megamol::core::utility::SDFFont::FontName::ROBOTO_SANS)
    , mouseX(0.0f)
    , mouseY(0.0f)
    , lastMouseX(0.0f)
    , lastMouseY(0.0f)
    , fontSize(22.0f)
    , texVa(0)
    , mouseButton(MouseButton::BUTTON_LEFT)
    , mouseAction(MouseButtonAction::RELEASE) {


    // Callee Slot
    this->getPosition.SetCallback(
        CallClusterPosition::ClassName(), CallClusterPosition::FunctionName(1), &ClusterRenderer::GetPositionExtents);
    this->getPosition.SetCallback(
        CallClusterPosition::ClassName(), CallClusterPosition::FunctionName(0), &ClusterRenderer::GetPositionData);
    this->MakeSlotAvailable(&this->getPosition);

    // CallerSlot
    this->clusterDataSlot.SetCompatibleCall<CallClusteringDescription>();
    this->MakeSlotAvailable(&this->clusterDataSlot);
    this->setPosition.SetCompatibleCall<CallClusterPositionDescription>();
    this->MakeSlotAvailable(&this->setPosition);

    // ParamSlot


    // Variablen
    this->lastHash = 0;
    this->DataHashPosition = 0;
    this->GetPositionDataHash = 0;
    this->hashoffset = 0;
    this->newposition = false;
    this->clustering = nullptr;
    this->reloadTexures = true;
    this->init = true;

    maxX = DBL_MAX * -1;
    maxY = DBL_MAX * -1;
    minX = DBL_MAX;
    minY = DBL_MAX;

    actionavailable = true;

    this->colors = nullptr;
}


/*
 * ClusterRenderer::~ClusterRenderer
 */
ClusterRenderer::~ClusterRenderer(void) { this->Release(); }


/*
 * ClusterRenderer::create
 */
bool ClusterRenderer::create(void) {

    // Initialise font
    if (!this->theFont.Initialise(this->GetCoreInstance())) {
        vislib::sys::Log::DefaultLog.WriteError("Couldn't initialize the font.");
        return false;
    }

    vislib::graphics::gl::ShaderSource texVertShader;
    vislib::graphics::gl::ShaderSource texFragShader;

    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("molsurfTexture::vertex", texVertShader)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to load vertex shader source for texture Vertex Shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("molsurfTexture::fragment", texFragShader)) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to load fragment shader source for texture Fragment Shader");
        return false;
    }

    try {
        if (!this->textureShader.Create(
                texVertShader.Code(), texVertShader.Count(), texFragShader.Code(), texFragShader.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to create shader: %s\n", e.GetMsgA());
        return false;
    }

    std::vector<float> texVerts = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 1.0f, 1.0f
    };

    this->texBuffer = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, texVerts, GL_STATIC_DRAW);

    glGenVertexArrays(1, &this->texVa);
    glBindVertexArray(this->texVa);

    this->texBuffer->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    return true;
}


/*
 * ClusterRenderer::release
 */
void ClusterRenderer::release(void) {
    if (this->texVa != 0) {
        glDeleteVertexArrays(1, &this->texVa);
        this->texVa = 0;
    }
}


/*
 * ClusterRenderer::GetExtents
 */
bool ClusterRenderer::GetExtents(view::CallRender2D& call) {

    // Incoming Call
    core::view::CallRender2D* cr = dynamic_cast<core::view::CallRender2D*>(&call);
    if (cr == nullptr) return false;

    cr->SetBoundingBox(cr->GetViewport());

    vislib::math::Vector<float, 2> currentViewport;
    currentViewport.SetX(static_cast<float>(cr->GetViewport().GetSize().GetWidth()));
    currentViewport.SetY(static_cast<float>(cr->GetViewport().GetSize().GetHeight()));

    // Check for new Data in clustering
    CallClustering* cc = this->clusterDataSlot.CallAs<CallClustering>();
    if (cc == nullptr) return false;

    if (!(*cc)(CallClustering::CallForGetExtent)) return false;

    // Check for new Position from HierarchieClusterRenderer
    CallClusterPosition* ccp = this->setPosition.CallAs<CallClusterPosition>();
    if (ccp == nullptr) return false;

    if (!(*ccp)(CallClusterPosition::CallForGetExtent)) return false;

    // if viewport changes ....
    if (currentViewport != this->viewport) {
        this->viewport = currentViewport;
    }

    return true;
}

double scale(double unscaled, double min, double max, double minAllowed, double maxAllowed) {
    return minAllowed + (((unscaled - min) * (maxAllowed - minAllowed)) / (max - min));
}

void ClusterRenderer::renderClusterText(HierarchicalClustering::CLUSTERNODE* node, double posx, double posy) {
    // Render cluster name
    float fgColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    vislib::StringA id;

    // Text to Render
    if (node->left == nullptr && node->right == nullptr) {
        // id.Format(node->pic->name.Substring(node->pic->name.FindLast("\\") + 1, 4));
        id.Format(std::filesystem::path(node->pic->path).stem().string().c_str());
    } else {
        id.Format(std::to_string(node->id).c_str());
    }

    float strHeight = this->theFont.LineHeight(this->fontSize);
    float strWidth = this->theFont.LineWidth(this->fontSize, id);

    // Render Background
    glBegin(GL_TRIANGLES);
    glColor3f(1, 1, 1);
    glVertex2f(posx - 0.5 * strWidth, posy - 0.5 * strHeight);
    glVertex2f(posx + 0.5 * strWidth, posy - 0.5 * strHeight);
    glVertex2f(posx + 0.5 * strWidth, posy + 0.5 * strHeight);

    glVertex2f(posx - 0.5 * strWidth, posy - 0.5 * strHeight);
    glVertex2f(posx + 0.5 * strWidth, posy + 0.5 * strHeight);
    glVertex2f(posx - 0.5 * strWidth, posy + 0.5 * strHeight);
    glEnd();

    // Render Text
    this->theFont.DrawString(
        fgColor, posx, posy, this->fontSize, false, id, megamol::core::utility::AbstractFont::ALIGN_CENTER_MIDDLE);
}

void ClusterRenderer::renderLeaveNode(HierarchicalClustering::CLUSTERNODE* node) {

    // Render cluster name
    float fgColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float size = this->fontSize + 15;
    vislib::StringA id;
    id.Format("PDB-ID: ");
    id.Append(std::filesystem::path(node->pic->path).stem().string().c_str());

    float strHeight = this->theFont.LineHeight(size);
    float strWidth = this->theFont.LineWidth(size, id);

    this->theFont.DrawString(fgColor, 0.5 * this->viewport.GetX() - 0.5 * strWidth, this->viewport.GetY() - strHeight,
        size, false, id, megamol::core::utility::AbstractFont::ALIGN_CENTER_BOTTOM);

    // Render Bild
    glEnable(GL_TEXTURE_2D);
    node->pic->texture->Bind();

    auto pos = node->pca2d;

    double posx = 0.5 * this->viewport.GetX();
    double posy = 0.5 * this->viewport.GetY();

    double picwidth = 0.9 * this->viewport.GetX();
    double picheight = 0.9 * this->viewport.GetY();

    glColor3f(1, 1, 1); // Reset Color for Textures

    glBegin(GL_QUADS);

    glTexCoord2f(0, 0);
    glVertex2f(posx - 0.5 * picwidth, posy - 0.5 * picheight);

    glTexCoord2f(1, 0);
    glVertex2f(posx + 0.5 * picwidth, posy - 0.5 * picheight);

    glTexCoord2f(1, 1);
    glVertex2f(posx + 0.5 * picwidth, this->viewport.GetY() - strHeight);

    glTexCoord2f(0, 1);
    glVertex2f(posx - 0.5 * picwidth, this->viewport.GetY() - strHeight);

    glEnd();

    glDisable(GL_TEXTURE_2D);
}

void ClusterRenderer::renderNode(
    HierarchicalClustering::CLUSTERNODE* node, double minX, double maxX, double minY, double maxY) {

    glActiveTexture(GL_TEXTURE0);
    node->pic->texture->Bind();
    glEnable(GL_TEXTURE_2D);

    auto pos = node->pca2d;

    double posx = scale((*pos)[0], minX, maxX, (PICSCALING * 0.5) * this->viewport.GetX(),
        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetX());
    double posy = scale((*pos)[1], minY, maxY, (PICSCALING * 0.5) * this->viewport.GetY(),
        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetY());

    double picwidth = PICSCALING * this->viewport.GetX();
    double picheight = PICSCALING * this->viewport.GetY();

    // Render Picture
    glColor3f(1, 0, 1); // Reset Color for Textures

    glBegin(GL_QUADS);

    glTexCoord2f(0, 0);
    glVertex2f(posx - 0.5 * picwidth, posy - 0.5 * picheight);

    glTexCoord2f(1, 0);
    glVertex2f(posx + 0.5 * picwidth, posy - 0.5 * picheight);

    glTexCoord2f(1, 1);
    glVertex2f(posx + 0.5 * picwidth, posy + 0.5 * picheight);

    glTexCoord2f(0, 1);
    glVertex2f(posx - 0.5 * picwidth, posy + 0.5 * picheight);

    glEnd();
    glDisable(GL_TEXTURE_2D);

    // Cluster beschriftung
    this->renderClusterText(node, posx, posy);
}

void ClusterRenderer::renderAllLeaves(
    HierarchicalClustering::CLUSTERNODE* node, double minX, double maxX, double minY, double maxY) {

    if (node->left != nullptr && node->right != nullptr) {
        auto leaves = this->clustering->getLeavesOfNode(node);

        // Select Color
        bool clusternode = false;
        for (std::tuple<HierarchicalClustering::CLUSTERNODE*, ClusterRenderer::RGBCOLOR*>* colortuple : *colors) {
            if (this->clustering->parentIs(node, std::get<0>(*colortuple))) {
                ClusterRenderer::RGBCOLOR* color = std::get<1>(*colortuple);
                double r = (255 - color->r) / 255;
                double g = (255 - color->g) / 255;
                double b = (255 - color->b) / 255;

                glColor3f(r, g, b);

                clusternode = true;
            }
        }

        for (HierarchicalClustering::CLUSTERNODE* leaf : *leaves) {
            std::vector<double>* pos = leaf->pca2d;
            double posx = scale((*pos)[0], minX, maxX, (PICSCALING * 0.5) * this->viewport.GetX(),
                (1.0 - (PICSCALING * 0.5)) * this->viewport.GetX());
            double posy = scale((*pos)[1], minY, maxY, (PICSCALING * 0.5) * this->viewport.GetY(),
                (1.0 - (PICSCALING * 0.5)) * this->viewport.GetY());

            glPointSize(10);
            glBegin(GL_POINTS);
            glVertex2f(posx, posy);
            glEnd();
        }
    }
}

void DrawCircle(float cx, float cy, float r, int num_segments) {
    glBegin(GL_LINE_LOOP);
    for (int ii = 0; ii < num_segments; ii++) {
        float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments); // get the current angle

        float x = r * cosf(theta); // calculate the x component
        float y = r * sinf(theta); // calculate the y component

        glVertex2f(x + cx, y + cy); // output vertex
    }
    glEnd();
}

void ClusterRenderer::renderRootNode(
    HierarchicalClustering::CLUSTERNODE* node, double minX, double maxX, double minY, double maxY) {
    auto pos = node->pca2d;

    double posx = scale((*pos)[0], minX, maxX, (PICSCALING * 0.5) * this->viewport.GetX(),
        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetX());
    double posy = scale((*pos)[1], minY, maxY, (PICSCALING * 0.5) * this->viewport.GetY(),
        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetY());

    double picwidth = PICSCALING * this->viewport.GetX();
    double picheight = PICSCALING * this->viewport.GetY();

    glColor3f(BLACK); // Reset Color for Textures

    // Render Cluster Mittelpunklt
    DrawCircle(posx, posy, this->viewport.GetY() * 0.02, 100000);
    this->renderClusterText(node, posx, posy);
}


void ClusterRenderer::connectNodes(
    HierarchicalClustering::CLUSTERNODE* node1, HierarchicalClustering::CLUSTERNODE* node2) {

    // Calculate Start and End Position
    auto pos1 = node1->pca2d;

    double posx1 = scale((*pos1)[0], minX, maxX, (PICSCALING * 0.5) * this->viewport.GetX(),
        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetX());
    double posy1 = scale((*pos1)[1], minY, maxY, (PICSCALING * 0.5) * this->viewport.GetY(),
        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetY());

    auto pos2 = node2->pca2d;

    double posx2 = scale((*pos2)[0], minX, maxX, (PICSCALING * 0.5) * this->viewport.GetX(),
        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetX());
    double posy2 = scale((*pos2)[1], minY, maxY, (PICSCALING * 0.5) * this->viewport.GetY(),
        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetY());

    double picwidth = PICSCALING * this->viewport.GetX();
    double picheight = PICSCALING * this->viewport.GetY();

    double distance = this->clustering->distance(node1->features, node2->features);
    double tmp = distance / this->maxdistance;

    // Draw Line
    glLineWidth(5);
    glColor3f(1 - tmp, 1 - tmp, 1 - tmp);
    glBegin(GL_LINES);
    glVertex2f(posx1, posy1);
    glVertex2f(posx2, posy2);
    glEnd();
}

void ClusterRenderer::setMinMax(std::vector<HierarchicalClustering::CLUSTERNODE*>* leaves) {
    // reset minmax
    this->maxX = DBL_MAX * -1;
    this->minX = DBL_MAX;
    this->maxY = DBL_MAX * -1;
    this->minY = DBL_MAX;

    // get max and min for scaling
    for (int i = 0; i < leaves->size(); i++) {
        double x = (*(*leaves)[i]->pca2d)[0];
        double y = (*(*leaves)[i]->pca2d)[1];

        if (x > this->maxX) this->maxX = x;
        if (x < this->minX) this->minX = x;
        if (y > this->maxY) this->maxY = y;
        if (y < this->minY) this->minY = y;
    }
}

/*
 * ClusterRenderer::Render
 */
bool ClusterRenderer::Render(view::CallRender2D& call) {

    core::view::CallRender2D* cr = dynamic_cast<core::view::CallRender2D*>(&call);
    if (cr == nullptr) return false;

    // Update data
    CallClustering* ccc = this->clusterDataSlot.CallAs<CallClustering>();
    if (!ccc) return false;
    if (!(*ccc)(CallClustering::CallForGetData)) return false;

    if (ccc->DataHash() != this->lastHash) {
        // update Clustering to work with
        this->clustering = ccc->getClustering();
        this->lastHash = ccc->DataHash();
        this->init = true;
    }

    if (this->clustering != nullptr) {
        if (this->clustering->finished()) {

            // Load root node first time
            if (this->init) {
                this->root = this->clustering->getRoot();
                this->cluster = this->clustering->getClusterNodesOfNode(this->root);
                this->maxdistance = this->clustering->getMaxDistanceOfLeavesToRoot();
                this->init = false;
                this->reloadTexures = true;
            } else {
                // Check for new position from hierarchie
                CallClusterPosition* ccp = this->setPosition.CallAs<CallClusterPosition>();
                if (!ccp) return false;
                if (!(*ccp)(CallClusterPosition::CallForGetData)) return false;

                if (ccp->DataHash() != this->GetPositionDataHash) {
                    // update Clustering to work with
                    this->root = ccp->getPosition();
                    this->cluster = this->clustering->getClusterNodesOfNode(root);
                    reloadTexures = true;
                    this->GetPositionDataHash = ccp->DataHash();
                }
            }

            if (this->reloadTexures) {
                // Load Textures
                TextureLoader::loadTextures(this->root, this->clustering);
                this->colors = this->getNdiffrentColors(this->cluster);
                this->reloadTexures = false;
            }

            auto leaves = clustering->getLeavesOfNode(root);
            setMinMax(leaves);

            if (root->left != nullptr && root->right != nullptr) {
                // Render Distance
                for (HierarchicalClustering::CLUSTERNODE* node : *this->cluster) {
                    this->connectNodes(root, node);
                }

                // Render Cluster
                for (HierarchicalClustering::CLUSTERNODE* node : *this->cluster) {
                    this->renderNode(node, this->minX, this->maxX, this->minY, this->maxY);

                    std::vector<BYTE> loctex = {0,   255, 0,
                                                255, 0,   0,
                                                0,   0,   255,
                                                255, 255, 255};
                    glowl::TextureLayout layout(GL_RGB8, 2, 2, 1, GL_RGB, GL_UNSIGNED_BYTE, 1);

                    std::unique_ptr<glowl::Texture2D> mytex =
                        std::make_unique<glowl::Texture2D>("test", layout, loctex.data());

                    glBindVertexArray(this->texVa);
                    this->textureShader.Enable();

                    glEnable(GL_TEXTURE_2D);
                    glActiveTexture(GL_TEXTURE0);
                    mytex->bindTexture();

                    glUniform1i(this->textureShader.ParameterLocation("tex"), 0);
                    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

                    this->textureShader.Disable();
                    glBindVertexArray(0);
                }

                // Render Clusterchildren
                for (HierarchicalClustering::CLUSTERNODE* node : *this->cluster) {
                    this->renderAllLeaves(node, this->minX, this->maxX, this->minY, this->maxY);
                }

                // Render Distance indikator
                this->renderDistanceIndikator();
            } else {
                this->renderLeaveNode(root);
            }
        }
    }
    return true;
}

/*
 * ClustrerRenderer::OnMouseButton
 */
bool ClusterRenderer::OnMouseButton(megamol::core::view::MouseButton button,
    megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) {

    auto down = action == MouseButtonAction::PRESS;
    this->mouseAction = action;
    this->mouseButton = button;

    if (actionavailable) {
        // Right-click ==> zoom out
        if (button == MouseButton::BUTTON_RIGHT) {
            if (root->parent != nullptr) {
                this->root = this->clustering->getClusterRootOfNode(root);
                this->cluster = this->clustering->getClusterNodesOfNode(root);
                reloadTexures = true;
                actionavailable = false;
                return true;
            }
        } // Left-click ==> zoom in
        else if (button == MouseButton::BUTTON_LEFT) {
            if (root->left != nullptr && root->right != nullptr) {
                // Check position
                for (HierarchicalClustering::CLUSTERNODE* node : *this->cluster) {
                    auto pos = node->pca2d;
                    double posx = scale((*pos)[0], minX, maxX, (PICSCALING * 0.5) * this->viewport.GetX(),
                        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetX());
                    double posy = scale((*pos)[1], minY, maxY, (PICSCALING * 0.5) * this->viewport.GetY(),
                        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetY());

                    double picwidth = PICSCALING * this->viewport.GetX();
                    double picheight = PICSCALING * this->viewport.GetY();

                    if (this->mouseX > (posx - 0.5 * picwidth) && this->mouseY > (posy - 0.5 * picheight) &&
                        this->mouseX < (posx + 0.5 * picwidth) && this->mouseY < (posy + 0.5 * picheight)) {

                        this->cluster = this->clustering->getClusterNodesOfNode(node);

                        // Wenn leaf
                        if (!(node->left != nullptr && node->right != nullptr)) {
                            node->clusterparent = this->root;
                        }
                        this->root = node;
                        reloadTexures = true;
                        actionavailable = false;
                        return true;
                    }
                }
            }
        }
    } else {
        if (action == MouseButtonAction::RELEASE) {
            actionavailable = true;
        }
        return true;
    }

    return false;
}

/*
 * ClusterRenderer::OnMouseMove
 */
bool ClusterRenderer::OnMouseMove(double x, double y) {
    // Only save actual Mouse Position

    this->mouseX = (float)static_cast<int>(x);
    this->mouseY = (float)static_cast<int>(y);

    return false;
}

bool ClusterRenderer::GetPositionExtents(Call& call) {
    CallClusterPosition* ccp = dynamic_cast<CallClusterPosition*>(&call);
    if (ccp == nullptr) return false;

    // Wenn neuer root node
    if (ccp->getPosition() != this->root) {
        this->hashoffset++;
        this->newposition = true;
        ccp->SetDataHash(this->DataHashPosition + this->hashoffset);
    }
    return true;
}

bool ClusterRenderer::GetPositionData(Call& call) {
    CallClusterPosition* ccp = dynamic_cast<CallClusterPosition*>(&call);
    if (ccp == nullptr) return false;
    if (this->newposition) {
        ccp->setPosition(this->root);
        ccp->setClusterColors(this->colors);
        this->newposition = false;
        this->DataHashPosition += this->hashoffset;
    }
    return true;
}

void ClusterRenderer::renderDistanceIndikator() {
    double height = this->viewport.GetY();
    double width = this->viewport.GetX();

    double indikatorheight = height * 0.02;
    double indikatorwidth = width * 0.2;

    double textheight = this->theFont.LineHeight(this->fontSize);

    glBegin(GL_QUADS);
    glColor3f(1, 1, 1);
    glVertex2f(width - indikatorwidth, indikatorheight + textheight);
    glVertex2f(width - indikatorwidth, 0);
    glVertex2f(width, 0);
    glVertex2f(width, indikatorheight + textheight);
    glEnd();


    // Render Rectangular
    glBegin(GL_QUADS);
    glColor3f(1, 1, 1);
    glVertex2f(width - indikatorwidth, indikatorheight);
    glVertex2f(width - indikatorwidth, 0);
    glColor3f(0, 0, 0);
    glVertex2f(width, 0);
    glVertex2f(width, indikatorheight);
    glEnd();

    // Render Text
    this->renderText(
        "min", width - indikatorwidth, indikatorheight, megamol::core::utility::AbstractFont::ALIGN_LEFT_BOTTOM);
    this->renderText("max", width, indikatorheight, megamol::core::utility::AbstractFont::ALIGN_RIGHT_BOTTOM);
}

void ClusterRenderer::renderText(
    vislib::StringA text, double x, double y, megamol::core::utility::AbstractFont::Alignment alignment) {
    // Render cluster name
    float fgColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    float strHeight = this->theFont.LineHeight(this->fontSize);
    float strWidth = this->theFont.LineWidth(this->fontSize, text);

    // Render Text
    this->theFont.DrawString(fgColor, x, y, this->fontSize, false, text, alignment);
}


std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, ClusterRenderer::RGBCOLOR*>*>*
ClusterRenderer::getNdiffrentColors(std::vector<HierarchicalClustering::CLUSTERNODE*>* cluster) {

    /* Dieser Teil der Funktion ermittelt für jeden Farbkanal einzeln, in wie viele Teile die 255 Werte
    des jeweiligen Farbkanals mindestens zerlegt werden müssen, damit die Funktion die geforderte Anzahl
    Farben durch Permutationen erzeugen kann.
        */

    double red_number, blue_number, green_number;
    int quantity = cluster->size();

    double root = pow(quantity, 1.0 / 3.0);
    if (ceil(root) * pow(floor(root), 2) >= quantity) {
        red_number = ceil(root);
        green_number = blue_number = floor(root);
    } else if (pow(ceil(root), 2) * floor(root) >= quantity) {
        red_number = green_number = ceil(root);
        blue_number = floor(root);
    } else {
        red_number = blue_number = green_number = ceil(root);
    }

    /* Dieser Teil berechnet die Permutationen und bricht ab, wenn genügend erzeugt wurden. */

    std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, RGBCOLOR*>*>* result =
        new std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, RGBCOLOR*>*>();
    int counter = 0;

    for (int red_counter = 0; red_counter <= red_number; red_counter++) {
        for (int green_counter = 0; green_counter <= green_number; green_counter++) {
            for (int blue_counter = 0; blue_counter <= blue_number; blue_counter++) {
                if (counter >= quantity) return result;

                RGBCOLOR* tmp = new RGBCOLOR();
                tmp->r = red_counter * floor(255 / red_number);
                tmp->g = green_counter * floor(255 / green_number);
                tmp->b = blue_counter * floor(255 / blue_number);

                std::tuple<HierarchicalClustering::CLUSTERNODE*, RGBCOLOR*>* tmpcolor =
                    new std::tuple<HierarchicalClustering::CLUSTERNODE*, RGBCOLOR*>();
                std::get<0>(*tmpcolor) = (*cluster)[counter];
                std::get<1>(*tmpcolor) = tmp;

                result->push_back(tmpcolor);
                counter++;
            }
        }
    }
}
