/*
 * ClusterRenderer.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include <tuple>

#include "mmcore/view/Renderer2DModule.h"

#include "vislib/sys/Log.h"

#include "CallClusterPosition.h"
#include "ClusterHierarchieRenderer.h"
#include "TextureLoader.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::MolSurfMapCluster;


/*
 * ClusterRenderer::ClusterRenderer
 */
ClusterHierarchieRenderer::ClusterHierarchieRenderer(void)
    : view::Renderer2DModule()
    , clusterDataSlot("inData", "The input data slot for sphere data.")
    , positionDataSlot("position", "The inoput data slot for the aktual position")
    , positionoutslot("getposition", "Returns the aktual Rendered-Root-Node from clustering")

    , theFont(megamol::core::utility::SDFFont::FontName::ROBOTO_SANS)
    , fontSize(22.0f) {

    // Callee Slot
    this->positionoutslot.SetCallback(CallClusterPosition::ClassName(), CallClusterPosition::FunctionName(1),
        &ClusterHierarchieRenderer::GetPositionExtents);
    this->positionoutslot.SetCallback(CallClusterPosition::ClassName(), CallClusterPosition::FunctionName(0),
        &ClusterHierarchieRenderer::GetPositionData);
    this->MakeSlotAvailable(&this->positionoutslot);

    // CallerSlot
    this->clusterDataSlot.SetCompatibleCall<CallClusteringDescription>();
    this->MakeSlotAvailable(&this->clusterDataSlot);

    this->positionDataSlot.SetCompatibleCall<CallClusterPositionDescription>();
    this->MakeSlotAvailable(&this->positionDataSlot);

    // ParamSlot


    // Variablen
    this->lastHashClustering = 0;
    this->lastHashPosition = 0;
    this->DataHashPosition = 0;
    this->clustering = nullptr;
    this->rendered = false;
    this->newposition = false;
    this->position = nullptr;

    this->newcolor = false;
    this->hashoffset = 0;
    this->colorhash = 0;

    this->actionavailable = true;

    this->popup = nullptr;
    this->x = 0;
    this->y = 0;
}


/*
 * ClusterRenderer::~ClusterRenderer
 */
ClusterHierarchieRenderer::~ClusterHierarchieRenderer(void) { this->Release(); }


/*
 * ClusterRenderer::create
 */
bool ClusterHierarchieRenderer::create(void) {

    // Create Shader
    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    ShaderSource vertextShaderSource;
    ShaderSource fragmentShaderSource;

    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "hierarchieShader::vertex", vertextShaderSource)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for Vertex Shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
            "hierarchieShader::fragment", vertextShaderSource)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for Fragment Shader");
        return false;
    }

    try {
        if (!this->shader.Create(vertextShaderSource.Code(), vertextShaderSource.Count(), fragmentShaderSource.Code(),
                fragmentShaderSource.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create shader: %s\n", e.GetMsgA());
        return false;
    }


    // Initialise font
    if (!this->theFont.Initialise(this->GetCoreInstance())) {
        vislib::sys::Log::DefaultLog.WriteError("Couldn't initialize the font.");
        return false;
    }
    return true;
}


/*
 * ClusterRenderer::release
 */
void ClusterHierarchieRenderer::release(void) {

    // nothing to do here ...
}


/*
 * ClusterRenderer::GetExtents
 */
bool ClusterHierarchieRenderer::GetExtents(view::CallRender2D& call) {

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

    // Check for new Position to render
    CallClusterPosition* ccp = this->positionDataSlot.CallAs<CallClusterPosition>();
    if (ccp == nullptr) return false;

    if (!(*ccp)(CallClusterPosition::CallForGetExtent)) return false;

    // if viewport changes ....
    if (currentViewport != this->viewport) {
        this->viewport = currentViewport;
    }

    return true;
}

double ClusterHierarchieRenderer::drawTree(HierarchicalClustering::CLUSTERNODE* node, double minheight, double minwidth,
    double spacey, double spacex,
    std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, ClusterRenderer::RGBCOLOR*>*>* colors) {

    double posx = 0;
    double posy = 0;
    double posLeft = 0;
    double posRight = 0;

    // draw child node
    if (node->level == 0) {
        posx = minwidth + (counter * spacex);
        posy = minheight + (node->level * spacey);
        this->counter++;

    } else {
        posLeft = drawTree(node->left, minheight, minwidth, spacey, spacex, colors);
        posRight = drawTree(node->right, minheight, minwidth, spacey, spacex, colors);

        posx = (posLeft + posRight) / 2;
        posy = minheight + (node->level * spacey);
    }

    // Select Color
    bool clusternode = false;
    if (this->colors != nullptr) {
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
    }

    if (!clusternode) {
        if (this->clustering->parentIs(node, this->position)) {
            glColor3f(0.5, 0.5, 0.5);
        } else {
            glColor3f(0, 0, 0);
        }
    }


    // Draw Point
    glPointSize(10);
    glBegin(GL_POINTS);
    glVertex2f(posx, posy);
    glEnd();

    if (node->level != 0) {
        // Connect the Nodes
        double posLeftY = minheight + (node->left->level * spacey);
        double posRightY = minheight + (node->right->level * spacey);

        glLineWidth(2);
        glBegin(GL_LINES);
        glVertex2f(posRight, posy);
        glVertex2f(posLeft, posy);

        glVertex2f(posRight, posy);
        glVertex2f(posRight, posRightY);

        glVertex2f(posLeft, posy);
        glVertex2f(posLeft, posLeftY);
        glEnd();
    }
    return posx;
}

/*
 * ClusterRenderer::Render
 */
bool ClusterHierarchieRenderer::Render(view::CallRender2D& call) {

    core::view::CallRender2D* cr = dynamic_cast<core::view::CallRender2D*>(&call);
    if (cr == nullptr) return false;

    // Update data Clustering
    CallClustering* ccc = this->clusterDataSlot.CallAs<CallClustering>();
    if (!ccc) return false;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallClustering::CallForGetData)) return false;

    // Update data Position
    CallClusterPosition* ccp = this->positionDataSlot.CallAs<CallClusterPosition>();
    if (!ccp) return false;
    // Updated data from cinematic camera call
    if (!(*ccp)(CallClusterPosition::CallForGetData)) return false;


    if (ccc->DataHash() != this->lastHashClustering) {
        // update Clustering to work with
        this->clustering = ccc->getClustering();
        this->lastHashClustering = ccc->DataHash();
        this->root = this->clustering->getRoot();
    }

    if (ccp->DataHash() != this->lastHashPosition && !this->newposition) {
        // update Clustering to work with
        this->position = ccp->getPosition();
        this->lastHashPosition = ccp->DataHash();

        this->cluster = this->clustering->getClusterNodesOfNode(this->position);
        this->colors = ccp->getClusterColors();
    }

    if (this->clustering != nullptr) {
        if (this->clustering->finished()) {

            this->counter = 0;

            double height = this->viewport.GetY() * 0.9;
            double width = this->viewport.GetX() * 0.9;

            double minheight = this->viewport.GetY() * 0.05;
            double minwidth = this->viewport.GetX() * 0.05;

            double spacey = height / (this->root->level);
            double spacex = width / (this->clustering->getLeaves()->size() - 1);

            drawTree(this->root, minheight, minwidth, spacey, spacex, colors);

            // Render Popup
            renderPopup();
        }
    }
    return true;
}

/*
 * ClusterHierarchieRenderer::OnMouseButton
 */
bool ClusterHierarchieRenderer::OnMouseButton(megamol::core::view::MouseButton button,
    megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) {

    auto down = action == MouseButtonAction::PRESS;
    this->mouseAction = action;
    this->mouseButton = button;

    if (actionavailable) {
        // Wenn mouse-click auf cluster => change position ...
        // Check position
        this->counter = 0;

        double height = this->viewport.GetY() * 0.9;
        double width = this->viewport.GetX() * 0.9;

        double minheight = this->viewport.GetY() * 0.05;
        double minwidth = this->viewport.GetX() * 0.05;

        double spacey = height / (this->root->level);
        double spacex = width / (this->clustering->getLeaves()->size() - 1);

        if (checkposition(this->root, this->mouseX, this->mouseY, minheight, minwidth, spacey, spacex) == -1) {
            this->position = this->popup;
        }

        // Todo Clusterparent neu berechnen wenn nicht gesetzt...
        auto parent = this->root;
        bool change = false;
        while (this->position->clusterparent == nullptr) {
            change = false;
            auto tmpcluster = this->clustering->getClusterNodesOfNode(parent);
            for (HierarchicalClustering::CLUSTERNODE* node : *tmpcluster) {
                if (this->position == node) {
                    this->position->clusterparent = parent;
                    change = true;
                    break;
                } else if (this->clustering->parentIs(this->position, node)) {
                    parent = node;
                    change = true;
                    break;
                }
            }

            if (!change) {
                this->position->clusterparent = parent;
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
 * ClusterHierarchieRenderer::OnMouseMove
 */
bool ClusterHierarchieRenderer::OnMouseMove(double x, double y) {
    // Only save actual Mouse Position

    this->mouseX = (float)static_cast<int>(x);
    this->mouseY = (float)static_cast<int>(y);

    // Check mouse position => if position is on cluster => render popup
    // first delete old popup boolean
    for (HierarchicalClustering::CLUSTERNODE* node : *this->clustering->getAllNodes()) {
        node->pic->popup = false;
    }
    this->popup = nullptr;

    // Check position
    this->counter = 0;

    double height = this->viewport.GetY() * 0.9;
    double width = this->viewport.GetX() * 0.9;

    double minheight = this->viewport.GetY() * 0.05;
    double minwidth = this->viewport.GetX() * 0.05;

    double spacey = height / (this->root->level);
    double spacex = width / (this->clustering->getLeaves()->size() - 1);

    checkposition(this->root, this->mouseX, this->mouseY, minheight, minwidth, spacey, spacex);

    this->x = this->mouseX;
    this->y = this->mouseY;

    return false;
}

void ClusterHierarchieRenderer::renderPopup() {
    if (this->popup != nullptr) {
        // Load texture if not loaded
        glEnable(GL_TEXTURE_2D);
        this->popup->pic->popup = true;
        TextureLoader::loadTexturesToRender(this->clustering);

        // Render Texture
        // Position -> Mouse Position? Cluster Center?
        int width = 200;
        int height = 100;

        // Berechne verschiebung
        int shiftx = 0;
        int shifty = 0;

        if (this->x + width > this->viewport.GetX()) {
            shiftx = this->viewport.GetX() - (this->x + width);
        }

        if (this->y + height > this->viewport.GetY()) {
            shifty = this->viewport.GetY() - (this->y + height);
        }

        this->popup->pic->texture->Bind();
        glColor3f(1, 1, 1);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2f((this->x + shiftx), (this->y + shifty));

        glTexCoord2f(1, 0);
        glVertex2f((this->x + shiftx) + width, (this->y + shifty));

        glTexCoord2f(1, 1);
        glVertex2f((this->x + shiftx) + width, (this->y + shifty) + height);

        glTexCoord2f(0, 1);
        glVertex2f((this->x + shiftx), (this->y + shifty) + height);
        glEnd();
        glDisable(GL_TEXTURE_2D);
    }
}

double ClusterHierarchieRenderer::checkposition(HierarchicalClustering::CLUSTERNODE* node, float x, float y,
    double minheight, double minwidth, double spacey, double spacex) {

    double posx = 0;
    double posy = 0;
    double posLeft = 0;
    double posRight = 0;

    // draw child node
    if (node->level == 0) {
        posx = minwidth + (counter * spacex);
        posy = minheight + (node->level * spacey);
        this->counter++;

        // Check position => if found return -1;
        if (x > posx - 5 && x < posx + 5 && y > posy - 5 && y < posy + 5) {
            this->popup = node;
            return -1;
        }

    } else {
        posLeft = checkposition(node->left, x, y, minheight, minwidth, spacey, spacex);
        posRight = checkposition(node->right, x, y, minheight, minwidth, spacey, spacex);

        if (posLeft == -1 || posRight == -1) {
            return -1;
        } else {
            // Check position
            // Check position => if found return -1;

            posx = (posLeft + posRight) / 2;
            posy = minheight + (node->level * spacey);

            if (x > posx - 5 && x < posx + 5 && y > posy - 5 && y < posy + 5) {
                this->popup = node;
                return -1;
            }
        }
    }
    return posx;
}


bool ClusterHierarchieRenderer::GetPositionExtents(Call& call) {
    CallClusterPosition* ccp = dynamic_cast<CallClusterPosition*>(&call);
    if (ccp == nullptr) return false;

    // Wenn neuer root node
    if (ccp->getPosition() != this->position) {
        this->hashoffset++;
        this->newposition = true;
        ccp->SetDataHash(this->DataHashPosition + this->hashoffset);
    }
    return true;
}

bool ClusterHierarchieRenderer::GetPositionData(Call& call) {
    CallClusterPosition* ccp = dynamic_cast<CallClusterPosition*>(&call);
    if (ccp == nullptr) return false;
    if (this->newposition) {
        ccp->setPosition(this->position);
        ccp->setClusterColors(nullptr);
        this->newposition = false;
        this->DataHashPosition += this->hashoffset;
    }
    return true;
}
