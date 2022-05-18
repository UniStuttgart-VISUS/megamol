/*
 * ClusterRenderer.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "mmcore_gl/view/Renderer2DModuleGL.h"

#include "mmcore/utility/log/Log.h"

#include "glm/glm.hpp"
#include <filesystem>
#include <glad/gl.h>
#include <istream>
#include <vector>

#include "CallClusterPosition.h"
#include "ClusterHierarchieRenderer.h"
#include "ClusterRenderer.h"
#include "TextureLoader.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/ColourParser.h"

#define LINECOLOR 0.0, 0.0, 1.0
#define BLACK 0, 0, 0
#define PICSCALING 0.2

#define VIEWPORT_WIDTH 2560
#define VIEWPORT_HEIGHT 1440


using namespace megamol;
using namespace megamol::core_gl;
using namespace megamol::core::view;
using namespace megamol::molsurfmapcluster;


/*
 * ClusterRenderer::ClusterRenderer
 */
ClusterRenderer::ClusterRenderer(void)
        : core_gl::view::Renderer2DModuleGL()
        , clusterDataSlot("inData", "The input data slot for sphere data.")
        , getPosition("getPosition", "Returns the aktual Rendered-Root-Node from clustering")
        , setPosition("setPosition", "Set the aktual position-root-node from clustering")
        , colorTableFileParam("colortable", "Path to the file containing an alternative color table")
        , theFont(megamol::core::utility::SDFFont::PRESET_ROBOTO_SANS)
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
    this->colorTableFileParam.SetParameter(
        new core::param::FilePathParam("", core::param::FilePathParam::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&this->colorTableFileParam);

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
ClusterRenderer::~ClusterRenderer(void) {
    this->Release();
}


/*
 * ClusterRenderer::create
 */
bool ClusterRenderer::create(void) {

    // Initialise font
    if (!this->theFont.Initialise(this->GetCoreInstance())) {
        core::utility::log::Log::DefaultLog.WriteError("Couldn't initialize the font.");
        return false;
    }

    vislib_gl::graphics::gl::ShaderSource texVertShader;
    vislib_gl::graphics::gl::ShaderSource texFragShader;

    auto ssf = std::make_shared<core_gl::utility::ShaderSourceFactory>(instance()->Configuration().ShaderDirectories());

    if (!ssf->MakeShaderSource("molsurfTexture::vertex", texVertShader)) {
        core::utility::log::Log::DefaultLog.WriteMsg(
            core::utility::log::Log::LEVEL_ERROR, "Unable to load vertex shader source for texture Vertex Shader");
        return false;
    }
    if (!ssf->MakeShaderSource("molsurfTexture::fragment", texFragShader)) {
        core::utility::log::Log::DefaultLog.WriteMsg(
            core::utility::log::Log::LEVEL_ERROR, "Unable to load fragment shader source for texture Fragment Shader");
        return false;
    }

    try {
        if (!this->textureShader.Create(
                texVertShader.Code(), texVertShader.Count(), texFragShader.Code(), texFragShader.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        core::utility::log::Log::DefaultLog.WriteError("Unable to create shader: %s\n", e.GetMsgA());
        return false;
    }

    if (!ssf->MakeShaderSource("molsurfPassthrough::vertex", texVertShader)) {
        core::utility::log::Log::DefaultLog.WriteMsg(
            core::utility::log::Log::LEVEL_ERROR, "Unable to load vertex shader source for passthrough Vertex Shader");
        return false;
    }
    if (!ssf->MakeShaderSource("molsurfPassthrough::fragment", texFragShader)) {
        core::utility::log::Log::DefaultLog.WriteMsg(core::utility::log::Log::LEVEL_ERROR,
            "Unable to load fragment shader source for passthrough Fragment Shader");
        return false;
    }

    try {
        if (!this->passthroughShader.Create(
                texVertShader.Code(), texVertShader.Count(), texFragShader.Code(), texFragShader.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    } catch (vislib::Exception e) {
        core::utility::log::Log::DefaultLog.WriteError("Unable to create shader: %s\n", e.GetMsgA());
        return false;
    }

    const float size = 1.0f;
    std::vector<float> texVerts = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, size, 0.0f, 0.0f, 0.0f, size, 0.0f, 0.0f, 1.0f,
        1.0f, size, size, 0.0f, 1.0f, 0.0f};

    this->texBuffer = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, texVerts, GL_STATIC_DRAW);
    this->geometrySSBO =
        std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, std::vector<glm::vec4>(), GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &this->dummyVa);

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
bool ClusterRenderer::GetExtents(core_gl::view::CallRender2DGL& call) {

    // Incoming Call
    core_gl::view::CallRender2DGL* cr = dynamic_cast<core_gl::view::CallRender2DGL*>(&call);
    if (cr == nullptr)
        return false;

    vislib::math::Vector<float, 2> currentViewport;
    currentViewport.SetX(static_cast<float>(VIEWPORT_WIDTH));
    currentViewport.SetY(static_cast<float>(VIEWPORT_HEIGHT));

    cr->AccessBoundingBoxes().SetBoundingBox(0, 0, currentViewport.GetX(), currentViewport.GetY());

    // Check for new Data in clustering
    CallClustering* cc = this->clusterDataSlot.CallAs<CallClustering>();
    if (cc == nullptr)
        return false;

    if (!(*cc)(CallClustering::CallForGetExtent))
        return false;

    // Check for new Position from HierarchieClusterRenderer
    CallClusterPosition* ccp = this->setPosition.CallAs<CallClusterPosition>();
    if (ccp == nullptr)
        return false;

    if (!(*ccp)(CallClusterPosition::CallForGetExtent))
        return false;

    // if viewport changes ....
    if (currentViewport != this->viewport) {
        this->viewport = currentViewport;
    }

    return true;
}

double scale(double unscaled, double min, double max, double minAllowed, double maxAllowed) {
    return minAllowed + (((unscaled - min) * (maxAllowed - minAllowed)) / (max - min));
}

void ClusterRenderer::renderClusterText(
    HierarchicalClustering::CLUSTERNODE* node, glm::mat4 mvp, double posx, double posy) {
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

    std::vector<glm::vec4> data(6);
    data[0] = glm::vec4(posx - 0.5f * strWidth, posy - 0.5f * strHeight, 0.0f, 1.0f);
    data[1] = glm::vec4(posx + 0.5f * strWidth, posy - 0.5f * strHeight, 0.0f, 1.0f);
    data[2] = glm::vec4(posx + 0.5f * strWidth, posy + 0.5f * strHeight, 0.0f, 1.0f);
    data[3] = glm::vec4(posx - 0.5f * strWidth, posy - 0.5f * strHeight, 0.0f, 1.0f);
    data[4] = glm::vec4(posx + 0.5f * strWidth, posy + 0.5f * strHeight, 0.0f, 1.0f);
    data[5] = glm::vec4(posx - 0.5f * strWidth, posy + 0.5f * strHeight, 0.0f, 1.0f);

    this->geometrySSBO->rebuffer(data);
    this->geometrySSBO->bind(11);

    glBindVertexArray(this->dummyVa);
    this->passthroughShader.Enable();
    glUniformMatrix4fv(this->passthroughShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform4f(this->passthroughShader.ParameterLocation("color"), 1.0f, 1.0f, 1.0f, 1.0f);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    this->passthroughShader.Disable();
    glBindVertexArray(0);

    // Render Text
    this->theFont.DrawString(
        mvp, fgColor, posx, posy, this->fontSize, false, id, megamol::core::utility::SDFFont::ALIGN_CENTER_MIDDLE);
}

void ClusterRenderer::renderLeaveNode(HierarchicalClustering::CLUSTERNODE* node, glm::mat4 mvp) {

    // Render cluster name
    float fgColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float size = this->fontSize + 15;
    vislib::StringA id;
    id.Format("PDB-ID: ");
    id.Append(std::filesystem::path(node->pic->path).stem().string().c_str());

    float strHeight = this->theFont.LineHeight(size);
    float strWidth = this->theFont.LineWidth(size, id);

    this->theFont.DrawString(mvp, fgColor, 0.5 * this->viewport.GetX() - 0.5 * strWidth,
        this->viewport.GetY() - strHeight, size, false, id, megamol::core::utility::SDFFont::ALIGN_CENTER_BOTTOM);

    // Render Bild
    glDisable(GL_CULL_FACE);
    glActiveTexture(GL_TEXTURE5);
    node->pic->texture->bindTexture();
    glEnable(GL_TEXTURE_2D);

    auto pos = node->pca2d;

    double posx = 0.5 * this->viewport.GetX();
    double posy = 0.5 * this->viewport.GetY();

    double picwidth = 0.9 * this->viewport.GetX();
    double picheight = 0.9 * this->viewport.GetY();

    glBindVertexArray(this->texVa);
    this->textureShader.Enable();

    glUniform2f(this->textureShader.ParameterLocation("lowerleft"), posx - 0.5 * picwidth, posy - 0.5 * picheight);
    glUniform2f(
        this->textureShader.ParameterLocation("upperright"), posx + 0.5 * picwidth, this->viewport.GetY() - strHeight);
    glUniformMatrix4fv(this->textureShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1i(this->textureShader.ParameterLocation("tex"), 5);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    this->textureShader.Disable();
    glBindVertexArray(0);

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_CULL_FACE);
}

void ClusterRenderer::renderNode(
    HierarchicalClustering::CLUSTERNODE* node, glm::mat4 mvp, double minX, double maxX, double minY, double maxY) {

    glDisable(GL_CULL_FACE);
    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE5);
    node->pic->texture->bindTexture();

    auto pos = node->pca2d;

    double posx = scale((*pos)[0], minX, maxX, (PICSCALING * 0.5) * this->viewport.GetX(),
        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetX());
    double posy = scale((*pos)[1], minY, maxY, (PICSCALING * 0.5) * this->viewport.GetY(),
        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetY());

    double picwidth = PICSCALING * this->viewport.GetX();
    double picheight = PICSCALING * this->viewport.GetY();

    glBindVertexArray(this->texVa);
    this->textureShader.Enable();

    glUniform2f(this->textureShader.ParameterLocation("lowerleft"), posx - 0.5 * picwidth, posy - 0.5 * picheight);
    glUniform2f(this->textureShader.ParameterLocation("upperright"), posx + 0.5 * picwidth, posy + 0.5 * picheight);
    glUniformMatrix4fv(this->textureShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1i(this->textureShader.ParameterLocation("tex"), 5);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    this->textureShader.Disable();
    glBindVertexArray(0);

    // Cluster beschriftung
    this->renderClusterText(node, mvp, posx, posy);

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_CULL_FACE);
}

void ClusterRenderer::renderAllLeaves(
    HierarchicalClustering::CLUSTERNODE* node, glm::mat4 mvp, double minX, double maxX, double minY, double maxY) {

    if (node->left != nullptr && node->right != nullptr) {
        auto leaves = this->clustering->getLeavesOfNode(node);

        // Select Color
        bool clusternode = false;
        glm::vec3 rescolor;
        for (std::tuple<HierarchicalClustering::CLUSTERNODE*, ClusterRenderer::RGBCOLOR*>* colortuple : *colors) {
            if (this->clustering->parentIs(node, std::get<0>(*colortuple))) {
                ClusterRenderer::RGBCOLOR* color = std::get<1>(*colortuple);
                double r = (color->r) / 255;
                double g = (color->g) / 255;
                double b = (color->b) / 255;
                rescolor = glm::vec3(r, g, b);
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
            this->geometrySSBO->rebuffer(std::vector<glm::vec4>{glm::vec4(posx, posy, 0.0f, 1.0f)});
            this->geometrySSBO->bind(11);

            glBindVertexArray(this->dummyVa);
            this->passthroughShader.Enable();
            glUniformMatrix4fv(this->passthroughShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
            glUniform4f(this->passthroughShader.ParameterLocation("color"), rescolor.r, rescolor.g, rescolor.b, 1.0f);
            glDrawArrays(GL_POINTS, 0, 1);
            this->passthroughShader.Disable();
            glBindVertexArray(0);
        }
    }
}

void ClusterRenderer::DrawCircle(glm::mat4 mvp, float cx, float cy, float r, int num_segments) {
    std::vector<glm::vec4> data(num_segments);
    for (int ii = 0; ii < num_segments; ii++) {
        float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments); // get the current angle
        float x = r * cosf(theta);                                         // calculate the x component
        float y = r * sinf(theta);                                         // calculate the y component
        data[ii] = glm::vec4(x + cx, y + cy, 0.0f, 1.0f);
    }

    this->geometrySSBO->rebuffer(data);
    this->geometrySSBO->bind(11);

    glBindVertexArray(this->dummyVa);
    this->passthroughShader.Enable();
    glUniformMatrix4fv(this->passthroughShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform4f(this->passthroughShader.ParameterLocation("color"), 0.0f, 0.0f, 0.0f, 1.0f);
    glDrawArrays(GL_LINE_LOOP, 0, num_segments);
    this->passthroughShader.Disable();
    glBindVertexArray(0);
}

void ClusterRenderer::renderRootNode(
    HierarchicalClustering::CLUSTERNODE* node, glm::mat4 mvp, double minX, double maxX, double minY, double maxY) {
    auto pos = node->pca2d;

    double posx = scale((*pos)[0], minX, maxX, (PICSCALING * 0.5) * this->viewport.GetX(),
        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetX());
    double posy = scale((*pos)[1], minY, maxY, (PICSCALING * 0.5) * this->viewport.GetY(),
        (1.0 - (PICSCALING * 0.5)) * this->viewport.GetY());

    double picwidth = PICSCALING * this->viewport.GetX();
    double picheight = PICSCALING * this->viewport.GetY();

    // Render Cluster Mittelpunklt
    this->DrawCircle(mvp, posx, posy, this->viewport.GetY() * 0.02, 100000);
    this->renderClusterText(node, mvp, posx, posy);
}


void ClusterRenderer::connectNodes(
    HierarchicalClustering::CLUSTERNODE* node1, HierarchicalClustering::CLUSTERNODE* node2, glm::mat4 mvp) {

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
    std::vector<glm::vec4> data = {glm::vec4(posx1, posy1, 0.0f, 1.0f), glm::vec4(posx2, posy2, 0.0f, 1.0f)};
    this->geometrySSBO->rebuffer(data);
    this->geometrySSBO->bind(11);

    glBindVertexArray(this->dummyVa);
    this->passthroughShader.Enable();
    glUniformMatrix4fv(this->passthroughShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform4f(this->passthroughShader.ParameterLocation("color"), 1 - tmp, 1 - tmp, 1 - tmp, 1.0f);
    glDrawArrays(GL_LINES, 0, 2);
    this->passthroughShader.Disable();
    glBindVertexArray(0);
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

        if (x > this->maxX)
            this->maxX = x;
        if (x < this->minX)
            this->minX = x;
        if (y > this->maxY)
            this->maxY = y;
        if (y < this->minY)
            this->minY = y;
    }
}

/*
 * ClusterRenderer::Render
 */
bool ClusterRenderer::Render(core_gl::view::CallRender2DGL& call) {

    core_gl::view::CallRender2DGL* cr = dynamic_cast<core_gl::view::CallRender2DGL*>(&call);
    if (cr == nullptr)
        return false;

    // Update data
    CallClustering* ccc = this->clusterDataSlot.CallAs<CallClustering>();
    if (!ccc)
        return false;
    if (!(*ccc)(CallClustering::CallForGetData))
        return false;

    if (this->colorTableFileParam.IsDirty()) {
        this->colorTableFileParam.ResetDirty();
        this->colortab = this->loadColorTable();
    }

    // read matrices (old bullshit)
    GLfloat viewMatrixColumn[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrixColumn);
    glm::mat4 view = glm::make_mat4(viewMatrixColumn);
    GLfloat projMatrixColumn[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrixColumn);
    glm::mat4 proj = glm::make_mat4(projMatrixColumn);
    glm::mat4 mvp = proj * view;

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
                if (!ccp) {
                    core::utility::log::Log::DefaultLog.WriteError(
                        "CallClusterPosition not connected, Module might not work correctly");
                    return false;
                }
                if (!(*ccp)(CallClusterPosition::CallForGetData))
                    return false;

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
                    this->connectNodes(root, node, mvp);
                }

                // Render Cluster
                for (HierarchicalClustering::CLUSTERNODE* node : *this->cluster) {
                    this->renderNode(node, mvp, this->minX, this->maxX, this->minY, this->maxY);
                }

                // Render Clusterchildren
                for (HierarchicalClustering::CLUSTERNODE* node : *this->cluster) {
                    this->renderAllLeaves(node, mvp, this->minX, this->maxX, this->minY, this->maxY);
                }

                // Render Distance indikator
                this->renderDistanceIndikator(mvp);
            } else {
                this->renderLeaveNode(root, mvp);
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

bool ClusterRenderer::GetPositionExtents(core::Call& call) {
    CallClusterPosition* ccp = dynamic_cast<CallClusterPosition*>(&call);
    if (ccp == nullptr) {
        core::utility::log::Log::DefaultLog.WriteError(
            "CallClusterPosition not connected, Module might not work correctly.");
        return false;
    }

    // Wenn neuer root node
    if (ccp->getPosition() != this->root) {
        this->hashoffset++;
        this->newposition = true;
        ccp->SetDataHash(this->DataHashPosition + this->hashoffset);
    }
    return true;
}

bool ClusterRenderer::GetPositionData(core::Call& call) {
    CallClusterPosition* ccp = dynamic_cast<CallClusterPosition*>(&call);
    if (ccp == nullptr)
        return false;
    ccp->setPosition(this->root);
    ccp->setClusterColors(this->colors);
    if (this->newposition) {
        this->newposition = false;
        this->DataHashPosition += this->hashoffset;
    }
    return true;
}

void ClusterRenderer::renderDistanceIndikator(glm::mat4 mvp) {
    double height = this->viewport.GetY();
    double width = this->viewport.GetX();

    double indikatorheight = height * 0.02;
    double indikatorwidth = width * 0.2;

    double textheight = this->theFont.LineHeight(this->fontSize);

    glm::vec4 white = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    glm::vec4 black = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    std::vector<glm::vec4> data(16);
    data[0] = glm::vec4(width - indikatorwidth, indikatorheight + textheight, 0.0f, 1.0f);
    data[1] = white;
    data[2] = glm::vec4(width - indikatorwidth, 0, 0.0f, 1.0f);
    data[3] = white;
    data[4] = glm::vec4(width, 0, 0.0f, 1.0f);
    data[5] = white;
    data[6] = glm::vec4(width, indikatorheight + textheight, 0.0f, 1.0f);
    data[7] = white;
    data[8] = glm::vec4(width - indikatorwidth, indikatorheight, 0.0f, 1.0f);
    data[9] = white;
    data[10] = glm::vec4(width - indikatorwidth, 0, 0.0f, 1.0f);
    data[11] = white;
    data[12] = glm::vec4(width, 0, 0.0f, 1.0f);
    data[13] = black;
    data[14] = glm::vec4(width, indikatorheight, 0.0f, 1.0f);
    data[15] = black;

    this->geometrySSBO->rebuffer(data);
    this->geometrySSBO->bind(11);

    glBindVertexArray(this->dummyVa);
    this->passthroughShader.Enable();
    glUniformMatrix4fv(this->passthroughShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1i(this->passthroughShader.ParameterLocation("coloredmode"), 1);
    glDrawArrays(GL_QUADS, 0, 8);
    glUniform1i(this->passthroughShader.ParameterLocation("coloredmode"), 0);
    this->passthroughShader.Disable();
    glBindVertexArray(0);

    // Render Text
    this->renderText(
        "min", mvp, width - indikatorwidth, indikatorheight, megamol::core::utility::SDFFont::ALIGN_LEFT_BOTTOM);
    this->renderText("max", mvp, width, indikatorheight, megamol::core::utility::SDFFont::ALIGN_RIGHT_BOTTOM);
}

void ClusterRenderer::renderText(
    vislib::StringA text, glm::mat4 mvp, double x, double y, megamol::core::utility::SDFFont::Alignment alignment) {
    // Render cluster name
    float fgColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    float strHeight = this->theFont.LineHeight(this->fontSize);
    float strWidth = this->theFont.LineWidth(this->fontSize, text);

    // Render Text
    glDisable(GL_CULL_FACE);
    this->theFont.DrawString(mvp, fgColor, x, y, this->fontSize, false, text, alignment);
    glEnable(GL_CULL_FACE);
}


std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, ClusterRenderer::RGBCOLOR*>*>*
ClusterRenderer::getNdiffrentColors(std::vector<HierarchicalClustering::CLUSTERNODE*>* cluster) {

    std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, RGBCOLOR*>*>* result =
        new std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, RGBCOLOR*>*>();
    int counter = 0;
    int quantity = cluster->size();

    if (!this->colortab.empty()) {
        for (int i = 0; i < quantity; ++i) {
            int index = i % colortab.size();
            RGBCOLOR* tmp = new RGBCOLOR();
            tmp->r = colortab[index].r;
            tmp->g = colortab[index].g;
            tmp->b = colortab[index].b;

            std::tuple<HierarchicalClustering::CLUSTERNODE*, RGBCOLOR*>* tmpcolor =
                new std::tuple<HierarchicalClustering::CLUSTERNODE*, RGBCOLOR*>();
            std::get<0>(*tmpcolor) = (*cluster)[i];
            std::get<1>(*tmpcolor) = tmp;
            result->push_back(tmpcolor);
        }
        return result;
    }


    /* Dieser Teil der Funktion ermittelt für jeden Farbkanal einzeln, in wie viele Teile die 255 Werte
    des jeweiligen Farbkanals mindestens zerlegt werden müssen, damit die Funktion die geforderte Anzahl
    Farben durch Permutationen erzeugen kann.
        */

    double red_number, blue_number, green_number;

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

    for (int red_counter = 0; red_counter <= red_number; red_counter++) {
        for (int green_counter = 0; green_counter <= green_number; green_counter++) {
            for (int blue_counter = 0; blue_counter <= blue_number; blue_counter++) {
                if (counter >= quantity)
                    return result;

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

std::vector<glm::uvec4> ClusterRenderer::loadColorTable(void) {
    std::vector<glm::uvec4> result;
    auto path = this->colorTableFileParam.Param<core::param::FilePathParam>()->Value();
    std::string pstring = path.string();
    if (pstring.empty())
        return result;
    std::ifstream file(pstring);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                std::array<unsigned char, 3> col;
                if (core::utility::ColourParser::FromString(line.c_str(), 3, col.data())) {
                    result.push_back(glm::uvec4(col[0], col[1], col[2], 255));
                }
            }
        }
        file.close();
    }
    return result;
}
