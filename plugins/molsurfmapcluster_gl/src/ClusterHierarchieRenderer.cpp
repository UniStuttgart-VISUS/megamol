/*
 * ClusterRenderer.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include <tuple>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore_gl/view/Renderer2DModuleGL.h"

#include "mmcore/utility/log/Log.h"

#include "CallClusterPosition.h"
#include "ClusterHierarchieRenderer.h"
#include "DBScanClusteringProvider.h"
#include "DistanceMatrixLoader.h"
#include "EnzymeClassProvider.h"
#include "TextureLoader.h"

#define VIEWPORT_WIDTH 2560
#define VIEWPORT_HEIGHT 1440

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core_gl;
using namespace megamol::core::view;
using namespace megamol::molsurfmapcluster;
using namespace megamol::molsurfmapcluster_gl;


/*
 * ClusterRenderer::ClusterRenderer
 */
ClusterHierarchieRenderer::ClusterHierarchieRenderer(void)
        : core_gl::view::Renderer2DModuleGL()
        , clusterDataSlot("inData", "The input data slot for sphere data.")
        , positionDataSlot("position", "The inoput data slot for the aktual position")
        , positionoutslot("getposition", "Returns the aktual Rendered-Root-Node from clustering")
        , showEnzymeClassesParam("showEnzymeClasses", "Display the Enzyme classes alongside with the popup renders")
        , showPDBIdsParam("showPDBIds", "Display the PDB Ids alongside with the popup renders")
        , fontSizeParam("fontSize", "Size of the rendered font")
        , useDistanceColors("useDistanceColors", "Use the distance colors instead of the coloring by cluster")
        , minColorParam("color::minColor", "the minimum color for interpolation")
        , midColorParam("color::midColor", "the mid color for interpolation")
        , maxColorParam("color::maxColor", "the maximum color for interpolation")
        , failColorParam("color::failColor", "color used for failed comparisons")
        , windowHeightParam("window::height", "height of the displayed window")
        , windowWidthParam("window::width", "width of the displayed window")
        , addBrendaParam("addparam::brendaClass", "Additionally display the brenda class below each leaf node")
        , addMapParam("addparam::map", "Additionally display the map below each leaf node")
        , addIdParam("addparam::pdbId", "Additionally display the pdb id below each leaf node")
        , distanceMatrixParam("distanceMatrixFile", "Path of the file containing a distance matrix to compare")
        , sizeMultiplierParam("sizeMultiplier", "Factor that is able to tweak the size of drawn GL_POINTS and GL_LINES")
        , theFont(megamol::core::utility::SDFFont::PRESET_ROBOTO_SANS)
        , texVa(0) {

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
    this->showEnzymeClassesParam.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->showEnzymeClassesParam);

    this->showPDBIdsParam.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->showPDBIdsParam);

    this->fontSizeParam.SetParameter(new core::param::FloatParam(100.0f, 5.0f, 300.0f));
    this->MakeSlotAvailable(&this->fontSizeParam);

    core::param::EnumParam* enpar = new core::param::EnumParam(static_cast<int>(DistanceColorMode::NONE));
    enpar->SetTypePair(static_cast<int>(DistanceColorMode::NONE), "None");
    enpar->SetTypePair(static_cast<int>(DistanceColorMode::BRENDA), "BRENDA");
    enpar->SetTypePair(static_cast<int>(DistanceColorMode::TMSCORE), "TM-Score");
    this->useDistanceColors.SetParameter(enpar);
    this->MakeSlotAvailable(&this->useDistanceColors);

    this->minColorParam.SetParameter(new param::ColorParam(0.266666f, 0.05098f, 0.32941f, 1.0f));
    this->MakeSlotAvailable(&this->minColorParam);

    this->midColorParam.SetParameter(new param::ColorParam(0.127255f, 0.54117f, 0.55294f, 1.0f));
    this->MakeSlotAvailable(&this->midColorParam);

    this->maxColorParam.SetParameter(new param::ColorParam(0.992157f, 0.90588f, 0.14509f, 1.0f));
    this->MakeSlotAvailable(&this->maxColorParam);

    this->failColorParam.SetParameter(new param::ColorParam(0.2f, 0.2f, 0.2f, 1.0f));
    this->MakeSlotAvailable(&this->failColorParam);

    this->windowHeightParam.SetParameter(new param::IntParam(VIEWPORT_HEIGHT, 1000, 20000));
    this->MakeSlotAvailable(&this->windowHeightParam);

    this->windowWidthParam.SetParameter(new param::IntParam(VIEWPORT_WIDTH, 1000, 20000));
    this->MakeSlotAvailable(&this->windowWidthParam);

    this->addBrendaParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->addBrendaParam);

    this->addIdParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->addIdParam);

    this->addMapParam.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->addMapParam);

    this->distanceMatrixParam.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->distanceMatrixParam);

    this->sizeMultiplierParam.SetParameter(new param::FloatParam(1.0, 0.5, 20.0));
    this->MakeSlotAvailable(&this->sizeMultiplierParam);

    // Variablen
    this->lastHashClustering = 0;
    this->lastHashPosition = 0;
    this->DataHashPosition = 0;
    this->clustering = nullptr;
    this->rendered = false;
    this->newposition = false;
    this->position = nullptr;
    this->root = nullptr;
    this->leftmarked = nullptr;
    this->rightmarked = nullptr;

    this->newcolor = false;
    this->hashoffset = 0;
    this->colorhash = 0;

    this->dbscanclustercolor = false;

    this->popup = nullptr;
    this->x = 0;
    this->y = 0;
}


/*
 * ClusterRenderer::~ClusterRenderer
 */
ClusterHierarchieRenderer::~ClusterHierarchieRenderer(void) {
    this->Release();
}


/*
 * ClusterRenderer::create
 */
bool ClusterHierarchieRenderer::create(void) {

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

    texVertShader.Clear();
    texFragShader.Clear();

    if (!ssf->MakeShaderSource("molsurfPassthrough::vertex", texVertShader)) {
        core::utility::log::Log::DefaultLog.WriteMsg(
            core::utility::log::Log::LEVEL_ERROR, "Unable to load vertex shader source for passthrough Vertex Shader");
        return false;
    }
    if (!ssf->MakeShaderSource(
            "molsurfPassthrough::fragment", texFragShader)) {
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
    this->geometrySSBO = std::make_unique<glowl::BufferObject>(
        GL_SHADER_STORAGE_BUFFER, std::vector<glm::vec3>(), GL_DYNAMIC_DRAW); // this is filled later

    glGenVertexArrays(1, &this->dummyVa);
    // it is only a dummy, so we do not fill it

    glGenVertexArrays(1, &this->texVa);
    glBindVertexArray(this->texVa);

    this->texBuffer->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*) (3 * sizeof(float)));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    return true;
}


/*
 * ClusterRenderer::release
 */
void ClusterHierarchieRenderer::release(void) {
    if (this->texVa != 0) {
        glDeleteVertexArrays(1, &this->texVa);
    }
}


/*
 * ClusterRenderer::GetExtents
 */
bool ClusterHierarchieRenderer::GetExtents(core_gl::view::CallRender2DGL& call) {

    this->windowMeasurements = call.GetViewResolution();

    vislib::math::Vector<float, 2> currentViewport;
    currentViewport.SetX(static_cast<float>(this->windowWidthParam.Param<param::IntParam>()->Value()));
    currentViewport.SetY(static_cast<float>(this->windowHeightParam.Param<param::IntParam>()->Value()));

    call.AccessBoundingBoxes().SetBoundingBox(0, 0, currentViewport.GetX(), currentViewport.GetY());

    // Check for new Data in clustering
    CallClustering* cc = this->clusterDataSlot.CallAs<CallClustering>();
    if (cc == nullptr)
        return false;

    if (!(*cc)(CallClustering::CallForGetExtent))
        return false;

    // Check for new Position to render
    CallClusterPosition* ccp = this->positionDataSlot.CallAs<CallClusterPosition>();
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

double ClusterHierarchieRenderer::drawTree(HierarchicalClustering::CLUSTERNODE* node, glm::mat4 mvp, double minheight,
    double minwidth, double spacey, double spacex,
    std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, ClusterRenderer::RGBCOLOR*>*>* colors) {

    double posx = 0;
    double posy = 0;
    double posLeft = 0;
    double posRight = 0;

    double totalheight = 0.9 * this->viewport.GetY();
    double maxheight = this->root->height;
    double myheight = node->height / maxheight;

    // draw child node
    if (node->level == 0) {
        posx = minwidth + (counter * spacex);
        this->counter++;

    } else {
        posLeft = drawTree(node->left, mvp, minheight, minwidth, spacey, spacex, colors);
        posRight = drawTree(node->right, mvp, minheight, minwidth, spacey, spacex, colors);
        posx = (posLeft + posRight) / 2;
    }
    // posy = minheight + (node->level * spacey);
    posy = minheight + myheight * totalheight;

    // Select Color
    bool clusternode = false;
    glm::vec4 currentcolor;
    if (colors != nullptr) {
        for (std::tuple<HierarchicalClustering::CLUSTERNODE*, ClusterRenderer::RGBCOLOR*>* colortuple : *colors) {
            if (this->clustering->parentIs(node, std::get<0>(*colortuple))) {
                ClusterRenderer::RGBCOLOR* color = std::get<1>(*colortuple);
                double r = (color->r) / 255;
                double g = (color->g) / 255;
                double b = (color->b) / 255;
                currentcolor = glm::vec4(r, g, b, 1.0f);
                clusternode = true;
            }
        }
    }

    auto enval = this->useDistanceColors.Param<param::EnumParam>()->Value();
    if (enval != static_cast<int>(DistanceColorMode::NONE)) {
        std::string leftpdb = node->left == nullptr ? node->pic->pdbid : node->left->pic->pdbid;
        std::string rightpdb = node->right == nullptr ? node->pic->pdbid : node->right->pic->pdbid;

        bool isleaf = true;
        if (node->left != nullptr && node->left->left != nullptr)
            isleaf = false;
        if (node->right != nullptr && node->right->left != nullptr)
            isleaf = false;

        float mindist;
        float middle;
        if (enval == static_cast<int>(DistanceColorMode::BRENDA)) {
            auto leftvec = EnzymeClassProvider::RetrieveClassesForPdbId(leftpdb, *this->GetCoreInstance());
            auto rightvec = EnzymeClassProvider::RetrieveClassesForPdbId(rightpdb, *this->GetCoreInstance());

            mindist = 10.0f;
            for (const auto& l : leftvec) {
                for (const auto& r : rightvec) {
                    auto d = this->enzymeClassDistance(l, r);
                    if (d < mindist)
                        mindist = d;
                }
            }
            middle = 2.0f;
            if (!isleaf)
                mindist = -1.0f;
        } else {
            mindist = DistanceMatrixLoader::GetDistance(leftpdb, rightpdb);
            if (mindist >= 0.0)
                mindist = 1.0 - mindist;
            middle = 0.5f;
            if (!isleaf)
                mindist = -1.0f;
        }

        auto minColor = glm::make_vec4(this->minColorParam.Param<param::ColorParam>()->Value().data());
        auto midColor = glm::make_vec4(this->midColorParam.Param<param::ColorParam>()->Value().data());
        auto maxColor = glm::make_vec4(this->maxColorParam.Param<param::ColorParam>()->Value().data());
        auto failColor = glm::make_vec4(this->failColorParam.Param<param::ColorParam>()->Value().data());

        if (mindist > 5.0f || mindist < 0.0f) {
            currentcolor = failColor;
        } else {
            if (mindist <= middle) {
                currentcolor = glm::mix(maxColor, midColor, mindist / 2.0f);
            } else {
                currentcolor = glm::mix(midColor, minColor, (mindist - middle) / 2.0f);
            }
        }
    }

    if (this->dbscanclustercolor) {
        if (this->dbscancluster.count(node->pic->pdbid) > 0) {
            currentcolor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
        } else {
            currentcolor = glm::vec4(0.3f, 0.3f, 0.3f, 1.0f);
        }
    }

    // Draw Point
    glPointSize(10 * this->sizeMultiplierParam.Param<param::FloatParam>()->Value());
    glBindVertexArray(this->dummyVa);
    this->passthroughShader.Enable();

    if (!clusternode) {
        if (this->clustering->parentIs(node, this->position)) {
            currentcolor = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
        } else {
            currentcolor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }
    glUniform4f(this->passthroughShader.ParameterLocation("color"), currentcolor.x, currentcolor.y, currentcolor.z,
        currentcolor.w);
    glUniformMatrix4fv(this->passthroughShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));

    this->geometrySSBO->rebuffer(std::vector<glm::vec4>{glm::vec4(posx, posy, 0.0f, 1.0f)});
    this->geometrySSBO->bind(11);
    glDrawArrays(GL_POINTS, 0, 1);

    if (node->level != 0) {
        // Connect the Nodes
        // double posLeftY = minheight + (node->left->level * spacey);
        // double posRightY = minheight + (node->right->level * spacey);
        double posLeftY = minheight + (node->left->height / maxheight) * totalheight;
        double posRightY = minheight + (node->right->height / maxheight) * totalheight;

        glLineWidth(2 * this->sizeMultiplierParam.Param<param::FloatParam>()->Value());
        std::vector<glm::vec4> data(6);
        data[0] = glm::vec4(posRight, posy, 0.0f, 1.0f);
        data[1] = glm::vec4(posLeft, posy, 0.0f, 1.0f);
        data[2] = glm::vec4(posRight, posy, 0.0f, 1.0f);
        data[3] = glm::vec4(posRight, posRightY, 0.0f, 1.0f);
        data[4] = glm::vec4(posLeft, posy, 0.0f, 1.0f);
        data[5] = glm::vec4(posLeft, posLeftY, 0.0f, 1.0f);
        this->geometrySSBO->rebuffer(data);
        glUniformMatrix4fv(this->passthroughShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
        glDrawArrays(GL_LINES, 0, 6);
    } else {

        float top = 0.0f;
        float width = spacex * 0.95f;
        float height = 0.5 * width;
        float xp = posx - width * 0.5f;
        float yp = posy - height * 0.55f;

        // draw stuff like the pdb id and brenda class
        if (this->addMapParam.Param<param::BoolParam>()->Value()) {
            glDisable(GL_CULL_FACE);
            glEnable(GL_TEXTURE_2D);
            // TextureLoader::loadTexturesToRender(this->clustering);
            // TODO load correct texture

            if (node->pic->texture == nullptr) {
                glowl::TextureLayout layout(
                    GL_RGB8, node->pic->width, node->pic->height, 1, GL_RGB, GL_UNSIGNED_BYTE, 1);
                node->pic->texture =
                    std::make_unique<glowl::Texture2D>("", layout, node->pic->image->PeekDataAs<BYTE>());
            }

            node->pic->texture->bindTexture();

            glBindVertexArray(this->texVa);
            this->textureShader.Enable();

            glUniform2f(this->textureShader.ParameterLocation("lowerleft"), xp, yp - height);
            glUniform2f(this->textureShader.ParameterLocation("upperright"), xp + width, yp);
            glUniformMatrix4fv(this->textureShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
            glUniform1i(this->textureShader.ParameterLocation("tex"), 0);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

            this->textureShader.Disable();
            glBindVertexArray(0);
            glDisable(GL_TEXTURE_2D);
            yp -= height * 1.55f;
        }

        if (this->addIdParam.Param<param::BoolParam>()->Value()) {
            auto substr = node->pic->pdbid.substr(0, 4);
            std::string leftsubstr = "", rightsubstr = "";
            auto stringToDraw = substr.c_str();
            auto lineWidth = theFont.LineWidth(height, stringToDraw);

            std::array<float, 4> black = {0.1f, 0.1f, 0.1f, 1.0f};
            std::array<float, 4> red = {1.0f, 0.0f, 0.0f, 1.0f};

            if (this->leftmarked) {
                leftsubstr = leftmarked->pic->pdbid.substr(0, 4);
            }
            if (this->rightmarked) {
                rightsubstr = rightmarked->pic->pdbid.substr(0, 4);
            }
            
            if (substr.compare(leftsubstr) && substr.compare(rightsubstr)) {
                this->theFont.DrawString(mvp, black.data(), posx, yp, height, false, stringToDraw,
                    core::utility::SDFFont::Alignment::ALIGN_CENTER_MIDDLE);
            } else {
                this->theFont.DrawString(mvp, red.data(), posx, yp, height, false, stringToDraw,
                    core::utility::SDFFont::Alignment::ALIGN_CENTER_MIDDLE);
            }
            yp -= height * 1.05f;
        }

        if (this->addBrendaParam.Param<param::BoolParam>()->Value()) {
            auto pdbid = node->pic->pdbid;
            auto classes = EnzymeClassProvider::RetrieveClassesForPdbId(pdbid, *this->GetCoreInstance());
            std::array<float, 4> color = {0.1f, 0.1f, 0.1f, 1.0f};
            auto curheight = height * 0.5f;
            for (const auto c : classes) {
                auto text = EnzymeClassProvider::ConvertEnzymeClassToString(c);
                auto lineWidth = theFont.LineWidth(curheight, text.c_str());

                this->theFont.DrawString(mvp, color.data(), posx, yp, curheight, false, text.c_str(),
                    core::utility::SDFFont::Alignment::ALIGN_CENTER_MIDDLE);

                yp -= curheight * 1.05f;
            }
        }
    }

    this->passthroughShader.Disable();
    glBindVertexArray(0);

    return posx;
}

/*
 * ClusterRenderer::Render
 */
bool ClusterHierarchieRenderer::Render(core_gl::view::CallRender2DGL& call) {

    this->windowMeasurements = call.GetViewResolution();

    if (this->distanceMatrixParam.IsDirty()) {
        this->distanceMatrixParam.ResetDirty();
        std::filesystem::path dpath = this->distanceMatrixParam.Param<param::FilePathParam>()->Value();
        DistanceMatrixLoader::load(dpath);
    }

    // Update data Clustering
    CallClustering* ccc = this->clusterDataSlot.CallAs<CallClustering>();
    if (!ccc)
        return false;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallClustering::CallForGetData))
        return false;

    // read matrices (old bullshit)
    GLfloat viewMatrixColumn[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrixColumn);
    glm::mat4 view = glm::make_mat4(viewMatrixColumn);
    GLfloat projMatrixColumn[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrixColumn);
    glm::mat4 proj = glm::make_mat4(projMatrixColumn);
    glm::mat4 mvp = proj * view;
    this->zoomFactor = view[0][0];

    // Update data Position
    CallClusterPosition* ccp = this->positionDataSlot.CallAs<CallClusterPosition>();
    if (!ccp) {
        core::utility::log::Log::DefaultLog.WriteError(
            "CallClusterPosition not connected, Module might not work correctly.");
        return false;
    }
    // Updated data from cinematic camera call
    if (!(*ccp)(CallClusterPosition::CallForGetData))
        return false;

    if (ccc->DataHash() != this->lastHashClustering) {
        // update Clustering to work with
        this->clustering = ccc->getClustering();
        this->lastHashClustering = ccc->DataHash();
        this->root = this->clustering->getRoot();
        this->hashoffset++;
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

            GLboolean blend, depth;
            glGetBooleanv(GL_BLEND, &blend);
            glGetBooleanv(GL_DEPTH_TEST, &depth);
            glDisable(GL_BLEND);
            glDisable(GL_DEPTH_TEST);
            drawTree(this->root, mvp, minheight, minwidth, spacey, spacex, colors);
            renderPopup(mvp);
            if (blend == GL_TRUE) {
                glEnable(GL_BLEND);
            }
            if (depth) {
                glEnable(GL_DEPTH_TEST);
            }
        }
    }
    return true;
}

/*
 * ClusterHierarchieRenderer::OnMouseButton
 */
bool ClusterHierarchieRenderer::OnMouseButton(megamol::core::view::MouseButton button,
    megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) {

    if (this->root == nullptr)
        return false;

    auto down = action == MouseButtonAction::PRESS;
    auto shiftmod = mods.test(Modifier::SHIFT);
    this->mouseAction = action;
    this->mouseButton = button;

    float sizefactor = this->sizeMultiplierParam.Param<param::FloatParam>()->Value();

    // Wenn mouse-click auf cluster => change position ...
    // Check position
    if (!shiftmod) {

        this->counter = 0;

        double height = this->viewport.GetY() * 0.9;
        double width = this->viewport.GetX() * 0.9;

        double minheight = this->viewport.GetY() * 0.05;
        double minwidth = this->viewport.GetX() * 0.05;

        double spacey = height / (this->root->level);
        double spacex = width / (this->clustering->getLeaves()->size() - 1);

        double distanceX = 30.0 * sizefactor / (windowMeasurements.x * 2.0 * this->zoomFactor);
        double distanceY = 30.0 * sizefactor / (windowMeasurements.y * 2.0 * this->zoomFactor);

        if (checkposition(this->root, this->mouseX, this->mouseY, minheight, minwidth, spacey, spacex, distanceX,
                distanceY) == -1) {
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

        this->counter = 0;

        double height = this->viewport.GetY() * 0.9;
        double width = this->viewport.GetX() * 0.9;

        double minheight = this->viewport.GetY() * 0.05;
        double minwidth = this->viewport.GetX() * 0.05;

        double spacey = height / (this->root->level);
        double spacex = width / (this->clustering->getLeaves()->size() - 1);

        double distanceX = 30.0 * sizefactor / (windowMeasurements.x * 2.0 * this->zoomFactor);
        double distanceY = 30.0 * sizefactor / (windowMeasurements.y * 2.0 * this->zoomFactor);

        if (checkposition(this->root, this->mouseX, this->mouseY, minheight, minwidth, spacey, spacex, distanceX,
                distanceY) == -1) {
            if (this->popup == nullptr)
                return false;
            auto pdbid = this->popup->pic->pdbid;

            if (action == core::view::MouseButtonAction::PRESS) {
                auto ci = this->GetCoreInstance();
                auto istart = ci->ModuleGraphRoot()->ChildList_Begin();
                auto iend = ci->ModuleGraphRoot()->ChildList_End();
                std::string instname = "";
                for (auto it = istart; it != iend; ++it) {
                    core::AbstractNamedObject::ptr_type ptr = *it;
                    if (ptr != nullptr)
                        instname = ptr->Name();
                }

                auto left =
                    ci->FindParameter((std::string("::") + instname + std::string("::leftpdb::pdbFilename")).c_str());
                auto right =
                    ci->FindParameter((std::string("::") + instname + std::string("::rightpdb::pdbFilename")).c_str());
                vislib::SmartPtr<core::param::AbstractParam> curparam = nullptr;

                if (button == core::view::MouseButton::BUTTON_LEFT) {
                    curparam = left;
                    this->leftmarked = this->popup;
                } else if (button == core::view::MouseButton::BUTTON_RIGHT) {
                    curparam = right;
                    this->rightmarked = this->popup;
                }

                if (!curparam.IsNull()) {
                    std::string pathstring = curparam->ValueString();
                    std::filesystem::path path = pathstring;
                    path.replace_filename(pdbid + ".pdb");
                    bool res = curparam->ParseValue(path.string());
                    if (!res)
                        std::cout << "Could not change parameter" << std::endl;
                }
            }
        } else {
            if (button == core::view::MouseButton::BUTTON_LEFT) {
                this->leftmarked = nullptr;
            } else if (button == core::view::MouseButton::BUTTON_RIGHT) {
                this->rightmarked = nullptr;
            }
        }
    }

    return false;
}

/*
 * ClusterHierarchieRenderer::OnMouseMove
 */
bool ClusterHierarchieRenderer::OnMouseMove(double x, double y) {
    // Only save actual Mouse Position

    this->mouseX = (float) static_cast<int>(x);
    this->mouseY = (float) static_cast<int>(y);

    if (this->clustering == nullptr)
        return false;

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

    float sizefactor = this->sizeMultiplierParam.Param<param::FloatParam>()->Value();
    double distanceX = 30.0 * sizefactor / (windowMeasurements.x * 2.0 * this->zoomFactor);
    double distanceY = 30.0 * sizefactor / (windowMeasurements.y * 2.0 * this->zoomFactor);

    checkposition(this->root, this->mouseX, this->mouseY, minheight, minwidth, spacey, spacex, distanceX, distanceY);

    this->x = this->mouseX;
    this->y = this->mouseY;

    return false;
}

void ClusterHierarchieRenderer::renderPopup(glm::mat4 mvp) {
    if (this->popup != nullptr) {
        // Load texture if not loaded
        glDisable(GL_CULL_FACE);
        glEnable(GL_TEXTURE_2D);
        this->popup->pic->popup = true;
        TextureLoader::loadTexturesToRender(this->clustering);

        // Render Texture
        // Position -> Mouse Position? Cluster Center?
        int width = static_cast<int>(1.0f / this->zoomFactor);
        if (width % 2 == 1)
            width++;
        int height = width / 2;

        // Berechne verschiebung
        int shiftx = 0;
        int shifty = 0;

        if (this->x + width > this->viewport.GetX()) {
            shiftx = this->viewport.GetX() - (this->x + width);
        }

        if (this->y + height > this->viewport.GetY()) {
            shifty = this->viewport.GetY() - (this->y + height);
        }

        this->popup->pic->texture->bindTexture();

        glBindVertexArray(this->texVa);
        this->textureShader.Enable();

        glUniform2f(this->textureShader.ParameterLocation("lowerleft"), this->x + shiftx, this->y + shifty);
        glUniform2f(
            this->textureShader.ParameterLocation("upperright"), this->x + shiftx + width, this->y + shifty + height);
        glUniformMatrix4fv(this->textureShader.ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform1i(this->textureShader.ParameterLocation("tex"), 0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        this->textureShader.Disable();
        glBindVertexArray(0);
        glDisable(GL_TEXTURE_2D);

        auto fontSize = this->fontSizeParam.Param<param::FloatParam>()->Value();
        auto lineHeight = theFont.LineHeight(fontSize);

        // render fonts if necessary
        if (this->showPDBIdsParam.Param<param::BoolParam>()->Value()) {
            auto stringToDraw = this->popup->pic->pdbid.c_str();
            auto lineWidth = theFont.LineWidth(fontSize, stringToDraw);

            std::array<float, 4> color = {0.0f, 0.0f, 0.0f, 1.0f};
            this->theFont.DrawString(mvp, color.data(), this->x + shiftx + width * 0.5f, this->y + shifty + height * 0.5f,
                fontSize, false, stringToDraw, core::utility::SDFFont::Alignment::ALIGN_CENTER_MIDDLE);
        }
        if (this->showEnzymeClassesParam.Param<param::BoolParam>()->Value()) {
            auto classes =
                EnzymeClassProvider::RetrieveClassesForPdbId(this->popup->pic->pdbid, *this->GetCoreInstance());
            auto numClasses = static_cast<uint32_t>(classes.size());
            float margin = 5.0f;

            if (numClasses > 0) {
                std::string text = "";
                uint32_t idx = 0;
                for (const auto& v : classes) {
                    text += EnzymeClassProvider::ConvertEnzymeClassToString(v);
                    if (idx < numClasses - 1) {
                        text += "\n";
                    }
                    ++idx;
                }

                std::array<float, 4> color = {1.0f, 1.0f, 1.0f, 1.0f};
                this->theFont.DrawString(mvp, color.data(), this->x + shiftx + width + margin,
                    this->y + shifty + height * 0.5f, fontSize, false, text.c_str(),
                    core::utility::SDFFont::Alignment::ALIGN_LEFT_MIDDLE);
            }
        }
        glEnable(GL_CULL_FACE);
    }
}

void ClusterHierarchieRenderer::renderMap(glm::mat4 mvp, glm::vec2 lowerleft, glm::vec2 upperright, PictureData* data) {

}

double ClusterHierarchieRenderer::checkposition(HierarchicalClustering::CLUSTERNODE* node, float x, float y,
    double minheight, double minwidth, double spacey, double spacex, double distanceX, double distanceY) {

    double posx = 0;
    double posy = 0;
    double posLeft = 0;
    double posRight = 0;

    double totalheight = 0.9 * this->viewport.GetY();
    double maxheight = this->root->height;

    // draw child node
    if (node->level == 0) {
        posx = minwidth + (counter * spacex);
        // posy = minheight + (node->level * spacey);
        posy = minheight + (node->height / maxheight) * totalheight;
        this->counter++;

        // Check position => if found return -1;
        if (x > posx - distanceX && x < posx + distanceX && y > posy - distanceY && y < posy + distanceY) {
            this->popup = node;
            return -1;
        }

    } else {
        
        posLeft = checkposition(node->left, x, y, minheight, minwidth, spacey, spacex, distanceX, distanceY);
        posRight = checkposition(node->right, x, y, minheight, minwidth, spacey, spacex, distanceX, distanceY);

        if (posLeft == -1 || posRight == -1) {
            return -1;
        } else {
            // Check position
            // Check position => if found return -1;

            posx = (posLeft + posRight) / 2;
            // posy = minheight + (node->level * spacey);
            posy = minheight + (node->height / maxheight) * totalheight;

            if (x > posx - distanceX && x < posx + distanceY && y > posy - distanceX && y < posy + distanceY) {
                this->popup = node;
                return -1;
            }
        }
    }
    return posx;
}


bool ClusterHierarchieRenderer::GetPositionExtents(Call& call) {
    CallClusterPosition* ccp = dynamic_cast<CallClusterPosition*>(&call);
    if (ccp == nullptr) {
        core::utility::log::Log::DefaultLog.WriteError(
            "CallClusterPosition not connected, Module might not work correctly.");
        return false;
    }

    // Wenn neuer root node
    if (ccp->getPosition() != this->position) {
        this->hashoffset++;
        this->newposition = true;
    }
    ccp->SetDataHash(this->DataHashPosition + this->hashoffset);
    return true;
}

bool ClusterHierarchieRenderer::GetPositionData(Call& call) {
    CallClusterPosition* ccp = dynamic_cast<CallClusterPosition*>(&call);
    if (ccp == nullptr) {
        core::utility::log::Log::DefaultLog.WriteError(
            "CallClusterPosition not connected, Module might not work correctly.");
        return false;
    }
    if (this->newposition) {
        this->newposition = false;
        this->DataHashPosition += this->hashoffset;
    }
    ccp->setPosition(this->position);
    ccp->setClusterColors(nullptr);
    return true;
}

float ClusterHierarchieRenderer::enzymeClassDistance(const std::array<int, 4>& arr1, const std::array<int, 4>& arr2) {
    if (arr1[0] < 0 || arr2[0] < 0)
        return 10.0f;
    if (arr1[0] == arr2[0]) {
        if (arr1[1] == arr2[1] || arr1[2] == arr2[2]) {
            if (arr1[1] == arr2[1] && arr1[2] == arr2[2]) {
                if (arr1[3] == arr2[3]) {
                    return 0.0f;
                }
                return 1.0f;
            }
            return 2.0f;
        }
        return 3.0f;
    }
    return 4.0f;
}
