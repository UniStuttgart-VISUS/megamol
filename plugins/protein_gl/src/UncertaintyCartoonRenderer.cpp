/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "UncertaintyCartoonRenderer.h"

#include <inttypes.h>
#include <stdint.h>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/light/DistantLight.h"
#include "mmcore/view/light/PointLight.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmcore_gl/utility/ShaderSourceFactory.h"
#include "mmcore_gl/view/CallGetTransferFunctionGL.h"

#include "protein_calls/ProteinColor.h"
#include "protein_calls/RMSF.h"

#include "vislib/assert.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/mathfunctions.h"

using namespace megamol::core;
using namespace megamol::protein_calls;
using namespace megamol::protein_gl;

const GLuint SSBObindingPoint = 2;

/*
 * UncertaintyCartoonRenderer::UncertaintyCartoonRenderer (CTOR)
 */
UncertaintyCartoonRenderer::UncertaintyCartoonRenderer(void)
        : Renderer3DModuleGL()
        , getPdbDataSlot("getPdbData", "Connects to the pdb data source.")
        , uncertaintyDataSlot(
              "uncertaintyDataSlot", "Connects the cartoon tesselation rendering with uncertainty data storage.")
        , getLightSlot("lightSlot", "Connects the tessellated uncertainty rendering with lights")
        , scalingParam("Scaling", "Scaling factor for particle radii.")
        , backboneParam("Backbone", "Render backbone as tubes.")
        , backboneWidthParam("Backbone width", "The width of the backbone.")
        , tessLevelParam("Tesselation level", "The tesselation level.")
        , lineDebugParam("Wireframe", "Render in wireframe mode.")
        , onlyTubesParam("Only tubes", "Render only tubes.")
        , methodDataParam("Method data", "Choose data of secondary structure assignment method.")
        , uncVisParam("Uncertainty visualisation", "The uncertainty visualisation.")
        , uncDistorGainParam("Distortion: GAIN ", "amplification of function")
        , uncDistorRepParam("Distortion: REPEAT", "repetition of function")
        , ditherParam("Dithering", "enable and add additional dithering passes, dithering is disabled for 0.")
        , outlineParam("Outlining", "The oulining visualisations.")
        , outlineScalingParam("Outline scaling", "The scaling of the ouline.")
        , outlineColorParam("Outline color", "The color of the outline.")
        , uncertainMaterialParam("Uncertain material", "material properties for uncertain structure assignment: "
                                                       "Ambient, diffuse, specular components + exponent")
        , materialParam("Material", "Ambient, diffuse, specular components + exponent.")
        , colorModeParam("Color mode", "Coloring mode for secondary structure.")
        , colorInterpolationParam("Color interpolation", "Should the colors be interpolated?")
        , lightPosParam("Light position", "The light position.")
        , buttonParam("Reload shaders", "Reload the shaders.")
        , colorTableFileParam("Color Table Filename", "The filename of the color table.")
        , bFactorAsUncertaintyParam("BFactor Uncertainty",
              "Use the value stored in the BFactor as uncertainty value. Only useful for preprocessed simulation data.")
        , showRMSFParam("Show RMSF", "Use the computed RMSF and visualize it as uncertainty.")
        , maxRMSFParam("Max. RMSF value", "The maximum RMSF value used for normalization.")
        , useAlphaBlendingParam("Alpha Blending instead of Dithering",
              "Switch from dithering to simple alpha blending of uncertain structure.")
        , fences()
        , currBuf(0)
        , bufSize(32 * 1024 * 1024)
        , numBuffers(3)
        , aminoAcidCount(0)
        , molAtomCount(0)
        , tubeShader_(nullptr)
        , pointLightBuffer_(nullptr)
        , distantLightBuffer_(nullptr)
        // this variant should not need the fence
        , singleBufferCreationBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT)
        , singleBufferMappingBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT)
        , firstframe(true) {

    // number of different secondary structure types
    this->structCount = static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE);

    // uncertainty data caller slot
    this->uncertaintyDataSlot.SetCompatibleCall<UncertaintyDataCallDescription>();
    this->uncertaintyDataSlot.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->uncertaintyDataSlot);

    // pdb data caller slot
    this->getPdbDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->getPdbDataSlot.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->getPdbDataSlot);

    this->getLightSlot.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->getLightSlot.SetNecessity(core::AbstractCallSlotPresentation::Necessity::SLOT_REQUIRED);
    this->MakeSlotAvailable(&this->getLightSlot);

    this->currentTessLevel = 16;
    this->currentUncVis = uncVisualisations::UNC_VIS_SIN_U;
    this->currentColoringMode = coloringModes::COLOR_MODE_STRUCT;
    this->currentScaling = 1.0f;
    this->currentBackboneWidth = 0.2f;
    this->currentMaterial = glm::vec4(0.4f, 0.8f, 0.3f, 10.0f);
    this->currentUncertainMaterial = glm::vec4(0.4f, 0.8f, 0.3f, 10.0f);
    this->currentColoringMode = coloringModes::COLOR_MODE_STRUCT;
    this->currentUncVis = uncVisualisations::UNC_VIS_SIN_UV;
    this->currentLightPos = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
    this->currentUncDist = glm::vec2(1.0f, 5.0f);
    this->currentDitherMode = 0;
    this->currentMethodData = UncertaintyDataCall::assMethod::UNCERTAINTY;
    this->currentOutlineMode = outlineOptions::OUTLINE_NONE;
    this->currentOutlineScaling = 1.0;
    this->currentOutlineColor = glm::vec3(0.0f, 0.0f, 0.0f);

    this->onlyTubesParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->onlyTubesParam);

    this->backboneParam << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->backboneParam);

    this->lineDebugParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->lineDebugParam);

    this->colorInterpolationParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->colorInterpolationParam);

    this->buttonParam << new core::param::ButtonParam();
    this->MakeSlotAvailable(&this->buttonParam);
    // init min max
    this->ditherParam << new core::param::IntParam(this->currentDitherMode, 0, this->structCount);
    this->MakeSlotAvailable(&this->ditherParam);
    // init min max
    this->tessLevelParam << new core::param::IntParam(this->currentTessLevel, 6, 1024);
    this->MakeSlotAvailable(&this->tessLevelParam);

    this->scalingParam << new core::param::FloatParam(this->currentScaling);
    this->MakeSlotAvailable(&this->scalingParam);

    this->outlineScalingParam << new core::param::FloatParam(this->currentOutlineScaling, 1.0f);
    this->MakeSlotAvailable(&this->outlineScalingParam);

    this->backboneWidthParam << new core::param::FloatParam(this->currentBackboneWidth);
    this->MakeSlotAvailable(&this->backboneWidthParam);

    this->materialParam << new core::param::Vector4fParam(
        vislib::math::Vector<float, 4>(glm::value_ptr(this->currentMaterial)));
    this->MakeSlotAvailable(&this->materialParam);

    this->uncertainMaterialParam << new core::param::Vector4fParam(
        vislib::math::Vector<float, 4>(glm::value_ptr(this->currentUncertainMaterial)));
    this->MakeSlotAvailable(&this->uncertainMaterialParam);

    this->lightPosParam << new core::param::Vector4fParam(
        vislib::math::Vector<float, 4>(glm::value_ptr(this->currentLightPos)));
    this->MakeSlotAvailable(&this->lightPosParam);

    this->uncDistorGainParam << new core::param::FloatParam(this->currentUncDist[0]);
    this->MakeSlotAvailable(&this->uncDistorGainParam);

    this->uncDistorRepParam << new core::param::FloatParam(this->currentUncDist[1]);
    this->MakeSlotAvailable(&this->uncDistorRepParam);


    this->outlineColorParam << new core::param::Vector3fParam(
        vislib::math::Vector<float, 3>(glm::value_ptr(this->currentOutlineColor)));
    this->MakeSlotAvailable(&this->outlineColorParam);

    param::EnumParam* tmpEnum = new param::EnumParam(static_cast<int>(this->currentColoringMode));
    tmpEnum->SetTypePair(COLOR_MODE_STRUCT, "Secondary Structure");
    tmpEnum->SetTypePair(COLOR_MODE_UNCERTAIN, "Uncertainty");
    tmpEnum->SetTypePair(COLOR_MODE_CHAIN, "Chains");
    tmpEnum->SetTypePair(COLOR_MODE_AMINOACID, "Aminoacids");
    tmpEnum->SetTypePair(COLOR_MODE_RESIDUE_DEBUG, "DEBUG residues");
    this->colorModeParam << tmpEnum;
    this->MakeSlotAvailable(&this->colorModeParam);

    tmpEnum = new param::EnumParam(static_cast<int>(this->currentUncVis));
    tmpEnum->SetTypePair(UNC_VIS_NONE, "None");
    tmpEnum->SetTypePair(UNC_VIS_SIN_U, "Sinus U");
    tmpEnum->SetTypePair(UNC_VIS_SIN_V, "Sinus V");
    tmpEnum->SetTypePair(UNC_VIS_SIN_UV, "Sinus UV");
    tmpEnum->SetTypePair(UNC_VIS_TRI_U, "Triangle U");
    tmpEnum->SetTypePair(UNC_VIS_TRI_UV, "Triangle UV");
    this->uncVisParam << tmpEnum;
    this->MakeSlotAvailable(&this->uncVisParam);

    tmpEnum = new param::EnumParam(static_cast<int>(this->currentMethodData));
    tmpEnum->SetTypePair(UncertaintyDataCall::assMethod::PDB, "PDB");
    tmpEnum->SetTypePair(UncertaintyDataCall::assMethod::STRIDE, "STRIDE");
    tmpEnum->SetTypePair(UncertaintyDataCall::assMethod::DSSP, "DSSP");
    tmpEnum->SetTypePair(UncertaintyDataCall::assMethod::UNCERTAINTY, "Uncertainty");
    tmpEnum->SetTypePair(UncertaintyDataCall::assMethod::PROSIGN, "PROSIGN");
    this->methodDataParam << tmpEnum;
    this->MakeSlotAvailable(&this->methodDataParam);

    tmpEnum = new param::EnumParam(static_cast<int>(this->currentOutlineMode));
    tmpEnum->SetTypePair(OUTLINE_NONE, "None");
    tmpEnum->SetTypePair(OUTLINE_LINE, "Line rendering");
    tmpEnum->SetTypePair(OUTLINE_FULL_UNCERTAIN, "Full rendering Uncertainty");
    tmpEnum->SetTypePair(OUTLINE_FULL_CERTAIN, "Full rendering Certainty");
    this->outlineParam << tmpEnum;
    this->MakeSlotAvailable(&this->outlineParam);

    // fill color table with default values and set the filename param
    vislib::StringA filename("colors.txt");
    this->colorTableFileParam.SetParameter(new param::FilePathParam(A2T(filename)));
    this->MakeSlotAvailable(&this->colorTableFileParam);
    auto pat = this->colorTableFileParam.Param<param::FilePathParam>()->Value();
    ProteinColor::ReadColorTableFromFile(pat, this->colorTable);

    this->bFactorAsUncertaintyParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->bFactorAsUncertaintyParam);

    this->showRMSFParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->showRMSFParam);

    this->useAlphaBlendingParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->useAlphaBlendingParam);

    this->maxRMSFParam << new core::param::FloatParam(10.0f, 0.0f);
    this->MakeSlotAvailable(&this->maxRMSFParam);

    this->fences.resize(this->numBuffers);
}

/*
 * UncertaintyCartoonRenderer::~UncertaintyCartoonRenderer (DTOR)
 */
UncertaintyCartoonRenderer::~UncertaintyCartoonRenderer(void) {
    this->Release(); // DON'T change !
}

/*
 * UncertaintyCartoonRenderer::loadTubeShader
 */
bool UncertaintyCartoonRenderer::loadTubeShader(void) {
    try {
        auto const shdr_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

        tubeShader_ = core::utility::make_shared_glowl_shader("cartoon", shdr_options,
            std::filesystem::path("protein_gl/uncertaintycartoon/uncertain.vert.glsl"),
            std::filesystem::path("protein_gl/uncertaintycartoon/uncertain.tesc.glsl"),
            std::filesystem::path("protein_gl/uncertaintycartoon/uncertain.tese.glsl"),
            std::filesystem::path("protein_gl/uncertaintycartoon/uncertain.geom.glsl"),
            std::filesystem::path("protein_gl/uncertaintycartoon/uncertain.frag.glsl"));

    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError( "[UncertaintyCartoonRenderer] %s", ex.what());
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[UncertaintyCartoonRenderer] Unable to compile shader: Unknown exception: %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[UncertaintyCartoonRenderer] Unable to compile shader: Unknown exception.");
    }

    return true;
}

/*
 * UncertaintyCartoonRenderer::create
 */
bool UncertaintyCartoonRenderer::create(void) {
    using namespace vislib::sys;
    using namespace vislib_gl::graphics::gl;

    // load tube shader
    if (!this->loadTubeShader()) {
        return false;
    }

    glGenVertexArrays(1, &this->vertArray);
    glBindVertexArray(this->vertArray);
    glGenBuffers(1, &this->theSingleBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSingleBuffer);
    glBufferStorage(
        GL_SHADER_STORAGE_BUFFER, this->bufSize * this->numBuffers, nullptr, this->singleBufferCreationBits);
    this->theSingleMappedMem = glMapNamedBufferRangeEXT(
        this->theSingleBuffer, 0, this->bufSize * this->numBuffers, this->singleBufferMappingBits);
    //    GLAPI void *APIENTRY glMapNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield
    //    access);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindVertexArray(0);

    pointLightBuffer_ = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    distantLightBuffer_ = std::make_unique<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    return true;
}

/*
 * UncertaintyCartoonRenderer::release
 */
void UncertaintyCartoonRenderer::release(void) {

    glUnmapNamedBufferEXT(this->theSingleBuffer);
    for (auto& x : this->fences) {
        if (x) {
            glDeleteSync(x);
        }
    }

    glDeleteVertexArrays(1, &this->vertArray);
    glDeleteBuffers(1, &this->theSingleBuffer);
}


/*
 * UncertaintyCartoonRenderer::GetExtents
 */
bool UncertaintyCartoonRenderer::GetExtents(core_gl::view::CallRender3DGL& call) {

    // get pointer to UncertaintyDataCall
    UncertaintyDataCall* udc = this->uncertaintyDataSlot.CallAs<UncertaintyDataCall>();
    if (udc == nullptr)
        return false;
    // execute the call
    if (!(*udc)(UncertaintyDataCall::CallForGetData))
        return false;

    // get pointer to MolecularDataCall
    MolecularDataCall* mol = this->getPdbDataSlot.CallAs<MolecularDataCall>();
    if ((mol != nullptr) && ((*mol)(MolecularDataCall::CallForGetExtent))) {
        call.SetTimeFramesCount(mol->FrameCount());
        call.AccessBoundingBoxes() = mol->AccessBoundingBoxes();
    } else {
        call.SetTimeFramesCount(1);
        call.AccessBoundingBoxes().Clear();
    }

    return true;
}

/*
 *  UncertaintyCartoonRenderer::GetData
 */
MolecularDataCall* UncertaintyCartoonRenderer::GetData(unsigned int t) {

    MolecularDataCall* mol = this->getPdbDataSlot.CallAs<MolecularDataCall>();

    if (mol != nullptr) {
        mol->SetFrameID(t);
        if (!(*mol)(MolecularDataCall::CallForGetExtent))
            return nullptr;

        mol->SetFrameID(t);
        if (!(*mol)(MolecularDataCall::CallForGetData))
            return nullptr;

        return mol;
    } else {
        return nullptr;
    }
}

/*
 * UncertaintyCartoonRenderer::GetUncertaintyData
 */
bool UncertaintyCartoonRenderer::GetUncertaintyData(UncertaintyDataCall* udc, MolecularDataCall* mol) {

    if (!udc)
        return false;
    if (!mol)
        return false;

    // execute the call
    if (!(*udc)(UncertaintyDataCall::CallForGetData))
        return false;

    // check molecular data
    // if (!(*mol)(MolecularDataCall::CallForGetData)) return false; // don't call twice ... ?

    // initialization
    this->aminoAcidCount = udc->GetAminoAcidCount();
    this->molAtomCount = mol->AtomCount();

    // reset arrays
    this->secStructColor.Clear();
    this->secStructColor.AssertCapacity(UncertaintyDataCall::secStructure::NOE);

    // reset arrays
    for (unsigned int i = 0; i < this->sortedSecStructAssignment.Count(); i++) {
        this->sortedSecStructAssignment.Clear();
    }
    this->sortedSecStructAssignment.Clear();
    for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); i++) {
        this->sortedSecStructAssignment.Add(vislib::Array<vislib::math::Vector<UncertaintyDataCall::secStructure,
                static_cast<int>(UncertaintyDataCall::secStructure::NOE)>>());
        this->sortedSecStructAssignment.Last().AssertCapacity(this->aminoAcidCount);
    }

    for (unsigned int i = 0; i < this->secStructUncertainty.Count(); i++) {
        this->secStructUncertainty.Clear();
    }
    this->secStructUncertainty.Clear();
    for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); i++) {
        this->secStructUncertainty.Add(
            vislib::Array<vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)>>());
        this->secStructUncertainty.Last().AssertCapacity(this->aminoAcidCount);
    }

    this->residueFlag.Clear();
    this->residueFlag.AssertCapacity(this->aminoAcidCount);

    this->uncertainty.Clear();
    this->uncertainty.AssertCapacity(this->aminoAcidCount);

    this->pdbIndex.Clear();
    this->pdbIndex.AssertCapacity(this->aminoAcidCount);

    this->chainColors.clear();
    this->chainColors.resize(this->aminoAcidCount);

    this->aminoAcidColors.clear();
    this->aminoAcidColors.resize(this->aminoAcidCount);

    // get secondary structure type colors
    for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); i++) {
        this->secStructColor.Add(udc->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(i)));
    }

    unsigned int cCnt = 0;
    char currentChainID = udc->GetChainID(0);

    // collect data from call
    for (unsigned int aa = 0; aa < this->aminoAcidCount; aa++) {

        // store the secondary structure element type of the current amino-acid for each assignment method
        for (unsigned int k = 0; k < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); k++) {
            this->sortedSecStructAssignment[k].Add(
                udc->GetSortedSecStructAssignment(static_cast<UncertaintyDataCall::assMethod>(k), aa));
            this->secStructUncertainty[k].Add(
                udc->GetSecStructUncertainty(static_cast<UncertaintyDataCall::assMethod>(k), aa));
        }

        // store residue flag
        this->residueFlag.Add(static_cast<unsigned int>(udc->GetResidueFlag(aa)));
        // store the uncertainty difference
        this->uncertainty.Add(udc->GetUncertainty(aa));
        // store the original pdb index
        this->pdbIndex.Add(udc->GetPDBAminoAcidIndex(aa));


        // count different chains and set chain color
        if (udc->GetChainID(aa) != currentChainID) {
            currentChainID = udc->GetChainID(aa);
            cCnt++;
        }
        // number of different chains: [A-Z] + [a-z] = 52
        this->chainColors.push_back(this->colorTable[(cCnt % this->colorTable.size())]);

        // set colors for amino-acids [A-Z] +'?' = 27
        unsigned int tmpAA = static_cast<unsigned int>(udc->GetAminoAcidOneLetterCode(aa));
        this->aminoAcidColors.push_back(this->colorTable[(tmpAA % this->colorTable.size())]);
    }


    // Synchronize data array index from MoleculeDataCall with data array index from UncertaintyDataCall via the
    // original pdb index
    unsigned int firstMol;
    unsigned int firstStruct;
    unsigned int uncIndex;
    unsigned int molIndex;
    unsigned int origMolIndex;

    this->synchronizedIndex.Clear();
    this->synchronizedIndex.AssertCapacity(this->molAtomCount);

    // loop over all chains of the molecular data
    for (unsigned int cCnt = 0; cCnt < mol->ChainCount(); cCnt++) { // all chains

        firstMol = mol->Chains()[cCnt].FirstMoleculeIndex();
        for (unsigned int mCnt = firstMol; mCnt < firstMol + mol->Chains()[cCnt].MoleculeCount();
             mCnt++) { // molecules in chain (?)

            firstStruct = mol->Molecules()[mCnt].FirstSecStructIndex();
            for (unsigned int sCnt = 0; sCnt < mol->Molecules()[mCnt].SecStructCount();
                 sCnt++) { // secondary structures in chain

                for (unsigned int rCnt = 0; rCnt < mol->SecondaryStructures()[firstStruct + sCnt].AminoAcidCount();
                     rCnt++) { // aminoacids in secondary structures

                    uncIndex = 0;
                    origMolIndex =
                        mol->Residues()[(mol->SecondaryStructures()[firstStruct + sCnt].FirstAminoAcidIndex() + rCnt)]
                            ->OriginalResIndex();

                    // go to right chain in uncertainty data
                    while ((uncIndex < this->aminoAcidCount) &&
                           (mol->Chains()[cCnt].Name() != udc->GetChainID(uncIndex))) {
                        uncIndex++;
                    }
                    // search for matching original pdb indices in both data loaders
                    while (uncIndex < this->aminoAcidCount) {
                        if (static_cast<std::string>(udc->GetPDBAminoAcidIndex(uncIndex).PeekBuffer())
                                .find_first_not_of("0123456789") == std::string::npos) { // C++11 function ...
                            if (std::atoi(udc->GetPDBAminoAcidIndex(uncIndex)) == origMolIndex) {
                                break;
                            }
                        }
                        uncIndex++;
                    }

                    // when indices in molecular data are missing, fill 'synchronizedIndex' with 'dummy' indices.
                    // in the end maximum molecular index must always match length of synchronizedIndex array!
                    molIndex = mol->SecondaryStructures()[firstStruct + sCnt].FirstAminoAcidIndex() + rCnt;
                    while (this->synchronizedIndex.Count() < molIndex) {
                        this->synchronizedIndex.Add(0);
                    }
                    this->synchronizedIndex.Add(uncIndex);

                    // DEBUG
                    /*
                                        unsigned int molIndex = (mol->SecondaryStructures()[firstStruct +
                       sCnt].FirstAminoAcidIndex() + rCnt); std::cout << " mol index: " <<
                       mol->SecondaryStructures()[firstStruct + sCnt].FirstAminoAcidIndex() + rCnt
                                                  << " | Chain Name: " << mol->Chains()[cCnt].Name()
                                                          << " | mol orig index: " << origMolIndex
                                                          << " | unc orig index: " << this->synchronizedIndex.Last()
                                                          << " | count: " << this->synchronizedIndex.Count()
                                                          << std::endl;
                                        */
                }
            }
        }
    }
    return true;
}

/*
 * UncertaintyCartoonRenderer::Render
 */
bool UncertaintyCartoonRenderer::Render(core_gl::view::CallRender3DGL& call) {

    // get new data from the MolecularDataCall
    MolecularDataCall* mol = this->GetData(static_cast<unsigned int>(call.Time()));
    if (mol == nullptr)
        return false;

    if (this->showRMSFParam.Param<megamol::core::param::BoolParam>()->Value()) {
        firstframe = protein_calls::computeRMSF(mol);
        if (firstframe) {
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "Successfully computed RMSF (min: %.3f, max: %.3f).", mol->MinimumBFactor(), mol->MaximumBFactor());
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("Could not compute RMSF.");
        }
    }

    // get pointer to UncertaintyDataCall
    UncertaintyDataCall* ud = this->uncertaintyDataSlot.CallAs<UncertaintyDataCall>();
    if (ud == nullptr)
        return false;

    // if amino-acid count changed get new data
    if ((ud->GetAminoAcidCount() != this->aminoAcidCount) && (mol->AtomCount() != this->molAtomCount)) {
        this->GetUncertaintyData(ud, mol); // use return value ...?
    }

    // update lights
    auto lc = this->getLightSlot.CallAs<core::view::light::CallLight>();
    this->RefreshLights(lc, call.GetCamera().getPose().direction);

    // get method data choice
    if (this->methodDataParam.IsDirty()) {
        this->methodDataParam.ResetDirty();
        this->currentMethodData =
            static_cast<UncertaintyDataCall::assMethod>(this->methodDataParam.Param<param::EnumParam>()->Value());
    }
    // get dither mode
    if (this->ditherParam.IsDirty()) {
        this->ditherParam.ResetDirty();
        this->currentDitherMode = this->ditherParam.Param<param::IntParam>()->Value();
    }
    // get scaling factor
    if (this->scalingParam.IsDirty()) {
        this->scalingParam.ResetDirty();
        this->currentScaling = static_cast<float>(this->scalingParam.Param<param::FloatParam>()->Value());
    }
    // get backbone width
    if (this->backboneWidthParam.IsDirty()) {
        this->backboneWidthParam.ResetDirty();
        this->currentBackboneWidth = static_cast<float>(this->backboneWidthParam.Param<param::FloatParam>()->Value());
    }
    // get material lighting properties
    if (this->materialParam.IsDirty()) {
        this->materialParam.ResetDirty();
        this->currentMaterial =
            glm::make_vec4(this->materialParam.Param<param::Vector4fParam>()->Value().PeekComponents());
    }
    // get material lighting properties
    if (this->uncertainMaterialParam.IsDirty()) {
        this->uncertainMaterialParam.ResetDirty();
        this->currentUncertainMaterial =
            glm::make_vec4(this->uncertainMaterialParam.Param<param::Vector4fParam>()->Value().PeekComponents());
    }

    // get uncertainty distortion: gain
    if (this->uncDistorGainParam.IsDirty()) {
        this->uncDistorGainParam.ResetDirty();
        this->currentUncDist[0] = this->uncDistorGainParam.Param<param::FloatParam>()->Value();
    }
    // get uncertainty distortion: repeat
    if (this->uncDistorRepParam.IsDirty()) {
        this->uncDistorRepParam.ResetDirty();
        this->currentUncDist[1] = this->uncDistorRepParam.Param<param::FloatParam>()->Value();
    }

    // get uncertainty visualisation mode
    if (this->uncVisParam.IsDirty()) {
        this->uncVisParam.ResetDirty();
        this->currentUncVis = static_cast<uncVisualisations>(this->uncVisParam.Param<param::EnumParam>()->Value());
    }
    // get outlining visualisation mode
    if (this->outlineParam.IsDirty()) {
        this->outlineParam.ResetDirty();
        this->currentOutlineMode = static_cast<outlineOptions>(this->outlineParam.Param<param::EnumParam>()->Value());
    }
    // get scaling of the outline
    if (this->outlineScalingParam.IsDirty()) {
        this->outlineScalingParam.ResetDirty();
        this->currentOutlineScaling = static_cast<float>(this->outlineScalingParam.Param<param::FloatParam>()->Value());
    }
    // get color of the outline
    if (this->outlineColorParam.IsDirty()) {
        this->outlineColorParam.ResetDirty();
        this->currentOutlineColor =
            glm::make_vec3(this->outlineColorParam.Param<param::Vector3fParam>()->Value().PeekComponents());
    }
    // read and update the color table, if necessary
    if (this->colorTableFileParam.IsDirty()) {
        ProteinColor::ReadColorTableFromFile(
            this->colorTableFileParam.Param<param::FilePathParam>()->Value(), this->colorTable);
        this->colorTableFileParam.ResetDirty();
    }
    // get lighting position
    if (this->lightPosParam.IsDirty()) {
        this->lightPosParam.ResetDirty();
        this->currentLightPos =
            glm::make_vec4(this->lightPosParam.Param<param::Vector4fParam>()->Value().PeekComponents());
    }
    // get coloring mode
    if (this->colorModeParam.IsDirty()) {
        this->colorModeParam.ResetDirty();
        this->currentColoringMode = static_cast<coloringModes>(this->colorModeParam.Param<param::EnumParam>()->Value());
    }
    // get new tesselation level
    if (this->tessLevelParam.IsDirty()) {
        this->tessLevelParam.ResetDirty();
        this->currentTessLevel = this->tessLevelParam.Param<param::IntParam>()->Value();
    }
    // reload shaders
    if (this->buttonParam.IsDirty()) {
        this->buttonParam.ResetDirty();
        if (!this->loadTubeShader()) {
            return false;
        }
    }

    // timer.BeginFrame();

    glm::vec4 clipDat(0.0f, 0.0f, 0.0f, 0.0f);
    glm::vec4 clipCol(0.75f, 0.75f, 0.75f, 1.0f);

    core::view::Camera cam = call.GetCamera();
    glm::mat4 view = cam.getViewMatrix();
    glm::mat4 proj = cam.getProjectionMatrix();
    glm::mat4 mvp = proj * view;
    glm::mat4 invview = glm::inverse(view);
    glm::mat4 invtransview = glm::transpose(invview);
    glm::mat4 invproj = glm::inverse(proj);
    glm::mat4 mvpinv = glm::inverse(mvp);
    glm::mat4 mvptrans = glm::transpose(mvp);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_CULL_FACE);

    glm::vec4 viewportStuff;
    ::glGetFloatv(GL_VIEWPORT, glm::value_ptr(viewportStuff));
    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));
    if (viewportStuff[2] < 1.0f)
        viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f)
        viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSingleBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer);

    // DEBUG
    /*std::cout << lightAmbient[0] << " " << lightAmbient[1] << " " << lightAmbient[2] << " " << lightAmbient[3] <<
    std::endl; std::cout << lightDiffuse[0] << " " << lightDiffuse[1] << " " << lightDiffuse[2] << " " <<
    lightDiffuse[3] << std::endl;
    std::cout << lightSpecular[0] << " " << lightSpecular[1] << " " << lightSpecular[2] << " " << lightSpecular[3] <<
    std::endl;*/


    // Render in wireframe mode
    if (this->lineDebugParam.Param<param::BoolParam>()->Value())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    unsigned int firstResIdx = 0;
    unsigned int lastResIdx = 0;
    unsigned int firstAtomIdx = 0;
    unsigned int lastAtomIdx = 0;
    unsigned int atomTypeIdx = 0;
    unsigned int firstSecIdx = 0;
    unsigned int lastSecIdx = 0;
    unsigned int firstAAIdx = 0;
    unsigned int lastAAIdx = 0;

    unsigned int uncIndex = 0;

    // Render backbone as tubes
    if (this->backboneParam.Param<param::BoolParam>()->Value()) {

        // copy data to: mainChain
        firstSecIdx = 0;
        lastSecIdx = 0;
        firstAAIdx = 0;
        lastAAIdx = 0;

        this->mainChain.clear();

        CAlpha lastCalpha;

        int molCount = mol->MoleculeCount();
        std::vector<int> molSizes;

        // loop over all molecules of the protein
        for (unsigned int molIdx = 0; molIdx < mol->MoleculeCount(); molIdx++) {

            MolecularDataCall::Molecule chain = mol->Molecules()[molIdx];

            molSizes.push_back(0);

            bool firstset = false;

            // is the first residue an aminoacid?
            // if first residue is no aminoacid the whole secondary structure is skipped!
            if (mol->Residues()[chain.FirstResidueIndex()]->Identifier() != MolecularDataCall::Residue::AMINOACID) {
                continue;
            }

            firstSecIdx = chain.FirstSecStructIndex();
            lastSecIdx = firstSecIdx + chain.SecStructCount();

            // loop over all secondary structures of the molecule
            for (unsigned int secIdx = firstSecIdx; secIdx < lastSecIdx; secIdx++) {
                firstAAIdx = mol->SecondaryStructures()[secIdx].FirstAminoAcidIndex();
                lastAAIdx = firstAAIdx + mol->SecondaryStructures()[secIdx].AminoAcidCount();

                // loop over all aminoacids inside the secondary structure
                for (unsigned int aaIdx = firstAAIdx; aaIdx < lastAAIdx; aaIdx++) {

                    MolecularDataCall::AminoAcid* acid;

                    // is the current residue really an aminoacid?
                    if (mol->Residues()[aaIdx]->Identifier() == MolecularDataCall::Residue::AMINOACID)
                        acid = (MolecularDataCall::AminoAcid*)(mol->Residues()[aaIdx]);
                    else
                        continue;

                    // extract relevant positions and other values
                    CAlpha calpha;

                    // DEBUG
                    /*
                    std::cout << " mol index: " << aaIdx
                                      << " | unc index: " << this->synchronizedIndex[aaIdx]
                                      << std::endl;
                    */

                    uncIndex = this->synchronizedIndex[aaIdx];
                    for (unsigned int k = 0; k < this->structCount; k++) {
                        calpha.sortedStruct[k] = static_cast<int>(
                            this->sortedSecStructAssignment[(int)this->currentMethodData][uncIndex][k]);
                        calpha.unc[k] = this->secStructUncertainty[(int)this->currentMethodData][uncIndex][k];
                        if (this->bFactorAsUncertaintyParam.Param<param::BoolParam>()->Value()) {
                            // calpha.unc[k] = (mol->AtomBFactors()[acid->CAlphaIndex()] -
                            // mol->MinimumBFactor())/(mol->MaximumBFactor() - mol->MinimumBFactor());
                            if (this->showRMSFParam.Param<param::BoolParam>()->Value()) {
                                calpha.unc[k] = mol->AtomBFactors()[acid->CAlphaIndex()] /
                                                (std::fmaxf(mol->MaximumBFactor(),
                                                    this->maxRMSFParam.Param<param::FloatParam>()->Value()));
                            } else {
                                calpha.unc[k] = mol->AtomBFactors()[acid->CAlphaIndex()];
                            }
                        }
                    }
                    if (this->currentColoringMode == (int)COLOR_MODE_CHAIN) {
                        for (unsigned int k = 0; k < 3; k++)
                            calpha.col[k] = this->chainColors[uncIndex][k];
                    } else if (this->currentColoringMode == (int)COLOR_MODE_AMINOACID) {
                        for (unsigned int k = 0; k < 3; k++)
                            calpha.col[k] = this->aminoAcidColors[uncIndex][k];
                    }
                    calpha.flag = this->residueFlag[uncIndex];
                    calpha.uncertainty = this->uncertainty[uncIndex];

                    if (this->bFactorAsUncertaintyParam.Param<param::BoolParam>()->Value()) {
                        // calpha.uncertainty = (mol->AtomBFactors()[acid->CAlphaIndex()] - mol->MinimumBFactor()) /
                        // (mol->MaximumBFactor() - mol->MinimumBFactor());
                        if (this->showRMSFParam.Param<param::BoolParam>()->Value()) {
                            calpha.uncertainty = mol->AtomBFactors()[acid->CAlphaIndex()] /
                                                 (std::fmaxf(mol->MaximumBFactor(),
                                                     this->maxRMSFParam.Param<param::FloatParam>()->Value()));
                        } else {
                            calpha.uncertainty = mol->AtomBFactors()[acid->CAlphaIndex()];
                        }
                    }

                    calpha.pos[0] = mol->AtomPositions()[3 * acid->CAlphaIndex()];
                    calpha.pos[1] = mol->AtomPositions()[3 * acid->CAlphaIndex() + 1];
                    calpha.pos[2] = mol->AtomPositions()[3 * acid->CAlphaIndex() + 2];
                    calpha.pos[3] = 1.0f;

                    // direction is vector from C_alpha atom to O(xygen) atom
                    calpha.dir[0] = mol->AtomPositions()[3 * acid->OIndex()] - calpha.pos[0];
                    calpha.dir[1] = mol->AtomPositions()[3 * acid->OIndex() + 1] - calpha.pos[1];
                    calpha.dir[2] = mol->AtomPositions()[3 * acid->OIndex() + 2] - calpha.pos[2];

                    // TODO: do this on GPU?
                    // orientation check for the direction
                    if (this->mainChain.size() != 0) {
                        CAlpha before = this->mainChain[this->mainChain.size() - 1];
                        float dotProd = calpha.dir[0] * before.dir[0] + calpha.dir[1] * before.dir[1] +
                                        calpha.dir[2] * before.dir[2];

                        if (dotProd < 0) // flip direction if the orientation is wrong
                        {
                            calpha.dir[0] = -calpha.dir[0];
                            calpha.dir[1] = -calpha.dir[1];
                            calpha.dir[2] = -calpha.dir[2];
                        }
                    }

                    this->mainChain.push_back(calpha);
                    molSizes[molIdx]++;

                    lastCalpha = calpha;

                    // add the first atom 3 times - for every different secondary structure
                    if (!firstset) {
                        this->mainChain.push_back(calpha);
                        this->mainChain.push_back(calpha);
                        molSizes[molIdx] += 2;
                        firstset = true;
                    }
                }
            }

            // add the last atom 3 times
            this->mainChain.push_back(lastCalpha);
            this->mainChain.push_back(lastCalpha);
            molSizes[molIdx] += 2;
        }

        // draw backbone as tubes
        unsigned int colBytes, vertBytes, colStride, vertStride;
        this->GetBytesAndStride(*mol, colBytes, vertBytes, colStride, vertStride);
        // this->currBuf = 0;
        tubeShader_->use();
        pointLightBuffer_->bind(4);
        distantLightBuffer_->bind(5);

        glColor4f(1.0f / this->mainChain.size(), 0.75f, 0.25f, 1.0f);

        tubeShader_->setUniform("viewAttr", viewportStuff);
        tubeShader_->setUniform("scaling", this->currentScaling);
        tubeShader_->setUniform("camPos", call.GetCamera().getPose().position);
        tubeShader_->setUniform("camIn", call.GetCamera().getPose().direction);
        tubeShader_->setUniform("camUp", call.GetCamera().getPose().up);
        tubeShader_->setUniform("camRight", call.GetCamera().getPose().right);
        tubeShader_->setUniform("clipDat", clipDat);
        tubeShader_->setUniform("clipCol", clipCol);
        tubeShader_->setUniform("MV", view);
        tubeShader_->setUniform("MVinv", invview);
        tubeShader_->setUniform("MVinvtrans", invtransview);
        tubeShader_->setUniform("MVP", mvp);
        tubeShader_->setUniform("MVPinv", mvpinv);
        tubeShader_->setUniform("MVPtransp", mvptrans);
        tubeShader_->setUniform("inConsts1", -1.0f, 0.0f, 0.0f, 0.0f);
        tubeShader_->setUniform("tessLevel", this->currentTessLevel);
        tubeShader_->setUniform("pipeWidth", this->currentBackboneWidth);
        tubeShader_->setUniform("interpolateColors", this->colorInterpolationParam.Param<param::BoolParam>()->Value());
        glUniform4fv(tubeShader_->getUniformLocation("structCol"), this->structCount,
            (GLfloat*)this->secStructColor.PeekElements());
        tubeShader_->setUniform("colorMode", this->currentColoringMode);
        tubeShader_->setUniform("onlyTubes", this->onlyTubesParam.Param<param::BoolParam>()->Value());
        tubeShader_->setUniform("uncVisMode", static_cast<int>(this->currentUncVis));
        tubeShader_->setUniform("uncDistor", this->currentUncDist);
        tubeShader_->setUniform("ProjInv", invproj);
        tubeShader_->setUniform("ambientColor", 1.0f, 1.0f, 1.0f, 1.0f);
        tubeShader_->setUniform("diffuseColor", 1.0f, 1.0f, 1.0f, 1.0f);
        tubeShader_->setUniform("specularColor", 1.0f, 1.0f, 1.0f, 1.0f);
        tubeShader_->setUniform("phong", currentMaterial);
        tubeShader_->setUniform("phongUncertain", this->currentUncertainMaterial);
        tubeShader_->setUniform("outlineScale", this->currentOutlineScaling);
        tubeShader_->setUniform("outlineColor", this->currentOutlineColor);
        tubeShader_->setUniform("point_light_cnt", static_cast<GLint>(pointLights_.size()));
        tubeShader_->setUniform("distant_light_cnt", static_cast<GLint>(distantLights_.size()));

        // Switch from dithering to alpha blending
        if (this->useAlphaBlendingParam.Param<megamol::core::param::BoolParam>()->Value()) {
            glEnable(GL_BLEND);
            glDisable(GL_DEPTH_TEST);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            tubeShader_->setUniform("alphaBlending", 1);
        } else {
            tubeShader_->setUniform("alphaBlending", 0);
        }

        // outlining
        int outlinePass = 0;
        if (this->currentOutlineMode != OUTLINE_NONE) {
            outlinePass = 1;
        }
        // dithering
        int ditherPass = 0;
        if (this->currentDitherMode > 0) {
            ditherPass = this->currentDitherMode - 1;
        }
        // geometry draw loop
        for (int pass = 0; pass <= (ditherPass + outlinePass); pass++) {

            // default values for dithering and outlining passes
            tubeShader_->setUniform("outlineMode", 0);
            tubeShader_->setUniform("ditherCount", 0);

            // if dithering is enabled increment dither pass count to enable dithering in shader
            if ((this->currentDitherMode > 0) && (pass <= ditherPass)) {
                tubeShader_->setUniform("ditherCount", pass + 1);
            }

            // if outlining is enabled wait for last pass to draw outline
            if ((this->currentOutlineMode != OUTLINE_NONE) && (pass == (ditherPass + outlinePass))) {
                // draw back faces
                glCullFace(GL_FRONT);
                if (this->currentOutlineMode == OUTLINE_LINE) {
                    glPolygonMode(GL_BACK, GL_LINE);
                    glEnable(GL_LINE_SMOOTH);
                    glLineWidth((float)this->currentOutlineScaling);
                } else {
                    glPolygonMode(GL_BACK, GL_FILL);
                }
                tubeShader_->setUniform("outlineMode", static_cast<int>(this->currentOutlineMode));
                // if dithering is enabled draw outline for "biggest" structure
                if (this->currentDitherMode > 0) {
                    tubeShader_->setUniform("ditherCount", ditherPass + 1);
                }
            }

            // drawing GEOMETRY
            UINT64 numVerts;
            numVerts =
                this->bufSize / vertStride; // bufSize = 32*1024*1024 - WHY? | vertStride = (unsigned int)sizeof(CAlpha)
            // numVert = number of vertices fitting into bufSize
            UINT64 stride = 0; // aminoacid index in mainChain

            for (int i = 0; i < (int)molSizes.size(); i++) { // loop over all secondary structures
                UINT64 vertCounter = 0;
                while (
                    vertCounter < molSizes[i]) { // loop over all aminoacids inside of one secondary structure - WHY ?

                    const char* currVert =
                        (const char*)(&this->mainChain[(unsigned int)vertCounter +
                                                       (unsigned int)stride]); // pointer to current vertex data in
                                                                               // mainChain

                    void* mem = static_cast<char*>(this->theSingleMappedMem) +
                                this->bufSize * this->currBuf; // pointer to the mapped memory - ?
                    const char* whence = currVert;             // copy of pointer currVert
                    UINT64 vertsThisTime = vislib::math::Min(molSizes[i] - vertCounter,
                        numVerts); // try to take all vertices of current secondary structure at once ...
                    // ... or at least as many as fit into buffer of size bufSize
                    this->WaitSignal(this->fences[this->currBuf]); // wait for buffer 'currBuf' to be "ready" - ?

                    memcpy(mem, whence,
                        (size_t)vertsThisTime *
                            vertStride); // copy data of current vertex data in mainChain to mapped memory - ?

                    glFlushMappedNamedBufferRangeEXT(theSingleBuffer, this->bufSize * this->currBuf,
                        (GLsizeiptr)vertsThisTime * vertStride); // parameter: buffer, offset, length

                    tubeShader_->setUniform("instanceOffset", 0);

                    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer,
                        this->bufSize * this->currBuf, this->bufSize); // bind Shader Storage Buffer Object
                    glPatchParameteri(
                        GL_PATCH_VERTICES, 1); // set parameter GL_PATCH_VERTICES to 1 (the number of vertices that will
                                               // be used to make up a single patch primitive)
                    glDrawArrays(
                        GL_PATCHES, 0, (GLsizei)(vertsThisTime - 3)); // draw as many as (vertsThisTime-3) patches
                    // -3 ? - because the first atom is added 3 times for each different secondary structure ??
                    this->QueueSignal(this->fences[this->currBuf]); // queue signal - tell that mapped memory
                                                                    // 'operations' are done - ?

                    this->currBuf = (this->currBuf + 1) % this->numBuffers; // switch to next buffer in range 0...3
                    vertCounter += vertsThisTime;                           // increase counter of processed vertices
                    currVert += vertsThisTime * vertStride; // unused - will be overwritten in next loop cycle
                }
                stride += molSizes[i];
            }
        }

        // glDisable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisable(GL_TEXTURE_1D);
        glUseProgram(0);

        // Switch from dithering to alpha blending
        if (this->useAlphaBlendingParam.Param<megamol::core::param::BoolParam>()->Value()) {
            glDisable(GL_BLEND);
            glEnable(GL_DEPTH_TEST);
        }
    }

    mol->Unlock();

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    // reset stuff
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    //  timer.EndFrame();

    return true;
}

/*
 * UncertaintyCartoonRenderer::GetBytesAndStride
 */
void UncertaintyCartoonRenderer::GetBytesAndStride(MolecularDataCall& mol, unsigned int& colBytes,
    unsigned int& vertBytes, unsigned int& colStride, unsigned int& vertStride) {

    vertBytes = 0;
    colBytes = 0;
    // colBytes = vislib::math::Max(colBytes, 3 * 4U);
    vertBytes = vislib::math::Max(vertBytes, (unsigned int)sizeof(CAlpha));

    colStride = 0;
    colStride = (colStride < colBytes) ? (colBytes) : (colStride);
    vertStride = 0;
    vertStride = (vertStride < vertBytes) ? (vertBytes) : (vertStride);
}

/*
 * UncertaintyCartoonRenderer::QueueSignal
 */
void UncertaintyCartoonRenderer::QueueSignal(GLsync& syncObj) {

    if (syncObj) {
        glDeleteSync(syncObj);
    }
    syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

/*
 * UncertaintyCartoonRenderer::WaitSignal
 */
void UncertaintyCartoonRenderer::WaitSignal(GLsync& syncObj) {

    if (syncObj) {
        while (1) {
            GLenum wait = glClientWaitSync(syncObj, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                return;
            }
        }
    }
}

void UncertaintyCartoonRenderer::RefreshLights(core::view::light::CallLight* lightCall, glm::vec3 camDir) {

    if (lightCall == nullptr || !(*lightCall)(core::view::light::CallLight::CallGetData)) {
        pointLights_.clear();
        distantLights_.clear();
        core::utility::log::Log::DefaultLog.WriteWarn(
            "[UncertaintyCartoonRenderer]: There are no proper lights connected no shading is happening");
    } else {
        if (lightCall->hasUpdate()) {
            auto& lights = lightCall->getData();

            pointLights_.clear();
            distantLights_.clear();

            auto point_lights = lights.get<core::view::light::PointLightType>();
            auto distant_lights = lights.get<core::view::light::DistantLightType>();

            for (const auto& pl : point_lights) {
                pointLights_.push_back({pl.position[0], pl.position[1], pl.position[2], pl.intensity});
            }

            for (const auto& dl : distant_lights) {
                if (dl.eye_direction) {
                    auto cd = glm::normalize(camDir); // paranoia
                    distantLights_.push_back({cd.x, cd.y, cd.z, dl.intensity});
                } else {
                    distantLights_.push_back({dl.direction[0], dl.direction[1], dl.direction[2], dl.intensity});
                }
            }
        }
    }

    pointLightBuffer_->rebuffer(pointLights_);
    distantLightBuffer_->rebuffer(distantLights_);

    if (pointLights_.empty() && distantLights_.empty()) {
        core::utility::log::Log::DefaultLog.WriteWarn("[DeferredRenderingProvider]: There are no directional or "
                                                      "positional lights connected. Lighting not available.");
    }
}
