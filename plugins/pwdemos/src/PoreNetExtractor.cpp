/*
 * PoreNetExtractor.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "PoreNetExtractor.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"
#include "vislib/Array.h"
#include "vislib/assert.h"
#include "vislib/Exception.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/MemmappedFile.h"
#include "vislib/math/Point.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"
#include "vislib/math/Vector.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <climits>
#include <cmath>


namespace megamol {
namespace demos {


/*
 * PoreNetExtractor::PoreNetExtractor
 */
PoreNetExtractor::PoreNetExtractor(void) : core::view::Renderer3DModule(), AbstractQuartzModule(),
typeTexture(0), bbox(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), cbox(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
filenameSlot("filename", "The file name of the pore network data file"),
streamSaveSlot("streamSave", "Saves the data to the pore network data file while extracting"),
extractDirSlot("extractDir", "The extraction direction"),
extractSizeSlot("extractSize", "The size of the extraction volume"),
extractTileSizeSlot("extractTileSize", "The size of a single rendering tile used for extraction"),
saveBtnSlot("save", "Saves the pore network data to the data file"),
loadBtnSlot("load", "Loads the pore network data from the data file"),
extractBtnSlot("extract", "Extractes the pore network data from the connected data modules"),
sizeX(1024), sizeY(1024), sizeZ(1024), edir(EXTDIR_X), saveFile(NULL), tile(NULL),
tileBuffer(NULL), slicesBuffers(), loopBuffers(), slice(UINT_MAX),
cryShader(), sliceProcessor(), meshProcessor(), debugLoopDataEntryObject() {

    //this->sizeX = 256;
    //this->sizeY = 256;
    //this->sizeZ = 256;

    this->MakeSlotAvailable(&this->dataInSlot);
    this->MakeSlotAvailable(&this->typesInSlot);

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->streamSaveSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->streamSaveSlot);

    core::param::EnumParam *dir = new core::param::EnumParam(static_cast<int>(this->edir));
    dir->SetTypePair(static_cast<int>(EXTDIR_X), "x");
    dir->SetTypePair(static_cast<int>(EXTDIR_Y), "y");
    dir->SetTypePair(static_cast<int>(EXTDIR_Z), "z");
    this->extractDirSlot << dir;
    this->MakeSlotAvailable(&this->extractDirSlot);

    vislib::StringA val;
    val.Format("%u;%u;%u", this->sizeX, this->sizeY, this->sizeZ);
    this->extractSizeSlot << new core::param::StringParam(val);
    this->MakeSlotAvailable(&this->extractSizeSlot);

    val = "1024;1024";
    this->extractTileSizeSlot << new core::param::StringParam(val);
    this->MakeSlotAvailable(&this->extractTileSizeSlot);

    this->saveBtnSlot << new core::param::ButtonParam();
    this->saveBtnSlot.SetUpdateCallback(&PoreNetExtractor::onSaveBtnClicked);
    this->MakeSlotAvailable(&this->saveBtnSlot);

    this->loadBtnSlot << new core::param::ButtonParam();
    this->loadBtnSlot.SetUpdateCallback(&PoreNetExtractor::onLoadBtnClicked);
    this->MakeSlotAvailable(&this->loadBtnSlot);

    this->extractBtnSlot << new core::param::ButtonParam();
    this->extractBtnSlot.SetUpdateCallback(&PoreNetExtractor::onExtractBtnClicked);
    this->MakeSlotAvailable(&this->extractBtnSlot);

    this->sliceProcessor.SetInputBuffers(this->slicesBuffers);
    this->sliceProcessor.SetOutputBuffers(this->loopBuffers);
    this->meshProcessor.SetInputBuffers(this->loopBuffers);

    this->debugLoopDataEntryObject.cnt = 0;
    this->debugLoopDataEntryObject.data = NULL;
    this->debugLoopDataEntryObject.next = NULL;
    this->meshProcessor.debugoutschlupp = &this->debugLoopDataEntryObject;
}


/*
 * PoreNetExtractor::~PoreNetExtractor
 */
PoreNetExtractor::~PoreNetExtractor(void) {
    this->Release();
}


/*
 * PoreNetExtractor::GetExtents
 */
bool PoreNetExtractor::GetExtents(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    cr->AccessBoundingBoxes().SetObjectSpaceClipBox(this->cbox);
    cr->AccessBoundingBoxes().MakeScaledWorld(1.0f); // for now
    cr->SetTimeFramesCount(1);

    return true;
}


/*
 * PoreNetExtractor::Render
 */
bool PoreNetExtractor::Render(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    if (this->isExtractionRunning()) {
        this->performExtraction();
    }
    else if (this->saveFile) {
        this->closeFile(*this->saveFile);
        this->saveFile->Flush();
        this->saveFile->Close();
        SAFE_DELETE(this->saveFile);
    }

    // TODO: Implement fancy rendering and fun and profit

    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ::glEnableClientState(GL_VERTEX_ARRAY);
    //::glColor3ub(255, 128, 0);
    PoreMeshProcessor::SliceLoops *i = this->debugLoopDataEntryObject.next;
    unsigned int c = 0;
    while (i != NULL) {
        ::glColor3ub(0, c, 255);
        c = (c + 128) % 256;
        ::glVertexPointer(3, GL_FLOAT, 0, i->data);
        ::glDrawArrays(GL_LINES, 0, i->cnt);
        i = i->next;
    }
    ::glDisableClientState(GL_VERTEX_ARRAY);

    return true;
}


/*
 * PoreNetExtractor::create
 */
bool PoreNetExtractor::create(void) {
    using vislib::graphics::gl::GLSLShader;
    using vislib::sys::Log;
    using vislib::graphics::gl::ShaderSource;

    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        Log::DefaultLog.WriteError("Unable to initialise GLSL extension");
        return false;
    }
    if (!vislib::graphics::gl::FramebufferObject::InitialiseExtensions()) {
        Log::DefaultLog.WriteError("Unable to initialise framebuffer object extension");
        return false;
    }
    if (!ogl_IsVersionGEQ(2, 0) || !isExtAvailable("GL_ARB_multitexture")) {
        Log::DefaultLog.WriteError("GL2.0 not present");
        return false;
    }

    ShaderSource vert, frag;
    try {
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("quartz::ray::plane::tex::vert", vert)) {
            throw vislib::Exception("Generic vertex shader build failure", __FILE__, __LINE__);
        }
        if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("quartz::ray::plane::tex::fragfaced", frag)) {
            throw vislib::Exception("Generic fragment shader build failure", __FILE__, __LINE__);
        }
        if (!this->cryShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            throw vislib::Exception("Generic shader create failure", __FILE__, __LINE__);
        }
    }
    catch (vislib::Exception ex) {
        Log::DefaultLog.WriteError("Unable to compile shader: %s", ex.GetMsgA());
        this->release(); // Because I know that 'release' ONLY releases all the shaders
        return false;
    }
    catch (...) {
        Log::DefaultLog.WriteError("Unable to compile shader: Unexpected Exception");
        this->release(); // Because I know that 'release' ONLY releases all the shaders
        return false;
    }

    // TODO: Implement

    return true;
}


/*
 * PoreNetExtractor::release
 */
void PoreNetExtractor::release(void) {
    if (this->isExtractionRunning()) {
        this->abortExtraction();
    }
    this->releaseTypeTexture();
    this->cryShader.Release();

    // TODO: Implement

}


/*
 * PoreNetExtractor::onExtractBtnClicked
 */
bool PoreNetExtractor::onExtractBtnClicked(core::param::ParamSlot& slot) {
    using vislib::sys::Log;

    if (this->isExtractionRunning()) {
        this->abortExtraction();
        return true;
    }

    Log::DefaultLog.WriteInfo("Pore network extraction requested");

    CrystalDataCall * cdc = this->getCrystaliteData();
    if (cdc == NULL) {
        Log::DefaultLog.WriteError("Crystal data not available\n");
        Log::DefaultLog.WriteError("Extraction aborted");
        return true;
    }
    ParticleGridDataCall *pgdc = this->getParticleData();
    if (pgdc == NULL) {
        cdc->Unlock();
        Log::DefaultLog.WriteError("Particle data not available\n");
        Log::DefaultLog.WriteError("Extraction aborted");
        return true;
    }

    // testing parameters
    ExtractionDir extDir = static_cast<ExtractionDir>(this->extractDirSlot.Param<core::param::EnumParam>()->Value());
    if ((extDir != EXTDIR_X) && (extDir != EXTDIR_Y) && (extDir != EXTDIR_Z)) {
        pgdc->Unlock();
        cdc->Unlock();
        Log::DefaultLog.WriteError("Extraction direction has an illegal value: %d\n", extDir);
        Log::DefaultLog.WriteError("Extraction aborted");
        return true;
    }
    this->edir = extDir;

    vislib::TString val = this->extractSizeSlot.Param<core::param::StringParam>()->Value();
    vislib::Array<vislib::TString> hit = vislib::TStringTokeniser::Split(val, _T(";"));
    if (hit.Count() == 3) {
        try {
            int x = vislib::TCharTraits::ParseInt(hit[0]);
            if (x < 1) throw vislib::Exception("x size must be 1 or larger", __FILE__, __LINE__);
            int y = vislib::TCharTraits::ParseInt(hit[1]);
            if (y < 1) throw vislib::Exception("y size must be 1 or larger", __FILE__, __LINE__);
            int z = vislib::TCharTraits::ParseInt(hit[2]);
            if (z < 1) throw vislib::Exception("z size must be 1 or larger", __FILE__, __LINE__);
            this->sizeX = static_cast<unsigned int>(x);
            this->sizeY = static_cast<unsigned int>(y);
            this->sizeZ = static_cast<unsigned int>(z);
        }
        catch (vislib::Exception ex) {
            pgdc->Unlock();
            cdc->Unlock();
            Log::DefaultLog.WriteError("Unable to parse \"extractSize\": %s", ex.GetMsgA());
            Log::DefaultLog.WriteError("Extraction aborted");
            return true;
        }
        catch (...) {
            pgdc->Unlock();
            cdc->Unlock();
            Log::DefaultLog.WriteError("Unable to parse \"extractSize\"");
            Log::DefaultLog.WriteError("Extraction aborted");
            return true;
        }
    }
    else {
        pgdc->Unlock();
        cdc->Unlock();
        Log::DefaultLog.WriteError("Unable to parse \"extractSize\"");
        Log::DefaultLog.WriteError("Extraction aborted");
        return true;
    }

    int tileW = 1024, tileH = 1024;
    val = this->extractTileSizeSlot.Param<core::param::StringParam>()->Value();
    hit = vislib::TStringTokeniser::Split(val, _T(";"));
    if (hit.Count() == 2) {
        try {
            int texSize = 256;
            ::glGetIntegerv(GL_MAX_TEXTURE_SIZE, &texSize);
            int x = vislib::TCharTraits::ParseInt(hit[0]);
            if (x < 1) throw vislib::Exception("width must be 1 or larger", __FILE__, __LINE__);
            if (x > texSize) throw vislib::Exception("OpenGL texture size limit reached", __FILE__, __LINE__);
            int y = vislib::TCharTraits::ParseInt(hit[1]);
            if (y < 1) throw vislib::Exception("height must be 1 or larger", __FILE__, __LINE__);
            if (y > texSize) throw vislib::Exception("OpenGL texture size limit reached", __FILE__, __LINE__);
            tileW = static_cast<unsigned int>(x);
            tileH = static_cast<unsigned int>(y);
        }
        catch (vislib::Exception ex) {
            pgdc->Unlock();
            cdc->Unlock();
            Log::DefaultLog.WriteError("Unable to parse \"extractTileSize\": %s", ex.GetMsgA());
            Log::DefaultLog.WriteError("Extraction aborted");
            return true;
        }
        catch (...) {
            pgdc->Unlock();
            cdc->Unlock();
            Log::DefaultLog.WriteError("Unable to parse \"extractTileSize\"");
            Log::DefaultLog.WriteError("Extraction aborted");
            return true;
        }
    }
    else {
        pgdc->Unlock();
        cdc->Unlock();
        Log::DefaultLog.WriteError("Unable to parse \"extractTileSize\"");
        Log::DefaultLog.WriteError("Extraction aborted");
        return true;
    }

    bool streamSave = this->streamSaveSlot.Param<core::param::BoolParam>()->Value();
    if (streamSave) {
        val = this->filenameSlot.Param<core::param::FilePathParam>()->Value();
        if (val.IsEmpty()) {
            Log::DefaultLog.WriteWarn("Cannot stream-save extracted data: no data file path was specified");
        }
        else {
            if (vislib::sys::File::Exists(val)) {
                Log::DefaultLog.WriteWarn("File specified for stream-save already exists and will be overwritten");
            }
            if (this->saveFile != NULL) delete this->saveFile;
            this->saveFile = new vislib::sys::MemmappedFile();
            if (!this->saveFile->Open(val, vislib::sys::File::WRITE_ONLY,
                vislib::sys::File::SHARE_READ, vislib::sys::File::CREATE_OVERWRITE)) {
                Log::DefaultLog.WriteError("Unable to open file for stream-save");
                SAFE_DELETE(this->saveFile);
            }
        }
    }

    // Initialize extraction objects
    if (this->tile != NULL) {
        delete this->tile;
    }
    this->tile = new vislib::graphics::gl::FramebufferObject();
    GLenum datatypetype;
    GLenum datatypeinterntype;
    switch (sizeof(ArxelBuffer::ArxelType)) {
    case 1:
        datatypetype = GL_UNSIGNED_BYTE;
        datatypeinterntype = GL_LUMINANCE8;
        break;
    case 2:
        datatypetype = GL_UNSIGNED_SHORT;
        datatypeinterntype = GL_LUMINANCE16;
        break;
    case 4:
        datatypetype = GL_FLOAT;
        datatypeinterntype = GL_LUMINANCE32F_ARB;
        break;
    }
    if (!this->tile->Create(tileW, tileH, datatypeinterntype, GL_LUMINANCE, datatypetype)) {
        pgdc->Unlock();
        cdc->Unlock();
        Log::DefaultLog.WriteError("Unable to create rendering tile");
        SAFE_DELETE(this->tile);
        if (this->saveFile != NULL) SAFE_DELETE(this->saveFile);
        Log::DefaultLog.WriteError("Extraction aborted");
        return true;
    }

    unsigned int bx, by, bz;
    vislib::math::Vector<float, 3> ax, ay, az;
    switch (this->edir) {
    case EXTDIR_X:
        bx = this->sizeZ;
        by = this->sizeY;
        bz = this->sizeX;
        ax.Set(0.0f, 0.0f, 1.0f / static_cast<float>(this->sizeZ));
        ay.Set(0.0f, 1.0f / static_cast<float>(this->sizeY), 0.0f);
        az.Set(1.0f / static_cast<float>(this->sizeX), 0.0f, 0.0f);
        break;
    case EXTDIR_Y:
        bx = this->sizeX;
        by = this->sizeZ;
        bz = this->sizeY;
        ax.Set(1.0f / static_cast<float>(this->sizeX), 0.0f, 0.0f);
        ay.Set(0.0f, 0.0f, 1.0f / static_cast<float>(this->sizeZ));
        az.Set(0.0f, 1.0f / static_cast<float>(this->sizeY), 0.0f);
        break;
    case EXTDIR_Z:
        bx = this->sizeX;
        by = this->sizeY;
        bz = this->sizeZ;
        ax.Set(1.0f / static_cast<float>(this->sizeX), 0.0f, 0.0f);
        ay.Set(0.0f, 1.0f / static_cast<float>(this->sizeY), 0.0f);
        az.Set(0.0f, 0.0f, 1.0f / static_cast<float>(this->sizeZ));
        break;
    default:
        pgdc->Unlock();
        cdc->Unlock();
        Log::DefaultLog.WriteError("Internal infernal error #7");
        SAFE_DELETE(this->tile);
        if (this->saveFile != NULL) SAFE_DELETE(this->saveFile);
        Log::DefaultLog.WriteError("Extraction aborted");
        return true;
    }
    ArxelBuffer::InitValues abiv;
    abiv.width = bx;
    abiv.height = by;

    if (this->tileBuffer != NULL) {
        delete[] this->tileBuffer;
    }
    this->tileBuffer = new ArxelBuffer::ArxelType[tileW * tileH];

    this->slice = 0;

    this->clear();

    if ((*pgdc)(ParticleGridDataCall::CallForGetExtent)) {
        this->bbox = pgdc->AccessBoundingBoxes().ObjectSpaceBBox();
        this->cbox = pgdc->AccessBoundingBoxes().ObjectSpaceClipBox();
    }
    else {
        this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        this->cbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    }

    if (this->saveFile != NULL) {
        this->writeFileHeader(*this->saveFile);
    }

    pgdc->Unlock();
    cdc->Unlock();

    ax.SetX(ax.X() * this->bbox.Width());
    ax.SetY(ax.Y() * this->bbox.Height());
    ax.SetZ(ax.Z() * this->bbox.Depth());
    ay.SetX(ay.X() * this->bbox.Width());
    ay.SetY(ay.Y() * this->bbox.Height());
    ay.SetZ(ay.Z() * this->bbox.Depth());
    az.SetX(az.X() * this->bbox.Width());
    az.SetY(az.Y() * this->bbox.Height());
    az.SetZ(az.Z() * this->bbox.Depth());
    this->meshProcessor.SetGeometryInformation(this->bbox.GetLeftBottomBack(), ax, ay, az);

    this->slicesBuffers.Open();
    this->loopBuffers.Open();
    this->sliceProcessor.Start();
    this->meshProcessor.Start();

    Log::DefaultLog.WriteInfo("Going to extract %u slices", bz);

    this->debugLoopDataEntryObject.next = NULL; // memory leak here!

    return true;
}


/*
 * PoreNetExtractor::onLoadBtnClicked
 */
bool PoreNetExtractor::onLoadBtnClicked(core::param::ParamSlot& slot) {
    using vislib::sys::Log;

    // TODO: Implement
    Log::DefaultLog.WriteWarn("Loading not implemented yet.");

    return true;
}


/*
 * PoreNetExtractor::onSaveBtnClicked
 */
bool PoreNetExtractor::onSaveBtnClicked(core::param::ParamSlot& slot) {
    using vislib::sys::Log;

    // TODO: Implement
    Log::DefaultLog.WriteWarn("Saving not implemented yet.");

    return true;
}


/*
 * PoreNetExtractor::isExtractionRunning
 */
bool PoreNetExtractor::isExtractionRunning(void) {
    return (this->tile != NULL) || (this->tileBuffer != NULL) || (this->slice < UINT_MAX)
        || this->sliceProcessor.IsRunning() || this->meshProcessor.IsRunning();
}


/*
 * PoreNetExtractor::abortExtraction
 */
void PoreNetExtractor::abortExtraction(void) {
    using vislib::sys::Log;

    // TODO: Implement

    this->slicesBuffers.AbortClose();
    this->loopBuffers.AbortClose();
    if (this->sliceProcessor.IsRunning()) {
        this->sliceProcessor.Terminate();
        this->sliceProcessor.Join();
    }
    if (this->meshProcessor.IsRunning()) {
        this->meshProcessor.Terminate();
        this->meshProcessor.Join();
    }
    if (this->saveFile != NULL) {
        this->saveFile->Flush();
        this->saveFile->Close();
        SAFE_DELETE(this->saveFile);
    }
    if (this->tile != NULL) {
        SAFE_DELETE(this->tile);
    }
    if (this->tileBuffer != NULL) {
        ARY_SAFE_DELETE(this->tileBuffer);
    }
    this->slice = UINT_MAX;
    Log::DefaultLog.WriteInfo("Extraction aborted");
}


/*
 * PoreNetExtractor::performExtraction
 */
void PoreNetExtractor::performExtraction(void) {
    using vislib::sys::Log;

    if (this->slice == UINT_MAX) {
        this->slicesBuffers.EndOfDataClose(); // we are done with rendering
        return;
    }

    //
    // Step 0: Data perparation
    //////////////////////////////////////////////////////////////////////////
    unsigned int bx, by, bz;
    vislib::math::Plane<float> plane;
    float planeD, planeDa;
    switch (this->edir) {
    case EXTDIR_X:
        bx = this->sizeZ;
        by = this->sizeY;
        bz = this->sizeX;
        planeD = static_cast<float>(this->slice) + 0.5f;
        planeD /= static_cast<float>(bz);
        planeDa = planeD;
        planeD *= this->bbox.Width();
        plane.Set(1.0, 0.0, 0.0, -planeD); // what here
        break;
    case EXTDIR_Y:
        bx = this->sizeX;
        by = this->sizeZ;
        bz = this->sizeY;
        planeD = static_cast<float>(this->slice) + 0.5f;
        planeD /= static_cast<float>(bz);
        planeDa = planeD;
        planeD *= this->bbox.Height();
        plane.Set(0.0, 1.0, 0.0, -planeD);
        break;
    case EXTDIR_Z:
        bx = this->sizeX;
        by = this->sizeY;
        bz = this->sizeZ;
        planeD = static_cast<float>(this->slice) + 0.5f;
        planeD /= static_cast<float>(bz);
        planeDa = planeD;
        planeD *= this->bbox.Depth();
        plane.Set(0.0, 0.0, 1.0, -planeD);
        break;
    default:
        Log::DefaultLog.WriteError("Internal infernal error #7");
        SAFE_DELETE(this->tile);
        if (this->saveFile != NULL) SAFE_DELETE(this->saveFile);
        Log::DefaultLog.WriteError("Extraction aborted");
        return;
    }

    if ((this->slice < UINT_MAX) && ((this->slice >= bz) || (this->tile == NULL) || (this->tileBuffer == NULL))) {
        Log::DefaultLog.WriteInfo("Slice rendering complete");
        this->slice = UINT_MAX;
        if (this->tile != NULL) { SAFE_DELETE(this->tile); }
        if (this->tileBuffer != NULL) { ARY_SAFE_DELETE(this->tileBuffer); }
        this->releaseTypeTexture();
        // Done with the rendering
        this->slicesBuffers.EndOfDataClose();

        // TODO: Change this!
        //Log::DefaultLog.WriteInfo("#=====================#");
        //Log::DefaultLog.WriteInfo("| Extraction complete |");
        //Log::DefaultLog.WriteInfo("#=====================#");
        //if (this->saveFile) {
        //    this->closeFile(*this->saveFile);
        //    this->saveFile->Flush();
        //    this->saveFile->Close();
        //    SAFE_DELETE(this->saveFile);
        //}
        return;
    }

    ArxelBuffer* buffer = this->slicesBuffers.GetEmptyBuffer(false);
    if (buffer == NULL) {
        // no buffer available, so we skip extraction for this frame
        return;
    }
    if ((buffer->Data() == NULL) || (buffer->Width() != bx) || (buffer->Height() != by)) {
        ArxelBuffer::InitValues iv;
        iv.width = bx;
        iv.height = by;
        int dummy;
        ArxelBuffer::Initialize(*buffer, dummy, iv);
    }

    CrystalDataCall * cdc = this->getCrystaliteData();
    if (cdc == NULL) {
        Log::DefaultLog.WriteError("Crystal data not available\n");
        this->abortExtraction();
        return;
    }
    ParticleGridDataCall *pgdc = this->getParticleData();
    if (pgdc == NULL) {
        cdc->Unlock();
        Log::DefaultLog.WriteError("Particle data not available\n");
        this->abortExtraction();
        return;
    }
    this->assertTypeTexture(*cdc);

    Log::DefaultLog.WriteInfo("Extract slice %d @ %f\n", this->slice, planeDa);

    //
    // Step 1: render tiles to extract slice
    //////////////////////////////////////////////////////////////////////////
    ::glMatrixMode(GL_PROJECTION);
    ::glPushMatrix();
    ::glMatrixMode(GL_MODELVIEW);
    ::glPushMatrix();

    ::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    ::glDisable(GL_DEPTH_TEST);
    ::glDisable(GL_LIGHTING);
    ::glColor3ub(255, 255, 255);
    ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    ::glDisable(GL_BLEND);
    ::glDisable(GL_CULL_FACE);
    ::glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    core::view::CallClipPlane ccp;
    ccp.SetColour(255, 255, 255);
    ccp.SetPlane(plane);

    vislib::math::Vector<float, 3> cx, cy, p;
    ccp.CalcPlaneSystem(cx, cy, p);

    p = this->bbox.GetLeftBottomBack();
    vislib::math::Point<float, 2> p2(p.Dot(cx), p.Dot(cy));
    vislib::math::Rectangle<float> brect(p2, vislib::math::Dimension<float, 2>(0.0f, 0.0f));

    p = this->bbox.GetRightBottomBack();
    p2.Set(p.Dot(cx), p.Dot(cy));
    brect.Union(vislib::math::Rectangle<float>(p2, vislib::math::Dimension<float, 2>(0.0f, 0.0f)));

    p = this->bbox.GetLeftTopBack();
    p2.Set(p.Dot(cx), p.Dot(cy));
    brect.Union(vislib::math::Rectangle<float>(p2, vislib::math::Dimension<float, 2>(0.0f, 0.0f)));

    p = this->bbox.GetRightTopBack();
    p2.Set(p.Dot(cx), p.Dot(cy));
    brect.Union(vislib::math::Rectangle<float>(p2, vislib::math::Dimension<float, 2>(0.0f, 0.0f)));

    p = this->bbox.GetLeftBottomFront();
    p2.Set(p.Dot(cx), p.Dot(cy));
    brect.Union(vislib::math::Rectangle<float>(p2, vislib::math::Dimension<float, 2>(0.0f, 0.0f)));

    p = this->bbox.GetRightBottomFront();
    p2.Set(p.Dot(cx), p.Dot(cy));
    brect.Union(vislib::math::Rectangle<float>(p2, vislib::math::Dimension<float, 2>(0.0f, 0.0f)));

    p = this->bbox.GetLeftTopFront();
    p2.Set(p.Dot(cx), p.Dot(cy));
    brect.Union(vislib::math::Rectangle<float>(p2, vislib::math::Dimension<float, 2>(0.0f, 0.0f)));

    p = this->bbox.GetRightTopFront();
    p2.Set(p.Dot(cx), p.Dot(cy));
    brect.Union(vislib::math::Rectangle<float>(p2, vislib::math::Dimension<float, 2>(0.0f, 0.0f)));

    for (unsigned int y = 0; y < by; y += this->tile->GetHeight()) {
        unsigned int h = this->tile->GetHeight();
        if (y + h > by) h = by - y;
        for (unsigned int x = 0; x < bx; x += this->tile->GetWidth()) {
            unsigned int w = this->tile->GetWidth();
            if (x + w > bx) w = bx - x;
            this->tile->Enable();

            ::glViewport(0, 0, w, h);
            ::glClear(GL_COLOR_BUFFER_BIT);

            ::glMatrixMode(GL_PROJECTION);
            ::glLoadIdentity();
            ::glMatrixMode(GL_MODELVIEW);
            ::glLoadIdentity();

            //::glColor3ub(128, 128, 128);
            //::glBegin(GL_LINES);
            //::glVertex2f(-1.0f, 1.0f);
            //::glVertex2f(1.0f, -1.0f);
            //::glEnd();

            ::glTranslatef(-1.0f, -1.0f, -1.0f);
            ::glScalef(static_cast<float>(bx) / static_cast<float>(w), static_cast<float>(by) / static_cast<float>(h), 1.0f);
            ::glTranslatef(-2.0f * static_cast<float>(x) / static_cast<float>(bx), -2.0f * static_cast<float>(y) / static_cast<float>(by), 0.0f);
            ::glScalef(2.0f / brect.Width(), 2.0f / brect.Height(), 1.0f);
            ::glTranslatef(-brect.Left(), -brect.Bottom(), 0.0f);

            //::glColor3ub(212, 212, 212);
            //::glBegin(GL_LINES);
            //::glVertex3f(this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back());
            //::glVertex3f(this->bbox.Right(), this->bbox.Top(), this->bbox.Front());
            //::glEnd();

            //::glColor3ub(192, 192, 192);
            //vislib::math::Point<float, 3> cp = this->bbox.CalcCenter();
            //vislib::math::Vector<float, 3> cxv(this->bbox.Width() * 0.5f, 0.0f, 0.0f);
            //vislib::math::Vector<float, 3> cyv(0.0f, this->bbox.Height() * 0.5f, 0.0f);
            //for (double r = 1.0; r > 0.1; r -= 0.1) {
            //    ::glBegin(GL_LINE_LOOP);
            //    for (double a = 0.0; a < 2.0 * M_PI; a += 0.01) {
            //        ::glVertex3fv((cp + static_cast<float>(r * sin(a)) * cxv + static_cast<float>(r * cos(a)) * cyv).PeekCoordinates());
            //    }
            //    ::glEnd();
            //}

            //::glColor3ub(255, 255, 255);
            this->drawParticles(pgdc, cdc, &ccp);

            ::glFlush();

            this->tile->Disable();
            GLenum datatypetype;
            switch (sizeof(ArxelBuffer::ArxelType)) {
            case 1: datatypetype = GL_UNSIGNED_BYTE; break;
            case 2: datatypetype = GL_UNSIGNED_SHORT; break;
            case 4: datatypetype = GL_FLOAT; break;
            }
            GLenum rv = this->tile->GetColourTexture(this->tileBuffer, 0, GL_LUMINANCE, datatypetype);
            if (rv != GL_NO_ERROR) {
                Log::DefaultLog.WriteError("Pixel-Fetch-Error: %d\n", static_cast<int>(rv));
            }
            for (unsigned int ly = 0; ly < h; ly++) {
                ::memcpy(buffer->Data() + ((y + ly) * bx + x),
                    this->tileBuffer + (ly * this->tile->GetWidth()),
                    w * sizeof(ArxelBuffer::ArxelType));
            }
        }
    }

    ::glPopMatrix();
    ::glMatrixMode(GL_PROJECTION);
    ::glPopMatrix();
    ::glMatrixMode(GL_MODELVIEW);

    this->slicesBuffers.BufferFilled(buffer);

    //
    // Finished
    //////////////////////////////////////////////////////////////////////////
    Log::DefaultLog.WriteInfo("Slice %d @ %f extracted\n", this->slice, planeDa);
    this->slice++;
    cdc->Unlock();
    pgdc->Unlock();

    //if (this->slice >= 2) { // TODO: This is DEBUG
    //    this->slice = UINT_MAX - 1;
    //}
}


/*
 * PoreNetExtractor::writeFileHeader
 */
void PoreNetExtractor::writeFileHeader(vislib::sys::File& file) {
    file.SeekToBegin();
    //file.Write("MMPN\x00\xFF", 6); // ID
    unsigned short vn = 100; // version 1.0
    //file.Write(&vn, 2);

    // TODO: Implement

    //
    //unsigned int bx, by, bz;
    //char edb;
    //switch (this->edir) {
    //    case EXTDIR_X:
    //        bx = this->sizeZ;
    //        by = this->sizeY;
    //        bz = this->sizeX;
    //        edb = 'x';
    //        break;
    //    case EXTDIR_Y:
    //        bx = this->sizeX;
    //        by = this->sizeZ;
    //        bz = this->sizeY;
    //        edb = 'y';
    //        break;
    //    case EXTDIR_Z:
    //        bx = this->sizeX;
    //        by = this->sizeY;
    //        bz = this->sizeZ;
    //        edb = 'z';
    //        break;
    //    default:
    //        bx = this->sizeX;
    //        by = this->sizeY;
    //        bz = this->sizeZ;
    //        edb = 'e';
    //        return;
    //}

    //file.Write(&edb, 1); // extraction direction
    //file.Write(&bx, 4); // slice width
    //file.Write(&by, 4); // slice height
    //file.Write(&bz, 4); // # slices
    //// end of 21 byte header
    //// now slice data
}


/*
 * PoreNetExtractor::closeFile
 */
void PoreNetExtractor::closeFile(vislib::sys::File& file) {
    // TODO: Implement
}


/*
 * PoreNetExtractor::clear
 */
void PoreNetExtractor::clear(void) {
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->cbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    // TODO: Implement
}


/*
 * PoreNetExtractor::assertTypeTexture
 */
void PoreNetExtractor::assertTypeTexture(CrystalDataCall& types) {
    if ((this->typesDataHash != 0) && (this->typesDataHash == types.DataHash())) return; // all up to date
    this->typesDataHash = types.DataHash();

    if (types.GetCount() == 0) {
        ::glDeleteTextures(1, &this->typeTexture);
        this->typeTexture = 0;
        return;
    }
    if (this->typeTexture == 0) {
        ::glGenTextures(1, &this->typeTexture);
    }

    unsigned mfc = 0;
    for (unsigned int i = 0; i < types.GetCount(); i++) {
        if (mfc < types.GetCrystals()[i].GetFaceCount()) {
            mfc = types.GetCrystals()[i].GetFaceCount();
        }
    }

    float *dat = new float[types.GetCount() * mfc * 4];

    for (unsigned int y = 0; y < types.GetCount(); y++) {
        const CrystalDataCall::Crystal& c = types.GetCrystals()[y];
        unsigned int x;
        for (x = 0; x < c.GetFaceCount(); x++) {
            vislib::math::Vector<float, 3> f = c.GetFace(x);
            dat[(x + y * mfc) * 4 + 3] = f.Normalise();
            dat[(x + y * mfc) * 4 + 0] = f.X();
            dat[(x + y * mfc) * 4 + 1] = f.Y();
            dat[(x + y * mfc) * 4 + 2] = f.Z();
        }
        for (; x < mfc; x++) {
            dat[(x + y * mfc) * 4 + 0] = 0.0f;
            dat[(x + y * mfc) * 4 + 1] = 0.0f;
            dat[(x + y * mfc) * 4 + 2] = 0.0f;
            dat[(x + y * mfc) * 4 + 3] = 0.0f;
        }
    }

    ::glBindTexture(GL_TEXTURE_2D, this->typeTexture);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    ::glPixelStorei(GL_PACK_ALIGNMENT, 1);
    ::glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mfc, types.GetCount(), 0, GL_RGBA, GL_FLOAT, dat);
    ::glBindTexture(GL_TEXTURE_2D, 0);

    delete[] dat;
}


/*
 * PoreNetExtractor::releaseTypeTexture
 */
void PoreNetExtractor::releaseTypeTexture(void) {
    ::glDeleteTextures(1, &this->typeTexture);
    this->typeTexture = 0;
}


/*
 * PoreNetExtractor::drawParticles
 */
void PoreNetExtractor::drawParticles(ParticleGridDataCall *pgdc,
    CrystalDataCall *tdc, core::view::CallClipPlane *ccp) {
    ASSERT(pgdc != NULL);
    ASSERT(tdc != NULL);
    ASSERT(ccp != NULL);

    vislib::math::Vector<float, 3> cx, cy, cz;
    ccp->CalcPlaneSystem(cx, cy, cz);

    float viewportStuff[4];
    ::glGetFloatv(GL_VIEWPORT, viewportStuff);
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    ::glPointSize(static_cast<float>(vislib::math::Max(this->sizeX, vislib::math::Max(this->sizeY, this->sizeZ))));

    ::glEnableClientState(GL_VERTEX_ARRAY); // xyzr
    ::glEnableClientState(GL_TEXTURE_COORD_ARRAY); // quart

    float planeZ = ccp->GetPlane().Distance(vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f));
    vislib::math::Point<float, 3> bboxmin(
        vislib::math::Min(this->bbox.Left(), this->bbox.Right()),
        vislib::math::Min(this->bbox.Bottom(), this->bbox.Top()),
        vislib::math::Min(this->bbox.Back(), this->bbox.Front()));
    vislib::math::Point<float, 3> bboxmax(
        vislib::math::Max(this->bbox.Left(), this->bbox.Right()),
        vislib::math::Max(this->bbox.Bottom(), this->bbox.Top()),
        vislib::math::Max(this->bbox.Back(), this->bbox.Front()));
    const bool fixPBC(true);
    if (!fixPBC) {
        bboxmin.Set(0.0f, 0.0f, 0.0f);
        bboxmax.Set(0.0f, 0.0f, 0.0f);
    }

    this->cryShader.Enable();
    this->cryShader.SetParameterArray4("viewAttr", 1, viewportStuff);
    this->cryShader.SetParameterArray3("camX", 1, cx.PeekComponents());
    this->cryShader.SetParameterArray3("camY", 1, cy.PeekComponents());
    this->cryShader.SetParameterArray3("camZ", 1, cz.PeekComponents());
    this->cryShader.SetParameterArray3("bboxmin", 1, bboxmin.PeekCoordinates());
    this->cryShader.SetParameterArray3("bboxmax", 1, bboxmax.PeekCoordinates());
    this->cryShader.SetParameter("planeZ", planeZ);

    ::glActiveTexture(GL_TEXTURE0);
    ::glBindTexture(GL_TEXTURE_2D, this->typeTexture);
    this->cryShader.SetParameter("typeData", 0);

    for (int cellX = (fixPBC ? -1 : 0); cellX < static_cast<int>(pgdc->SizeX() + (fixPBC ? 1 : 0)); cellX++) {
        int ccx = cellX;
        float xoff = 0.0f;
        if (ccx < 0) {
            ccx = pgdc->SizeX() - 1;
            xoff -= this->bbox.Width();
        }
        if (ccx >= static_cast<int>(pgdc->SizeX())) {
            ccx = 0;
            xoff += this->bbox.Width();
        }

        for (int cellY = (fixPBC ? -1 : 0); cellY < static_cast<int>(pgdc->SizeY() + (fixPBC ? 1 : 0)); cellY++) {
            int ccy = cellY;
            float yoff = 0.0f;
            if (ccy < 0) {
                ccy = pgdc->SizeY() - 1;
                yoff -= this->bbox.Height();
            }
            if (ccy >= static_cast<int>(pgdc->SizeY())) {
                ccy = 0;
                yoff += this->bbox.Height();
            }

            for (int cellZ = (fixPBC ? -1 : 0); cellZ < static_cast<int>(pgdc->SizeZ() + (fixPBC ? 1 : 0)); cellZ++) {
                int ccz = cellZ;
                float zoff = 0.0f;
                if (ccz < 0) {
                    ccz = pgdc->SizeZ() - 1;
                    zoff -= this->bbox.Depth();
                }
                if (ccz >= static_cast<int>(pgdc->SizeZ())) {
                    ccz = 0;
                    zoff += this->bbox.Depth();
                }
                GL_VERIFY(this->cryShader.SetParameter("posoffset", xoff, yoff, zoff));

                unsigned int cellIdx = static_cast<unsigned int>(ccx
                    + pgdc->SizeX() * (ccy + pgdc->SizeY() * ccz));

                const ParticleGridDataCall::Cell& cell = pgdc->Cells()[cellIdx];

                bool hasPos = false, hasNeg = false;
                vislib::math::Cuboid<float> ccbox = cell.ClipBox();
                ccbox.Move(xoff, yoff, zoff);
                if (ccp->GetPlane().Halfspace(ccbox.GetRightTopFront()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetRightTopBack()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetRightBottomFront()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetRightBottomBack()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetLeftTopFront()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetLeftTopBack()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetLeftBottomFront()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (ccp->GetPlane().Halfspace(ccbox.GetLeftBottomBack()) ==
                    vislib::math::Plane<float>::POSITIVE_HALFSPACE) {
                    hasPos = true;
                }
                else { hasNeg = true; }
                if (!hasPos || !hasNeg) continue; // not visible cell

                for (unsigned int listIdx = 0; listIdx < cell.Count(); listIdx++) {
                    const ParticleGridDataCall::List& list = cell.Lists()[listIdx];

                    this->cryShader.SetParameter("typeInfo",
                        static_cast<int>(list.Type()),
                        static_cast<int>(tdc->GetCrystals()[list.Type()].GetFaceCount()));
                    this->cryShader.SetParameter("outerRad",
                        tdc->GetCrystals()[list.Type()].GetBoundingRadius());

                    ::glVertexPointer(4, GL_FLOAT, 8 * sizeof(float), list.Data());
                    ::glTexCoordPointer(4, GL_FLOAT, 8 * sizeof(float), list.Data() + 4);
                    ::glDrawArrays(GL_POINTS, 0, list.Count());
                }
            }
        }
    }

    this->cryShader.Disable();
    ::glBindTexture(GL_TEXTURE_2D, 0);
    ::glDisableClientState(GL_VERTEX_ARRAY); // xyzr
    ::glDisableClientState(GL_TEXTURE_COORD_ARRAY); // quart

}

} /* end namespace demos */
} /* end namespace megamol */