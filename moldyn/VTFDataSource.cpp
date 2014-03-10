/*
 * VTFDataSource.cpp
 *
 * Copyright (C) 2014 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "VTFDataSource.h"
#include "param/FilePathParam.h"
#include "param/StringParam.h"
#include "MultiParticleDataCall.h"
#include "CoreInstance.h"
#include "vislib/error.h"
#include "vislib/Log.h"
#include "vislib/Path.h"
#include "vislib/PtrArray.h"
#include "vislib/RawStorageWriter.h"
#include "vislib/ShallowPoint.h"
#include "vislib/ShallowQuaternion.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sysfunctions.h"
#include "vislib/SystemInformation.h"
#include "vislib/Trace.h"
#include "vislib/ConsoleProgressBar.h"

using namespace megamol::core;


/* defines for the frame cache size */
// minimum number of frames in the cache (2 for interpolation; 1 for loading)
#define CACHE_SIZE_MIN 3
// maximum number of frames in the cache (just a nice number)
#define CACHE_SIZE_MAX 100000
// factor multiplied to the frame size for estimating the overhead to the pure data.
#define CACHE_FRAME_FACTOR 1.15f

/*****************************************************************************/

/*
 * moldyn::VTFDataSource::Frame::Frame
 */

moldyn::VTFDataSource::Frame::Frame(view::AnimDataModule& owner)
        : view::AnimDataModule::Frame(owner), typeCnt(0), partCnt(),
        pos() {
}


/*
 * moldyn::VTFDataSource::Frame::~Frame
 */
moldyn::VTFDataSource::Frame::~Frame() {
    this->typeCnt = 0;
    this->partCnt.Clear();
	this->pos.Clear();
}


/*
 * moldyn::VTFDataSource::Frame::Clear
 */
void moldyn::VTFDataSource::Frame::Clear(void) {
    for (unsigned int i = 0; i < this->typeCnt; i++) {
        this->pos[i].EnforceSize(0);
    }
}


/*
 * moldyn::VTFDataSource::Frame::LoadFrame
 */
bool moldyn::VTFDataSource::Frame::LoadFrame(vislib::sys::File *file, unsigned int idx, vislib::Array<SimpleType> &types) {
/*
	timestep indexed
	0 88.08974923911063 93.53975290469917 41.0842180843088940
	1 50.542528784672555 69.2565090323587 62.71274455546361
	2 62.747125087753524 49.00074973246766 75.57611795542917
	3 9.46248516175452 43.90389079646931 43.07396560057581
	4 0.32858672109087456 58.02125782527474 64.42774367401746
*/
	
    this->frame = idx;
	this->partCnt.Resize(types.Count());
	while(!file->IsEOF()) {
        vislib::StringA line = vislib::sys::ReadLineFromFileA(*file);
        line.TrimSpaces();
    
		vislib::Array<vislib::StringA> shreds = vislib::StringTokeniserA::Split(line, ' ', true);

		if (line.IsEmpty() || line.StartsWithInsensitive("timestep")) {
            break;
		}

		vislib::math::Vector<float, 3> pos;
		pos.Set(vislib::CharTraitsA::ParseDouble(shreds[1]),
			    vislib::CharTraitsA::ParseDouble(shreds[2]),
				vislib::CharTraitsA::ParseDouble(shreds[3]));
		this->pos[0].Append(pos.PeekComponents(), 3 * sizeof(float));

		++this->partCnt[0];
	}

    VLTRACE(VISLIB_TRCELVL_INFO, "Frame %u loaded\n", this->frame);

    return true;
}

/*
 * moldyn::VTFDataSource::Frame::PartPoss
 */
const float *moldyn::VTFDataSource::Frame::PartPoss(unsigned int type) const {
    ASSERT(type < this->typeCnt);
    return this->pos[type].As<float>();
}


/*
 * moldyn::VTFDataSource::Frame::SizeOf
 */
SIZE_T moldyn::VTFDataSource::Frame::SizeOf(void) const {
    SIZE_T size = 0;
    for (unsigned int i = 0; i < this->typeCnt; i++) {
        size += this->pos[i].GetSize();
    }
    return size;
}


/*
 * moldyn::VTFDataSource::Frame::MakeInterpolationFrame
 */
const moldyn::VTFDataSource::Frame *
moldyn::VTFDataSource::Frame::MakeInterpolationFrame(float alpha,
        const moldyn::VTFDataSource::Frame &a,
        const moldyn::VTFDataSource::Frame &b) {
    ASSERT(a.typeCnt == b.typeCnt);

    if (alpha < 0.0000001f) return &a;
    if (alpha > 0.9999999f) return &b;
    float beta = 1.0f - alpha;

    for (unsigned int t = 0; t < a.typeCnt; t++) {
        if (a.partCnt[t] != b.partCnt[t]) return &a;
        this->partCnt[t] = a.partCnt[t];
        this->pos[t].AssertSize(this->partCnt[t] * 3 * sizeof(float));
    }

    for (unsigned int t = 0; t < a.typeCnt; t++) {
        for (unsigned int i = 0; i < this->partCnt[t]; i++) {
            vislib::math::ShallowPoint<float, 3> av((float*)a.pos[t].As<float>() + i * 3);
            vislib::math::ShallowPoint<float, 3> bv((float*)b.pos[t].As<float>() + i * 3);
            vislib::math::ShallowPoint<float, 3> tv(this->pos[t].As<float>() + i * 3);

            if (av.SquareDistance(bv) > 0.01) {
                tv = (alpha < 0.5f) ? av : bv;
            } else {
                tv.Set(av.X() * beta + bv.X() * alpha, 
                    av.Y() * beta + bv.Y() * alpha, 
                    av.Z() * beta + bv.Z() * alpha);
            }
        }
    }

    return this;
}


/*
 * moldyn::VTFDataSource::Frame::parseParticleLine
 */
void moldyn::VTFDataSource::Frame::parseParticleLine(vislib::StringA &line,
        int &outType, float &outX, float &outY, float &outZ) {

    vislib::Array<vislib::StringA> shreds
        = vislib::StringTokeniserA::Split(line, ' ', true);
    if ((shreds.Count() != 9) && (shreds.Count() != 5)) {
        throw 0; // invalid line separations
    }
    if (!shreds[0].Equals("!")) {
        throw 0; // invalid line marker
    }

    outType = vislib::CharTraitsA::ParseInt(shreds[1]);
    outX = float(vislib::CharTraitsA::ParseInt(shreds[2])); // de-quantization of positions is done later
    outY = float(vislib::CharTraitsA::ParseInt(shreds[3]));
    outZ = float(vislib::CharTraitsA::ParseInt(shreds[4]));
}


/*
 * moldyn::VTFDataSource::Frame::SetTypeCount
 */
void moldyn::VTFDataSource::Frame::SetTypeCount(unsigned int cnt) {
    this->typeCnt = cnt;
    this->partCnt.Clear();
    this->pos.Clear();
    this->partCnt.Resize(cnt);
    this->pos.Resize(cnt);
    for (unsigned int i = 0; i < cnt; i++) {
		this->pos.Append(*new vislib::RawStorage());
        this->partCnt.Append(0);
	}
}

/*****************************************************************************/


/*
 * moldyn::VTFDataSource::VTFDataSource
 */
moldyn::VTFDataSource::VTFDataSource(void) : view::AnimDataModule(),
        filename("filename", "The path to the trisoup file to load."),
        getData("getdata", "Slot to request data from this data source."),
        file(NULL), types(), frameIdx(),
        datahash(0) {

    this->filename.SetParameter(new param::FilePathParam(""));
    this->filename.SetUpdateCallback(&VTFDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->getData.SetCallback("MultiParticleDataCall", "GetData",
        &VTFDataSource::getDataCallback);
    this->getData.SetCallback("MultiParticleDataCall", "GetExtent",
        &VTFDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->setFrameCount(1);
    this->initFrameCache(1);
}


/*
 * moldyn::VTFDataSource::~VTFDataSource
 */
moldyn::VTFDataSource::~VTFDataSource(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * moldyn::VTFDataSource::constructFrame
 */
view::AnimDataModule::Frame*
moldyn::VTFDataSource::constructFrame(void) const {
    Frame *f = new Frame(*const_cast<moldyn::VTFDataSource*>(this));
    f->SetTypeCount(this->types.Count());
    return f;
}


/*
 * moldyn::VTFDataSource::create
 */
bool moldyn::VTFDataSource::create(void) {
    return true;
}


/*
 * moldyn::VTFDataSource::loadFrame
 */
void moldyn::VTFDataSource::loadFrame(view::AnimDataModule::Frame *frame,
        unsigned int idx) {
    Frame *f = dynamic_cast<Frame*>(frame);
    if (f == NULL) return;
    if (this->file == NULL) {
        f->Clear();
        return;
    }
    ASSERT(idx < this->FrameCount());

    this->file->Seek(this->frameIdx[idx]);
    f->LoadFrame(this->file, idx, this->types);
}


/*
 * moldyn::VTFDataSource::release
 */
void moldyn::VTFDataSource::release(void) {
    this->resetFrameCache();
    if (this->file != NULL) {
        vislib::sys::File *f = this->file;
        this->file = NULL;
        f->Close();
        delete f;
    }
	this->types.Clear();
	this->frameIdx.Clear();
}

/*
 * moldyn::VTFDataSource::filenameChanged
 */
bool moldyn::VTFDataSource::filenameChanged(param::ParamSlot& slot) {

	this->types.Clear();
	this->frameIdx.Clear();
	this->resetFrameCache();

    this->datahash++;

    if (this->file == NULL) {
        //this->file = new vislib::sys::MemmappedFile();
        this->file = new vislib::sys::File();
    } else {
        this->file->Close();
    }
    ASSERT(this->filename.Param<param::FilePathParam>() != NULL);

    if (!this->file->Open(this->filename.Param<param::FilePathParam>()->Value(),
            vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        vislib::sys::SystemMessage err(::GetLastError());
        this->GetCoreInstance()->Log().WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to open VTF-File \"%s\": %s", vislib::StringA(
            this->filename.Param<param::FilePathParam>()->Value()).PeekBuffer(),
            static_cast<const char*>(err));

        SAFE_DELETE(this->file);
        this->setFrameCount(1);
        this->initFrameCache(1);

        return true;
    }

    if (!this->parseHeaderAndFrameIndices(this->filename.Param<param::FilePathParam>()->Value())) {
        this->GetCoreInstance()->Log().WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to read VTF-Header from file \"%s\". Wrong format?", vislib::StringA(
            this->filename.Param<param::FilePathParam>()->Value()).PeekBuffer());

        this->file->Close();
        SAFE_DELETE(this->file);
        this->setFrameCount(1);
        this->initFrameCache(1);

        return true;
    }

    Frame tmpFrame(*this);
	tmpFrame.SetTypeCount(this->types.Count());
    // use frame zero to estimate the frame size in memory to calculate the
    // frame cache size
    this->loadFrame(&tmpFrame, 0);
    SIZE_T frameSize = tmpFrame.SizeOf();
    tmpFrame.Clear();
    frameSize = static_cast<SIZE_T>(float(frameSize) * CACHE_FRAME_FACTOR);
    UINT64 mem = vislib::sys::SystemInformation::AvailableMemorySize();
    unsigned int cacheSize = static_cast<unsigned int>(mem / frameSize);

    if (cacheSize > CACHE_SIZE_MAX) {
        cacheSize = CACHE_SIZE_MAX;
    }
    if (cacheSize < CACHE_SIZE_MIN) {
        vislib::StringA msg;
        msg.Format("Frame cache size forced to %i. Calculated size was %u.\n",
            CACHE_SIZE_MIN, cacheSize);
        this->GetCoreInstance()->Log().WriteMsg(vislib::sys::Log::LEVEL_WARN, msg);
        cacheSize = CACHE_SIZE_MIN;
    } else {
        vislib::StringA msg;
        msg.Format("Frame cache size set to %i.\n", cacheSize);
        this->GetCoreInstance()->Log().WriteMsg(vislib::sys::Log::LEVEL_INFO, msg);
    }

    this->initFrameCache(cacheSize);

    return true; // to reset the dirty flag of the param slot
}

/*
 * moldyn::VTFDataSource::readHeader
 */
bool moldyn::VTFDataSource::parseHeaderAndFrameIndices(const vislib::TString& filename) {

    ASSERT(this->file->Tell() == 0);

	/*
	pbc 100.0 100.0 100.0
	atom 0:999 radius 0.5 name O type 0

	timestep indexed
	0 88.08974923911063 93.53975290469917 41.0842180843088940
	1 50.542528784672555 69.2565090323587 62.71274455546361
	2 62.747125087753524 49.00074973246766 75.57611795542917
	3 9.46248516175452 43.90389079646931 43.07396560057581
	4 0.32858672109087456 58.02125782527474 64.42774367401746

	pbc X Y Z
	atom from:to radius R name O type 0
	*/

	bool haveBoundingBox = false;
	bool haveAtomType = false;

	this->types.Clear();
	
    vislib::sys::ConsoleProgressBar cpb;
    cpb.Start("Progress Loading VTF File", static_cast<vislib::sys::ConsoleProgressBar::Size>(this->file->GetSize()));

    // read the header
    while (!this->file->IsEOF()) {
		vislib::sys::File::FileSize currentFileCursor = this->file->Tell();
        vislib::StringA line = vislib::sys::ReadLineFromFileA(*this->file);
        line.TrimSpaces();

        if (line.IsEmpty())
            continue;

		vislib::Array<vislib::StringA> shreds = vislib::StringTokeniserA::Split(line, ' ', true);

		if(!haveBoundingBox) {
			if(shreds[0].CompareInsensitive("pbc")){
				extents.Set(vislib::CharTraitsA::ParseDouble(shreds[1]),
					        vislib::CharTraitsA::ParseDouble(shreds[2]),
							vislib::CharTraitsA::ParseDouble(shreds[3]));

				haveBoundingBox = true;
				continue;
			}
		}
		if(!haveAtomType) {
			if(shreds[0].CompareInsensitive("atom")){
				SimpleType type;
				type.SetID(vislib::CharTraitsA::ParseInt(shreds[7]));
				type.SetRadius(vislib::CharTraitsA::ParseDouble(shreds[3]));
				this->types.Append(type);
				haveAtomType = true;
				continue;
			}
		}
	
		if(haveBoundingBox && haveAtomType) {
			if(line.CompareInsensitive("timestep indexed")) {
				this->frameIdx.Append(this->file->Tell());
				cpb.Set(static_cast<vislib::sys::ConsoleProgressBar::Size>(this->file->Tell()));
			}
		}

		/*
        vislib::StringA line = vislib::sys::ReadLineFromFileA(*this->file);
        line.TrimSpaces();

        if (line.IsEmpty()) {
            continue;

        } else if (line[0] == '~') {
            SimpleType *element = NULL;
            try {// a type line!
                int type;
                element = this->parseTypeLine(line, type);
                if (element == NULL) {
                    throw 0;
                }

                for (unsigned int i = 0; i < types.Count(); i++) {
                    if (types[i].ID() == static_cast<unsigned int>(type)) {
                        if (types[i].Radius() < element->Radius()) {
                            types[i].SetRadius(element->Radius());
                        }
                        element = NULL;
                        break;
                    }
                }
                if (element != NULL) {
                    //type = types.Count;
                    types.Append(*element);
                    types.Last() = *element;
                }
                element = NULL;
            } catch(...) {
                this->GetCoreInstance()->Log().WriteMsg(50, "Error parsing type line.");
            }
            SAFE_DELETE(element);
        } else if (line[0] == '>') {
            // very extream file redirection
            vislib::StringW vnfn(vislib::StringA(line.PeekBuffer() + 1));
            vnfn.TrimSpaces();
            vislib::StringW base = vislib::sys::Path::Resolve(vislib::StringW(filename));
            base.Truncate(base.FindLast(vislib::sys::Path::SEPARATOR_W) + 1);
            vnfn = vislib::sys::Path::Resolve(vnfn, base);

            this->file->Close();
            if (!this->file->Open(vnfn, vislib::sys::File::READ_ONLY, 
                    vislib::sys::File::SHARE_READ, 
                    vislib::sys::File::OPEN_ONLY)) {

                SAFE_DELETE(this->file);
                return false;
            }

            this->buildFrameTable();
            break;

        } else if (line[0] == '#') {
            // beginning of the body!
            break;
        }
		*/
		
    }
	this->setFrameCount(this->frameIdx.Count());
	//this->initFrameCache(1);

    return true;
}


/*
 * moldyn::VTFDataSource::getDataCallback
 */
bool moldyn::VTFDataSource::getDataCallback(Call& caller) {
    MultiParticleDataCall *c2 = dynamic_cast<MultiParticleDataCall*>(&caller);

    Frame *f = NULL;
    if (c2 != NULL) {
        f = dynamic_cast<Frame *>(this->requestLockedFrame(c2->FrameID()));
        if (f == NULL) return false;
        c2->SetDataHash((this->file == NULL) ? 0 : this->datahash);
        c2->SetUnlocker(new Unlocker(*f));
        c2->SetParticleListCount(this->types.Count());
        for (unsigned int i = 0; i < this->types.Count(); i++) {
            c2->AccessParticles(i).SetGlobalRadius(this->types[i].Radius()/* / this->boxScaling*/);
            c2->AccessParticles(i).SetGlobalColour(this->types[i].Red(), this->types[i].Green(), this->types[i].Blue());
            c2->AccessParticles(i).SetCount(f->PartCnt(i));
            c2->AccessParticles(i).SetColourData(moldyn::MultiParticleDataCall::Particles::COLDATA_NONE, NULL);
            const float *vd = f->PartPoss(i);
            c2->AccessParticles(i).SetVertexData((vd == NULL)
                ? moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE
                : moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
                vd);
        }
        return true;
    }

    return false;
}


/*
 * moldyn::VTFDataSource::getExtentCallback
 */
bool moldyn::VTFDataSource::getExtentCallback(Call& caller) {
    MultiParticleDataCall *c2 = dynamic_cast<MultiParticleDataCall*>(&caller);
    float border = 0.0f;

    if (c2 != NULL) {
        for (unsigned int i = 0; i < this->types.Count(); i++) {
            float r = this->types[i].Radius();// / this->boxScaling;
            if (r > border) 
				border = r;
        }

        c2->SetDataHash((this->file == NULL) ? 0 : this->datahash);
        c2->SetFrameCount(this->FrameCount());
        c2->AccessBoundingBoxes().Clear();
        c2->AccessBoundingBoxes().SetObjectSpaceBBox(0, 0, 0, this->extents.GetX(), this->extents.GetY(), this->extents.GetZ());
        c2->AccessBoundingBoxes().SetObjectSpaceClipBox(-border, -border, -border,
            this->extents.GetX() + border, this->extents.GetY() + border, this->extents.GetZ() + border);
        return true;
    }

    return false;
}
