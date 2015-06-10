/*
 * Mol20DataSource.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Mol20DataSource.h"
#include "mmcore/vismol2/Mol20DataCall.h"
#include "mmcore/param/StringParam.h"
#include "vislib/sys/Log.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/SystemInformation.h"

using namespace megamol::core;
using namespace vislib::sys;

/*****************************************************************************/


/*
 * vismol2::Mol20DataSource::Frame::Frame
 */
vismol2::Mol20DataSource::Frame::Frame(view::AnimDataModule& owner,
        const int& pfc, const int *const& pf, const int& withQ1)
        : view::AnimDataModule::Frame(owner), data(),
        pointFormatSize(pfc), pointFormat(pf), hasQ1(withQ1) {
    this->data.myTime = UINT_MAX;
    this->data.start.clusters = NULL;
    this->data.start.points = NULL;
    this->data.start.scale = 1.0f;
    this->data.start.size = 0;
}


/*
 * vismol2::Mol20DataSource::Frame::~Frame
 */
vismol2::Mol20DataSource::Frame::~Frame(void) {
    this->clear();
}


/*
 * vismol2::Mol20DataSource::Frame::clear
 */
void vismol2::Mol20DataSource::Frame::clear(void) {
    this->clearCluster(&this->data.start);
}


/*
 * vismol2::Mol20DataSource::Frame::clearCluster
 */
void vismol2::Mol20DataSource::Frame::clearCluster(vismol2::cluster_t *c) {
    int i;
    if (c->clusters) {
        for (i = 0; i < c->size; i++) {
            this->clearCluster(&c->clusters[i]);
        }

        free(c->clusters);  // better use delete?
        c->clusters = NULL;

    }
    if (c->points) {

        free(c->points);  // better use delete?
        c->points = NULL;

    }

    c->size = 0;
}


/*
 * vismol2::Mol20DataSource::Frame::LoadCluster
 */
vislib::sys::File::FileSize vismol2::Mol20DataSource::Frame::LoadCluster(vismol2::cluster_t *c, vismol2::point_t *p, vislib::sys::File& file) {
    vislib::sys::File::FileSize r = 0;
    unsigned char b;
    unsigned long count;
    unsigned long i;

    c->scale = 1.0f;

    r += this->LoadPoint(p, file);
    //Mol20LoadPointBuffered(p);

    r += file.Read(&b, sizeof(unsigned char));
    //b = MOL20LOADBUFFER(unsigned char); mol20LoadBufPos += 1;
    r += file.Read(&count, sizeof(unsigned long));
    c->size = count;
    //c->size = count = MOL20LOADBUFFER(unsigned long); mol20LoadBufPos += 4;

    c->points = static_cast<vismol2::point_t*>(malloc(count * sizeof(point_t)));  // better use new?

    if (b == 1) {

        c->clusters = static_cast<vismol2::cluster_t*>(malloc(count * sizeof(cluster_t)));  // better use new?

        for (i = 0; i < count; i++) {
            r += this->LoadCluster(&c->clusters[i], &c->points[i], file);
    //        Mol20LoadClusterBuffered(&c->clusters[i], &c->points[i]);
        }
    } else if (b == 2) {
        c->clusters = NULL;
        for (i = 0; i < count; i++) {
            r += this->LoadPoint(&c->points[i], file);
    //        Mol20LoadPointBuffered(&c->points[i]);
        }
    } else {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "File format error while reading cluster\n");
        return static_cast<vislib::sys::File::FileSize>(UINT_MAX);
    //    fprintf(stderr, "File format error while reading cluster\n");
    //    return;
    }

    return r;
}


/*
 * vismol2::Mol20DataSource::Frame::LoadPoint
 */
vislib::sys::File::FileSize vismol2::Mol20DataSource::Frame::LoadPoint(vismol2::point_t *p, vislib::sys::File& file) {
    vislib::sys::File::FileSize mol20LoadBufPos = 0;
    int seekoff = 0;

    float tmpDevEll[] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
    unsigned char b;
    int i;
    int interTarget[12];
    const float ellipsoidSizeBonus = 0.0f;

    for (i = 0; i < 12; i++) {
        interTarget[i] = 1;
    }

    if (p) {
        // Init all values of p (constructor work of class point :-P )
        p->x = 0.0f;
        p->y = 0.0f;
        p->z = 0.0f;
        p->s = 1.0f;
        p->id = 0;
        p->semClustID = 0;
        p->r = 255;
        p->g = 0;
        p->b = 0;
        p->a = 255;
        p->q1 = 0.0f;
        p->q2 = 0.0f;
        p->q3 = 0.0f;
        p->q4 = 0.0f;
        p->type = 0;
        p->tpx = 0.0f;
        p->tpy = 0.0f;
        p->tpz = 0.0f;
        p->tsize = 1.0f;
        p->tq1 = 0.0f;
        p->tq2 = 0.0f;
        p->tq3 = 0.0f;
        p->tq4 = 0.0f;
        p->tcr = 255;
        p->tcg = 0;
        p->tcb = 0;
        p->tca = 255;
        p->interpolType = 0;
        p->clusTimeDist = -126;
        p->clusShape = 0;
        memcpy(p->devEllPos, tmpDevEll, 36);
        p->clustGrowth = 0.0f;
    }

    for (i = 0; i < this->pointFormatSize; i++) {
        switch (this->pointFormat[i]) {
            case 0 : 
                if (p) file.Read(&p->x, sizeof(float));
                mol20LoadBufPos += 4;
                break;
            case 1 : 
                if (p) file.Read(&p->y, sizeof(float));
                mol20LoadBufPos += 4;
                break;
            case 2 : 
                if (p) file.Read(&p->z, sizeof(float));
                mol20LoadBufPos += 4;
                break;
            case 3 : 
                if (p) file.Read(&p->s, sizeof(float));
                mol20LoadBufPos += 4;
                break;
            case 4 : 
                if (p) file.Read(&p->id, sizeof(unsigned long));
                mol20LoadBufPos += 4;
                break;
            case 5 : // semClusID
                if (p) file.Read(&p->semClustID, sizeof(int));
                mol20LoadBufPos += 4;
                break;
            case 6 : 
                if (p) file.Read(&p->r, sizeof(unsigned char));
                mol20LoadBufPos += 1;
                break;
            case 7 : 
                if (p) file.Read(&p->g, sizeof(unsigned char));
                mol20LoadBufPos += 1;
                break;
            case 8 : 
                if (p) file.Read(&p->b, sizeof(unsigned char));
                mol20LoadBufPos += 1;
                break;
            case 9 : 
                if (p) file.Read(&p->a, sizeof(unsigned char));
                mol20LoadBufPos += 1;
                break;
            case 10 : 
                if (p) file.Read(&p->q1, sizeof(float));
                mol20LoadBufPos += 4;
                break;
            case 11 : 
                if (p) file.Read(&p->q2, sizeof(float));
                mol20LoadBufPos += 4;
                break;
            case 12 : 
                if (p) file.Read(&p->q3, sizeof(float));
                mol20LoadBufPos += 4;
                break;
            case 13 : 
                if (p) file.Read(&p->q4, sizeof(float));
                mol20LoadBufPos += 4;
                break;
            case 14 : 
                if (p) file.Read(&p->type, sizeof(unsigned short));
                mol20LoadBufPos += 2;
                break;
            case 15 : 
                if (p) file.Read(&p->tpx, sizeof(float));
                mol20LoadBufPos += 4;
                interTarget[0] = 0;
                break;
            case 16 : 
                if (p) file.Read(&p->tpy, sizeof(float));
                mol20LoadBufPos += 4;
                interTarget[1] = 0;
                break;
            case 17 : 
                if (p) file.Read(&p->tpz, sizeof(float));
                mol20LoadBufPos += 4;
                interTarget[2] = 0;
                break;
            case 18 : 
                if (p) file.Read(&p->tsize, sizeof(float));
                mol20LoadBufPos += 4;
                interTarget[3] = 0;
                break;
            case 19 : 
                if (p) file.Read(&p->tq1, sizeof(float));
                mol20LoadBufPos += 4;
                interTarget[4] = 0;
                break;
            case 20 : 
                if (p) file.Read(&p->tq2, sizeof(float));
                mol20LoadBufPos += 4;
                interTarget[5] = 0;
                break;
            case 21 : 
                if (p) file.Read(&p->tq3, sizeof(float));
                mol20LoadBufPos += 4;
                interTarget[6] = 0;
                break;
            case 22 : 
                if (p) file.Read(&p->tq4, sizeof(float));
                mol20LoadBufPos += 4;
                interTarget[7] = 0;
                break;
            case 23 : 
                if (p) file.Read(&p->tcr, sizeof(unsigned char));
                mol20LoadBufPos += 1;
                interTarget[8] = 0;
                break;
            case 24 : 
                if (p) file.Read(&p->tcg, sizeof(unsigned char));
                mol20LoadBufPos += 1;
                interTarget[9] = 0;
                break;
            case 25 : 
                if (p) file.Read(&p->tcb, sizeof(unsigned char));
                mol20LoadBufPos += 1;
                interTarget[10] = 0;
                break;
            case 26 : 
                if (p) file.Read(&p->tca, sizeof(unsigned char));
                mol20LoadBufPos += 1;
                interTarget[11] = 0;
                break;
            case 27 :
                if (p) file.Read(&p->interpolType, sizeof(signed char));
                mol20LoadBufPos += 1;
                break;
            case 28 :
                if (p) file.Read(&p->clusTimeDist, sizeof(signed char));
                mol20LoadBufPos += 1;
                break;
            case 29 :
                file.Read(&b, sizeof(signed char));
                seekoff -= 1;
                mol20LoadBufPos += 1;
                switch (b) {
                case 1 :
                    
                    if (p) file.Read(&p->devEllPos[0], sizeof(float)); mol20LoadBufPos += 4;
                    if (p) file.Read(&p->devEllPos[1], sizeof(float)); mol20LoadBufPos += 4;
                    if (p) file.Read(&p->devEllPos[2], sizeof(float)); mol20LoadBufPos += 4;
                    if (p) file.Read(&p->devEllPos[3], sizeof(float)); mol20LoadBufPos += 4;
                    if (p) file.Read(&p->devEllPos[4], sizeof(float)); mol20LoadBufPos += 4;
                    if (p) file.Read(&p->devEllPos[5], sizeof(float)); mol20LoadBufPos += 4;
                    if (p) file.Read(&p->devEllPos[6], sizeof(float)); mol20LoadBufPos += 4;
                    if (p) file.Read(&p->devEllPos[7], sizeof(float)); mol20LoadBufPos += 4;
                    if (p) file.Read(&p->devEllPos[8], sizeof(float)); mol20LoadBufPos += 4;
/*
                    if (p) p->devEllPos[0] = MOL20LOADBUFFER(float); mol20LoadBufPos += 4;
                    if (p) p->devEllPos[3] = MOL20LOADBUFFER(float); mol20LoadBufPos += 4;
                    if (p) p->devEllPos[6] = MOL20LOADBUFFER(float); mol20LoadBufPos += 4;
                    if (p) p->devEllPos[1] = MOL20LOADBUFFER(float); mol20LoadBufPos += 4;
                    if (p) p->devEllPos[4] = MOL20LOADBUFFER(float); mol20LoadBufPos += 4;
                    if (p) p->devEllPos[7] = MOL20LOADBUFFER(float); mol20LoadBufPos += 4;
                    if (p) p->devEllPos[2] = MOL20LOADBUFFER(float); mol20LoadBufPos += 4;
                    if (p) p->devEllPos[5] = MOL20LOADBUFFER(float); mol20LoadBufPos += 4;
                    if (p) p->devEllPos[8] = MOL20LOADBUFFER(float); mol20LoadBufPos += 4;
*/ /*
                    printf("ElVec1 (%f, %f, %f)\n", p->devEllPos[0], p->devEllPos[1], p->devEllPos[2]);
                    printf("ElVec2 (%f, %f, %f)\n", p->devEllPos[3], p->devEllPos[4], p->devEllPos[5]);
                    printf("ElVec3 (%f, %f, %f)\n", p->devEllPos[6], p->devEllPos[7], p->devEllPos[8]);
*/
                    break;
                }
                break;
            case 30 : 
                file.Read(&b, sizeof(signed char));
                seekoff -= 1;
                mol20LoadBufPos += 1;
                if (p) p->clusShape = b;
                switch (b) {
                case 1 : 
                    if (p) {
                        file.Read(&p->r1, sizeof(float)); mol20LoadBufPos += 4;
                        file.Read(&p->r2, sizeof(float)); mol20LoadBufPos += 4;
                        file.Read(&p->r3, sizeof(float)); mol20LoadBufPos += 4;
                        file.Read(&p->tr1, sizeof(float)); mol20LoadBufPos += 4;
                        file.Read(&p->tr2, sizeof(float)); mol20LoadBufPos += 4;
                        file.Read(&p->tr3, sizeof(float)); mol20LoadBufPos += 4;
                    } else mol20LoadBufPos += 4 * 6;
                    if (!this->hasQ1) {
                        if (p) {
                            file.Read(&p->q1, sizeof(float)); mol20LoadBufPos += 4;
                            file.Read(&p->q2, sizeof(float)); mol20LoadBufPos += 4;
                            file.Read(&p->q3, sizeof(float)); mol20LoadBufPos += 4;
                            file.Read(&p->q4, sizeof(float)); mol20LoadBufPos += 4;
                            file.Read(&p->tq1, sizeof(float)); mol20LoadBufPos += 4;
                            file.Read(&p->tq2, sizeof(float)); mol20LoadBufPos += 4;
                            file.Read(&p->tq3, sizeof(float)); mol20LoadBufPos += 4;
                            file.Read(&p->tq4, sizeof(float)); mol20LoadBufPos += 4;
                            interTarget[4] = 0;
                            interTarget[5] = 0;
                            interTarget[6] = 0;
                            interTarget[7] = 0;
                        } else mol20LoadBufPos += 4 * 8;

                    }

                    if (p) {
                        if (p->r1 < 0.0f) p->r1 = -p->r1;
                        if (p->r1 > 0.0f) p->r1 += ellipsoidSizeBonus;
                        if (p->r2 < 0.0f) p->r2 = -p->r2;
                        if (p->r2 > 0.0f) p->r2 += ellipsoidSizeBonus;
                        if (p->r3 < 0.0f) p->r3 = -p->r3;
                        if (p->r3 > 0.0f) p->r3 += ellipsoidSizeBonus;
                        if (p->tr1 < 0.0f) p->tr1 = -p->tr1;
                        if (p->tr1 > 0.0f) p->tr1 += ellipsoidSizeBonus;
                        if (p->tr2 < 0.0f) p->tr2 = -p->tr2;
                        if (p->tr2 > 0.0f) p->tr2 += ellipsoidSizeBonus;
                        if (p->tr3 < 0.0f) p->tr3 = -p->tr3;
                        if (p->tr3 > 0.0f) p->tr3 += ellipsoidSizeBonus;
                    }

                    break;
                }
                break;
            case 31: if (p) {
                file.Read(&p->clustGrowth, sizeof(float));
                    /* ensure valid range */
                    if (p->clustGrowth > 1.0f) p->clustGrowth = 1.0f;
                    if (p->clustGrowth < -1.0f) p->clustGrowth = -1.0f;
                } 
                mol20LoadBufPos += 4; 
                break;
            default : 
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unsupported point format entry (%d)\n", this->pointFormat[i]);
                return static_cast<vislib::sys::File::FileSize>(UINT_MAX);
                break;
        }
    }

    // fill missing interpolation values
    if (p) {
        for (i = 0; i < 12; i++) {
            if (interTarget[i]) {
                switch (i) {
                    case  0 : p->tpx = p->x; break;
                    case  1 : p->tpy = p->y; break;
                    case  2 : p->tpz = p->z; break;
                    case  3 : p->tsize = p->s; break;
                    case  4 : p->tq1 = p->q1; break;
                    case  5 : p->tq2 = p->q2; break;
                    case  6 : p->tq3 = p->q3; break;
                    case  7 : p->tq4 = p->q4; break;
                    case  8 : p->tcr = p->r; break;
                    case  9 : p->tcg = p->g; break;
                    case 10 : p->tcb = p->b; break;
                    case 11 : p->tca = p->a; break;
                }
            }
        }
    } else {
        file.Seek(mol20LoadBufPos + seekoff, File::CURRENT);
    }

    return mol20LoadBufPos;
}


/*
 * vismol2::Mol20DataSource::Frame::InMemSize
 */
unsigned int vismol2::Mol20DataSource::Frame::InMemSize(void) const {
    return sizeof(vismol2::Mol20DataSource::Frame) + this->InMemSize(this->data.start);
}

extern "C" {

    static int mol20firstPoint;
    static float mol20MinX, mol20MaxX;
    static float mol20MinY, mol20MaxY;
    static float mol20MinZ, mol20MaxZ;

    static void Mol20CalcScalePoint(float x, float y, float z) {
        if (mol20firstPoint) {
            mol20firstPoint = 0;
            mol20MinX = mol20MaxX = x;
            mol20MinY = mol20MaxY = y;
            mol20MinZ = mol20MaxZ = z;
            return;
        }
        if (mol20MinX > x) mol20MinX = x;
        if (mol20MaxX < x) mol20MaxX = x;
        if (mol20MinY > y) mol20MinY = y;
        if (mol20MaxY < y) mol20MaxY = y;
        if (mol20MinZ > z) mol20MinZ = z;
        if (mol20MaxZ < z) mol20MaxZ = z;
    }

    static void Mol20CalcScaleCluster(vismol2::cluster_t *c, float s, float x, float y, float z/*, int *memSize*/) {
        int i;
        //*memSize += sizeof(point_t) * c->size;
        if (c->clusters) {
            //*memSize += sizeof(cluster_t) * c->size;
            for (i = 0; i < c->size; i++) {
                // TODO: NOONE knows if this is correct. The old c-code here is crap!
                Mol20CalcScaleCluster(&c->clusters[i],
                    s * c->scale,
                    (x + c->points[i].x * s),
                    (y + c->points[i].y * s),
                    (z + c->points[i].z * s)/*,
                    memSize*/);
            }
        } else {
            for (i = 0; i < c->size; i++) {
                // TODO: NOONE knows if this is correct. The old c-code here is crap!
                Mol20CalcScalePoint(
                    (x + c->points[i].x * s),
                    (y + c->points[i].y * s),
                    (z + c->points[i].z * s)
                    );
            }
        }
    }

}


/*
 * vismol2::Mol20DataSource::Frame::CalcBBox
 */
void vismol2::Mol20DataSource::Frame::CalcBBox(void) {
    // TODO: Implement
    mol20firstPoint = 1;
    Mol20CalcScaleCluster(&this->data.start, 1.0f, 0.0f, 0.0f, 0.0f);

    this->bbox.Set(mol20MinX, mol20MinY, mol20MinZ, mol20MaxX, mol20MaxY, mol20MaxZ);
}


/*
 * vismol2::Mol20DataSource::Frame::InMemSize
 */
unsigned int vismol2::Mol20DataSource::Frame::InMemSize(const vismol2::cluster_t& c) const {
    unsigned int r = 0;
    if (c.clusters != NULL) {
        for (unsigned int i = 0; i < static_cast<unsigned int>(c.size); i++) {
            r += sizeof(vismol2::cluster_t) + this->InMemSize(c.clusters[i]);
        }
    }
    if (c.points != NULL) r += sizeof(vismol2::point_t) * c.size;
    return r;
}

/*****************************************************************************/


/*
 * vismol2::Mol20DataSource::FluxStore::FluxStore
 */
vismol2::Mol20DataSource::FluxStore::FluxStore(void) {
    // TODO: Implement
}


/*
 * vismol2::Mol20DataSource::FluxStore::~FluxStore
 */
vismol2::Mol20DataSource::FluxStore::~FluxStore(void) {
    this->Clear();
}


/*
 * vismol2::Mol20DataSource::FluxStore::SetFrameCnt
 */
void vismol2::Mol20DataSource::FluxStore::SetFrameCnt(unsigned int cnt) {
    this->Clear();
    // TODO: Implement
}


/*
 * vismol2::Mol20DataSource::FluxStore::Clear
 */
void vismol2::Mol20DataSource::FluxStore::Clear(void) {
    // TODO: Implement
}


/*
 * vismol2::Mol20DataSource::FluxStore::AddFlux
 */
void vismol2::Mol20DataSource::FluxStore::AddFlux(float sx, float sy, float sz, int st,
        float tx, float ty, float tz, int tt, int cnt, float sp, float tp) {
    // TODO: Implement
}

/*****************************************************************************/


/*
 * vismol2::Mol20DataSource::Mol20DataSource
 */
vismol2::Mol20DataSource::Mol20DataSource(void) : view::AnimDataModule(),
        filename("filename", "The path of the mol2.0 file to load"),
        file(NULL),
        getData("getdata", "Slot to request data from this data source.") {

    this->filename << new param::StringParam("");
    this->filename.SetUpdateCallback(this, &Mol20DataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->getData.SetCallback("Mol20DataCall", "GetData",
        &Mol20DataSource::getDataCallback);
    this->getData.SetCallback("Mol20DataCall", "GetExtent",
        &Mol20DataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->setFrameCount(1);
    this->initFrameCache(1);
}


/*
 * vismol2::Mol20DataSource::~Mol20DataSource
 */
vismol2::Mol20DataSource::~Mol20DataSource(void) {
    this->Release(); // will call 'release' as sfx
}


/*
 * vismol2::Mol20DataSource::constructFrame
 */
view::AnimDataModule::Frame* vismol2::Mol20DataSource::constructFrame(void) const {
    return new Frame(*const_cast<vismol2::Mol20DataSource*>(this),
        this->pointFormatSize, this->pointFormat, this->hasQ1);
}


/*
 * vismol2::Mol20DataSource::create
 */
bool vismol2::Mol20DataSource::create(void) {
    this->pointFormat = NULL;
    this->pointFormatSize = 0;
    this->frameTable = NULL;
    this->num_molecules = 0;

    return true;
}


/*
 * vismol2::Mol20DataSource::loadFrame
 */
void vismol2::Mol20DataSource::loadFrame(view::AnimDataModule::Frame *frame, unsigned int idx) {
    Frame *f = dynamic_cast<Frame *>(frame);
    if (f == NULL) return;
    if (this->file == NULL) {
        f->clear();
        f->frame = UINT_MAX;
        return;
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 1000, "Loading Frame %u ...\n", idx);

    if (idx >= this->FrameCount()) idx = this->FrameCount() - 1;

    this->file->Seek(this->frameTable[idx]);

    //const float mol20_x_offset = 0.0f;
    //const float mol20_y_offset = 0.0f;
    //const float mol20_z_offset = 0.0f;
    //const float mol20_scale = 1.0f;

    unsigned char b;
    unsigned long size;
    unsigned long rsize;
    unsigned long count, ci;
    float sx, sy, sz, tx, ty, tz;
    int st, tt, c;
    vislib::sys::File::FileSize p1;
//    __int64 p1;
//#ifdef MEM_ERROR_COMPATIBLE
//    __int64 p2;
//#endif // MEM_ERROR_COMPATIBLE
//    int i;
    float f1, f2;

    //if (dat->myTime != MY_TIME_NO_TIME) Mol2FrameUnloaded(dat->myTime);
    //dat->myTime = MY_TIME_NO_TIME;

    this->file->Read(&size, 4);
    p1 = this->file->Tell();
    //ReadBigFile(&size, 4, mol20FileInfo.fileHandle);
    //p1 = TellBigFile(mol20FileInfo.fileHandle);

    f->clear();
    //Mol20ClearCluster(&dat->start);
    //mol20LoadBuffer = (char *)malloc(size); 
    //if (mol20LoadBuffer) {
        //ReadBigFile(mol20LoadBuffer, size, mol20FileInfo.fileHandle);
        //unsigned long mol20LoadBufPos = 0;

    // load hierarchy data
    vislib::sys::File::FileSize mol20LoadBufPos =
        f->LoadCluster(&f->data.start, NULL, *this->file);
//        Mol20LoadClusterBuffered(&dat->start, 0);

    // load additional data blocks if possible
    while (size > static_cast<unsigned int>(mol20LoadBufPos)) {
        mol20LoadBufPos += this->file->Read(&b, sizeof(unsigned char));
    //    b = MOL20LOADBUFFER(unsigned char); mol20LoadBufPos += 1;

        if (b == 1) { // flux data
            mol20LoadBufPos += this->file->Read(&count, sizeof(unsigned long));
    //        count = MOL20LOADBUFFER(unsigned long); mol20LoadBufPos += 4;
            //printf("Flux Count %d", count);
            for (ci = 0; ci < count; ci++) {
                mol20LoadBufPos += this->file->Read(&sx, sizeof(float)); //sx += mol20_x_offset;
                mol20LoadBufPos += this->file->Read(&sy, sizeof(float)); //sy += mol20_y_offset;
                mol20LoadBufPos += this->file->Read(&sz, sizeof(float)); //sz += mol20_z_offset;
                mol20LoadBufPos += this->file->Read(&st, sizeof(int));
                mol20LoadBufPos += this->file->Read(&tx, sizeof(float)); //tx += mol20_x_offset;
                mol20LoadBufPos += this->file->Read(&ty, sizeof(float)); //ty += mol20_y_offset;
                mol20LoadBufPos += this->file->Read(&tz, sizeof(float)); //tz += mol20_z_offset;
                mol20LoadBufPos += this->file->Read(&tt, sizeof(int));
                mol20LoadBufPos += this->file->Read(&c, sizeof(int));
    //            sx = MOL20LOADBUFFER(float) + mol20_x_offset; mol20LoadBufPos += 4;
    //            sy = MOL20LOADBUFFER(float) + mol20_y_offset; mol20LoadBufPos += 4;
    //            sz = MOL20LOADBUFFER(float) + mol20_z_offset; mol20LoadBufPos += 4;
    //            st = MOL20LOADBUFFER(int); mol20LoadBufPos += 4;
    //            tx = MOL20LOADBUFFER(float) + mol20_x_offset; mol20LoadBufPos += 4;
    //            ty = MOL20LOADBUFFER(float) + mol20_y_offset; mol20LoadBufPos += 4;
    //            tz = MOL20LOADBUFFER(float) + mol20_z_offset; mol20LoadBufPos += 4;
    //            tt = MOL20LOADBUFFER(int); mol20LoadBufPos += 4;
    //            c = MOL20LOADBUFFER(int); mol20LoadBufPos += 4;

                // only add flux data for the current animation direction to reduce data transfer amount
                if (idx == static_cast<unsigned int>(st)) {
    //            if ( ((animDirection ==  1) && (wTime == st))
    //              || ((animDirection == -1) && (wTime + 1 == tt)))
                    this->flux.AddFlux(sx, sy, sz, st, tx, ty, tz, tt, c, 0.0f, 0.0f);
    //                AddMol2Flux(sx, sy, sz, st, tx, ty, tz, tt, c, mol20_scale, 0.0f, 0.0f);
                }
            }
            //printf("...\n");
        } else if (b == 2) { // ext. flux data
            mol20LoadBufPos += this->file->Read(&count, sizeof(unsigned long));
    //        count = MOL20LOADBUFFER(unsigned long); mol20LoadBufPos += 4;
            //printf("Flux Count %d", count);
            for (ci = 0; ci < count; ci++) {
                mol20LoadBufPos += this->file->Read(&sx, sizeof(float)); //sx += mol20_x_offset;
                mol20LoadBufPos += this->file->Read(&sy, sizeof(float)); //sy += mol20_y_offset;
                mol20LoadBufPos += this->file->Read(&sz, sizeof(float)); //sz += mol20_z_offset;
                mol20LoadBufPos += this->file->Read(&st, sizeof(int));
                mol20LoadBufPos += this->file->Read(&tx, sizeof(float)); //tx += mol20_x_offset;
                mol20LoadBufPos += this->file->Read(&ty, sizeof(float)); //ty += mol20_y_offset;
                mol20LoadBufPos += this->file->Read(&tz, sizeof(float)); //tz += mol20_z_offset;
                mol20LoadBufPos += this->file->Read(&tt, sizeof(int));
                mol20LoadBufPos += this->file->Read(&c, sizeof(int));
                mol20LoadBufPos += this->file->Read(&f1, sizeof(float));
                mol20LoadBufPos += this->file->Read(&f2, sizeof(float));
    //            sx = MOL20LOADBUFFER(float) + mol20_x_offset; mol20LoadBufPos += 4;
    //            sy = MOL20LOADBUFFER(float) + mol20_y_offset; mol20LoadBufPos += 4;
    //            sz = MOL20LOADBUFFER(float) + mol20_z_offset; mol20LoadBufPos += 4;
    //            st = MOL20LOADBUFFER(int); mol20LoadBufPos += 4;
    //            tx = MOL20LOADBUFFER(float) + mol20_x_offset; mol20LoadBufPos += 4;
    //            ty = MOL20LOADBUFFER(float) + mol20_y_offset; mol20LoadBufPos += 4;
    //            tz = MOL20LOADBUFFER(float) + mol20_z_offset; mol20LoadBufPos += 4;
    //            tt = MOL20LOADBUFFER(int); mol20LoadBufPos += 4;
    //            c = MOL20LOADBUFFER(int); mol20LoadBufPos += 4;
    //            f1 = MOL20LOADBUFFER(float); mol20LoadBufPos += 4;
    //            f2 = MOL20LOADBUFFER(float); mol20LoadBufPos += 4;

                // only add flux data for the current animation direction to reduce data transfer amount
                if (idx == static_cast<unsigned int>(st)) {
    //            if ( ((animDirection ==  1) && (wTime == st))
    //              || ((animDirection == -1) && (wTime + 1 == tt)))
                    this->flux.AddFlux(sx, sy, sz, st, tx, ty, tz, tt, c, f1, f2);
    //                AddMol2Flux(sx, sy, sz, st, tx, ty, tz, tt, c, mol20_scale, f1, f2);
                }
            }
            //printf("...\n");
        } else {
            // unknown data.
            break;
        }
    }

    //    free(mol20LoadBuffer);
    //} else {
    //    fprintf(stderr, "Failed to allocate loading buffer\nUnable to load time frame\n");
    //}

    this->file->Read(&rsize, 4);
    //ReadBigFile(&rsize, 4, mol20FileInfo.fileHandle);
    if (size != rsize) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "File format error after reading cluster\n");
        f->clear();
        f->frame = UINT_MAX;
        return;
    }

    // we do not scale atm, but we calc the bbox
    f->CalcBBox();

    //Mol20Scaling_Crowbar(&dat->start);
    //for (i = 0; i < dat->start.size; i++) {
    //    dat->start.points[i].x += mol20_x_offset;
    //    dat->start.points[i].y += mol20_y_offset;
    //    dat->start.points[i].z += mol20_z_offset;
    //    dat->start.points[i].tpx += mol20_x_offset;
    //    dat->start.points[i].tpy += mol20_y_offset;
    //    dat->start.points[i].tpz += mol20_z_offset;
    //}

    f->frame = idx;
    //if (setMyTime) dat->myTime = wTime;
    //Mol2FrameLoaded(wTime);
}


/*
 * vismol2::Mol20DataSource::release
 */
void vismol2::Mol20DataSource::release(void) {
    this->resetFrameCache();
    if (this->file != NULL) {
        this->file->Close();
        SAFE_DELETE(this->file);
    }
    if (this->pointFormat != NULL) {
        ARY_SAFE_DELETE(this->pointFormat);
    }
    if (this->frameTable != NULL) {
        ARY_SAFE_DELETE(this->frameTable);
    }
    this->flux.Clear();
}


/*
 * vismol2::Mol20DataSource::filenameChanged
 */
bool vismol2::Mol20DataSource::filenameChanged(param::ParamSlot& slot) {
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Loading \"%s\" ...\n",
        vislib::StringA(this->filename.Param<param::StringParam>()->Value()).PeekBuffer());

    this->resetFrameCache();
    if (this->file != NULL) {
        this->file->Close();
        SAFE_DELETE(this->file);
    }
    this->flux.Clear();

    unsigned char byte;
//    int retVal;
    int newLineFix = 0;
    char idHeader[] = {'B', 'I', 'N', '/', 'm', 'c', 'l', 'd', '0', '0', '2', '\n'};
    char idTest[sizeof(idHeader) + 1];

    assert(sizeof(unsigned int) == 4);
    assert(sizeof(unsigned short) == 2);
    assert(sizeof(unsigned char) == 1);
    assert(POINT_RELATIVE);

    //timeSliderMol2Texture = 0;
    //texCells = 0;
    //getNumberOfFrames = nullGetFrameCount;
    //CloseMol2File = nullCloseMol2File;
    //getLockedPointCloud = nullGetLockedPointCloud;
    //unlockPointCloud = nullUnlockPointCloud;

    vislib::sys::File *fil = new vislib::sys::FastFile();
    if (!fil->Open(this->filename.Param<param::StringParam>()->Value(), vislib::sys::File::READ_ONLY,
            vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        SAFE_DELETE(fil);
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to open file \"%s\"\n",
            vislib::StringA(this->filename.Param<param::StringParam>()->Value()).PeekBuffer());
        return false;
    }

    fil->Read(idTest, sizeof(idHeader) + 1);
    if ((idTest[11] == 0x0D && idTest[12] == 0x0A) /*|| (idTest[11] == 0x0A && idTest[12] == 0x0D)*/) { // dos newline hack
        newLineFix = 1;
        idTest[11] = '\n';
    } else if (idTest[11] == 0x0D) idTest[11] = '\n';
    if (memcmp(idTest, idHeader, sizeof(idHeader)) != 0) {
        SAFE_DELETE(fil);
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "File \"%s\" has invalid file ID\n",
            vislib::StringA(this->filename.Param<param::StringParam>()->Value()).PeekBuffer());
        return false;
    }

    if (newLineFix) fil->Read(&byte, 1);
               else byte = idTest[12];
    if (byte != 0) {
        SAFE_DELETE(fil);
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "File \"%s\" has wrong subformat\n",
            vislib::StringA(this->filename.Param<param::StringParam>()->Value()).PeekBuffer());
        return false;
    }
    //mol2SubFormat = byte;
    //switch (mol2SubFormat) {
    //    case 0 : retVal = Mol20InitFile(filename, hMol2File); break;
    //    case 1 : retVal = Mol21InitFile(filename, hMol2File); break;
    //    default:
    //        fprintf(stderr, "Error: Unsupported Subversion (%d)\n", mol2SubFormat);
    //        retVal = -2;
    //}

    unsigned short shbid;
    float f;
    int i;
//    unsigned int u;
    unsigned long shbCount;
    unsigned short word;
    unsigned long dword;
//    pointcloud_t testCfg;
#ifndef SCALE_DATA_TO_BOX
//    float worldSize[6];
#endif

    //ellipsoidSizeBonus = 0.0f;

    //InitializeCriticalSection(&mol20FileData.criticalSection);

    /* initializing scaling stuff */
    //mol20_x_offset = 0.0f;
    //mol20_y_offset = 0.0f;
    //mol20_z_offset = 0.0f;
    //mol20_scale = 1.0f;

    /* store file handle */
    //mol20FileInfo.fileHandle = file;
    //assert(mol20FileInfo.fileHandle != 0);

    /* rest of the header */
    fil->Read(&shbid, 2);
    if (shbid != 0x1234) {
        SAFE_DELETE(fil);
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Error in File \"%s\": Byte order check failed.\n",
            vislib::StringA(this->filename.Param<param::StringParam>()->Value()).PeekBuffer());
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
            "Functions to byte order switching are not implemented.\n");
        return false;
    }

    /* number of configurations */
    fil->Read(&shbCount, 4);
    if (this->frameTable != NULL) {
        ARY_SAFE_DELETE(this->frameTable);
    }
    this->setFrameCount(shbCount);
    this->flux.SetFrameCnt(shbCount);
    //ReadBigFile(&shbCount, 4, mol20FileInfo.fileHandle);
    //mol20FileInfo.frameCount = shbCount;
    //printf("%d time frames\n", mol20FileInfo.frameCount);

    /* set framecount callback functions */
    //getNumberOfFrames = Mol20FrameCount;
    //Mol2InitFrameLoadBuffer();

    /* load configurations point format */
    fil->Read(&shbid, 2);
    this->pointFormatSize = shbid;
    if (this->pointFormat != NULL) {
        delete[] this->pointFormat;
    }
    this->pointFormat = new int[this->pointFormatSize];
    this->hasQ1 = 0;
    for (i = 0; i < this->pointFormatSize; i++) {
        fil->Read(&shbid, 2);
        this->pointFormat[i] = shbid;
        if (this->pointFormat[i] == 10) this->hasQ1 = 1;
    }

    /* sub header blocks for molecule type definitions */
    this->num_molecules = 0;
    fil->Read(&shbCount, 4);
    for (i = 0; i < (int)shbCount; i++) {
        fil->Read(&shbid, 2);
        switch (shbid) {
            case 1 : 
                /* molecule type */
                fil->Read(&dword, 4);
                if (dword != 22) {
                    SAFE_DELETE(fil);
                    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "Error in File \"%s\": Generic file format error.\n",
                        vislib::StringA(this->filename.Param<param::StringParam>()->Value()).PeekBuffer());
                    return false;
                }
                fil->Read(&word, 2);
                if (word >= NUM_MOLECULE_TYPES) {
                    SAFE_DELETE(fil);
                    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "Error in File \"%s\": Kabooom!\nUnable to handle molecule ids greater than or equal %d.\n",
                        vislib::StringA(this->filename.Param<param::StringParam>()->Value()).PeekBuffer(), NUM_MOLECULE_TYPES);
                    return false;
                }
                fil->Read(&f, 4);
                this->molecules[word].dist = f;
                fil->Read(&f, 4);
                this->molecules[word].radius1 = f;
                fil->Read(&f, 4);
                this->molecules[word].radius2 = f;
                fil->Read(&f, 4);
                this->molecules[word].radius3 = f;
                fil->Read(&f, 4);
                this->molecules[word].cyllength = f;
                this->molecules[word].colindex = 0;

                if (this->num_molecules < word + 1) this->num_molecules = word + 1;

                Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                    "Molecule %u: dist %f, rad1 %f, rad2 %f, rad3 %f, cyllength %f, colindex %f\n",
                    word, this->molecules[word].dist, this->molecules[word].radius1, this->molecules[word].radius2,
                    this->molecules[word].radius3, this->molecules[word].cyllength, this->molecules[word].colindex);

                break;
            default: 
                SAFE_DELETE(fil);
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Error in File \"%s\": Unsupported subheader block found.\n",
                    vislib::StringA(this->filename.Param<param::StringParam>()->Value()).PeekBuffer());
                return false;
        }
    }

    /* store seeking positions */
    this->frameTable = new vislib::sys::File::FileSize[this->FrameCount()];
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Prewarming of file started ...\n");
//    printf("\n");
    for (i = 0; i < static_cast<int>(this->FrameCount()); i++) {
//        printf("\rPrewarming file: %3d%%", (int)(i * 100 / mol20FileInfo.frameCount));
        this->frameTable[i] = fil->Tell();
        fil->Read(&dword, 4);
        fil->Seek(dword, vislib::sys::File::CURRENT);
        fil->Read(&shbCount, 4);
        if (shbCount != dword) {
            SAFE_DELETE(fil);
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Error in File \"%s\": Seeking configuration %d failed.\n",
                vislib::StringA(this->filename.Param<param::StringParam>()->Value()).PeekBuffer(), i);
            return false;
        }
    }
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "File is now warm!\n");
//    printf("\rPrewarming file: 100%%\n");

    this->file = fil;
    fil = NULL;
    Frame tmpFrame(*this, this->pointFormatSize, this->pointFormat, this->hasQ1);
//    tmpFrame.SetTypeCount(this->typeCnt);
    // use frame zero to estimate the frame size in memory to calculate the
    // frame cache size
    this->loadFrame(&tmpFrame, 0);
    unsigned int ffsize = static_cast<unsigned int>(
        static_cast<double>(tmpFrame.InMemSize()) * 1.15);
    this->bbox = tmpFrame.bbox;

    this->loadFrame(&tmpFrame, this->FrameCount() - 1);
    unsigned int lfsize = static_cast<unsigned int>(
        static_cast<double>(tmpFrame.InMemSize()) * 1.15);
    this->bbox.Union(tmpFrame.bbox);

    int framz = static_cast<int>(
        vislib::sys::SystemInformation::AvailableMemorySize()
        / vislib::math::Max(ffsize, lfsize));

    if (framz < 4) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
            "Short on Memory!");
        framz = 4;
    }
    if (framz > static_cast<int>(this->FrameCount())) {
        framz = static_cast<int>(this->FrameCount());
    }
    if (framz == static_cast<int>(this->FrameCount())) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "Enough Memory to load whole data set.");
    }

//    /* loading first frame for calculation stuff */
//    SeekBigFile(mol20FileInfo.fileHandle, mol20FileInfo.frameTable[0], SEEKBIGFILE_BEGIN);
//    /* init testCfg or else the hole crap will explode */
//    testCfg.myTime = MY_TIME_NO_TIME;
//    testCfg.start.clusters= 0;
//    testCfg.start.points = 0;
//    testCfg.start.size = 0;
//    testCfg.start.scale = 1.0f;
//    testCfg.genpointsize = 1.0f;
//    testCfg.maxreclevel = 20;
//    testCfg.delta_n = 1.0f;
//    /* load the configuration */
//    Mol20LoadFrame(&testCfg, 0, 1, 0);
//    /* calculate spatial size and memory footprint */
//    Mol20CalculateScalings(&testCfg, &i);
//    dword = i;
//    printf("Memory Footprint of a single configuration is %u Byte.\n", dword);
//
//    ellipsoidSizeBonus = 0.0f;
//    /* calculate spatial scalings */
//    for (i = 0; i < num_molecules; i++) {
//        molecules[i].cyllength *= mol20_scale;
//        molecules[i].dist *= mol20_scale;
//        molecules[i].radius1 *= mol20_scale;
//        molecules[i].radius2 *= mol20_scale;
//        molecules[i].radius3 *= mol20_scale;
//
//        ellipsoidSizeBonus += max(max(molecules[i].radius1, molecules[i].radius2), molecules[i].radius3);
//    }
//    ellipsoidSizeBonus /= (float)num_molecules;
//    printf("Pssst. This is a secret! \"EBS = %f\"! But don't tell anyone!\n", ellipsoidSizeBonus);
//    calc_mol_size_stuff();
//
//    /* calculate number of configurations that would fit into memory */
//    dword = MOL2_MAX_MEMORY_USAGE / dword;
//    if (dword < 4) {
//        fprintf(stderr, "Warning: possibly memory problem detected.\nOnly %u configurations would fit into memory.\n", dword);
//        dword = 4;
//    }
//    if (dword > (unsigned int)mol20FileInfo.frameCount) {
//        dword = mol20FileInfo.frameCount;
//        printf("Memory sufficient to store whole trajectory.\n");
//    }
//    printf("Loading %u configurations into memory.\n", dword);
//    mol20FileData.cfgCount = dword;
//
//    mol20FileData.cfgs = malloc(sizeof(pointcloud_t) * mol20FileData.cfgCount);
//    for (u = 0; u < mol20FileData.cfgCount; u++) {
//        mol20FileData.cfgs[u].myTime = MY_TIME_NO_TIME;
//        mol20FileData.cfgs[u].start.clusters= 0;
//        mol20FileData.cfgs[u].start.points = 0;
//        mol20FileData.cfgs[u].start.size = 0;
//        mol20FileData.cfgs[u].start.scale = 1.0f;
//        mol20FileData.cfgs[u].genpointsize = 1.0f;
//        mol20FileData.cfgs[u].maxreclevel = 20;
//        mol20FileData.cfgs[u].delta_n = 1.0f;
//    }
//
//    mol20FileData.voidCloud.myTime = MY_TIME_NO_TIME;
//    mol20FileData.voidCloud.start.clusters= 0;
//    mol20FileData.voidCloud.start.points = 0;
//    mol20FileData.voidCloud.start.size = 0;
//    mol20FileData.voidCloud.start.scale = 1.0f;
//    mol20FileData.voidCloud.genpointsize = 1.0f;
//    mol20FileData.voidCloud.maxreclevel = 20;
//    mol20FileData.voidCloud.delta_n = 1.0f;
//
//    /* set further callback functions*/
//    CloseMol2File = Mol20CloseFile;
//    getLockedPointCloud = Mol20GetLockedPointCloud;
//    unlockPointCloud = Mol20UnlockPointCloud;	
//
//    /* calc the size of the world */
//    SeekBigFile(mol20FileInfo.fileHandle, mol20FileInfo.frameTable[0], SEEKBIGFILE_BEGIN);
//    Mol20LoadFrame(&testCfg, 0, 1, 0);
//    SeekBigFile(mol20FileInfo.fileHandle, mol20FileInfo.frameTable[mol20FileInfo.frameCount - 1], SEEKBIGFILE_BEGIN);
//    Mol20LoadFrame(&testCfg, 0, 1, 0);
//#ifndef SCALE_DATA_TO_BOX
//    worldSize[0] = minX;
//    worldSize[1] = maxX;
//    worldSize[2] = minY;
//    worldSize[3] = maxY;
//    worldSize[4] = minZ;
//    worldSize[5] = maxZ;
//    Mol20CalculateScalings(&testCfg, 0);
//    minX = min(minX, worldSize[0]);
//    maxX = max(maxX, worldSize[1]);
//    minY = min(minY, worldSize[2]);
//    maxY = max(maxY, worldSize[3]);
//    minZ = min(minZ, worldSize[4]);
//    maxZ = max(maxZ, worldSize[5]);
//#endif

    this->initFrameCache(static_cast<unsigned int>(framz));

    return true; // to reset the dirty flag
}


/*
 * vismol2::Mol20DataSource::getDataCallback
 */
bool vismol2::Mol20DataSource::getDataCallback(Call& caller) {
    vismol2::Mol20DataCall *call = dynamic_cast<vismol2::Mol20DataCall*>(&caller);
    if (call == NULL) return false;

    Frame *f = dynamic_cast<Frame *>(this->requestLockedFrame(call->Time()));
    if (f == NULL) return false;

    call->SetFrameData(f);
    call->SetTime(f->FrameNumber());
    // call->SetBoundingBox(f->bbox); // not required

    return true;
}


/*
 * vismol2::Mol20DataSource::getExtentCallback
 */
bool vismol2::Mol20DataSource::getExtentCallback(Call& caller) {
    vismol2::Mol20DataCall *call = dynamic_cast<vismol2::Mol20DataCall*>(&caller);
    if (call == NULL) return false;

    call->SetBoundingBox(this->bbox);
    call->SetTime(this->FrameCount());

    return true;
}
