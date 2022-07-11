/*
 * WavefrontObjDataSource.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "WavefrontObjDataSource.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/Array.h"
#include "vislib/PtrArray.h"
#include "vislib/StringConverter.h"
#include "vislib/assert.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/math/Vector.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/sys/File.h"
#include "vislib/sys/MemmappedFile.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/PerformanceCounter.h"
#include <cfloat>
#include <map>

using namespace megamol;
using namespace megamol::trisoup_gl;


/*
 * WavefrontObjDataSource::WavefrontObjDataSource
 */
WavefrontObjDataSource::WavefrontObjDataSource(void) : AbstractTriMeshLoader() {
    lineVerts.AssertCapacity(1000);
    lineVerts.SetCapacityIncrement(1000);
}


/*
 * WavefrontObjDataSource::~WavefrontObjDataSource
 */
WavefrontObjDataSource::~WavefrontObjDataSource(void) {
    this->Release();
}


/*
 * WavefrontObjDataSource::load
 */
bool WavefrontObjDataSource::load(const vislib::TString& filename) {
    using megamol::core::utility::log::Log;

    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteInfo( "Filename is empty");
        return true;
    }
    if (!vislib::sys::File::Exists(filename)) {
        Log::DefaultLog.WriteError( "Specified file does not exist");
        return false;
    }

    vislib::sys::ASCIIFileBuffer linesA(vislib::sys::ASCIIFileBuffer::PARSING_WORDS);
    if (!linesA.LoadFile(filename)) {
        Log::DefaultLog.WriteError( "Unable to load file");
        return false;
    }

    this->objs.Clear();
    this->mats.Clear();
    this->lines.clear();

    Log::DefaultLog.WriteInfo( "Start loading \"%s\"\n", vislib::StringA(filename).PeekBuffer());
    double startTime = vislib::sys::PerformanceCounter::QueryMillis();

    vislib::TString path = vislib::sys::Path::GetDirectoryName(filename);

    float vec3[3];
    vislib::math::ShallowVector<float, 3> vec3v3(vec3);
    vislib::math::ShallowVector<float, 2> vec3v2(vec3);
    vislib::Array<vislib::math::Vector<float, 3>> vert;
    vislib::Array<vislib::math::Vector<float, 3>> norm;
    vislib::Array<vislib::math::Vector<float, 2>> texc;
    std::map<size_t, size_t> lineID2Idx;
    vislib::Array<vislib::StringA> objsMats;
    vislib::PtrArray<vislib::Array<Tri>> objs;
    vislib::Array<Tri>* obj = NULL;
    vislib::Array<vislib::StringA> matNames;

    const SIZE_T capacityGrowth = 10 * 1024;

    for (SIZE_T li = 0; li < linesA.Count(); li++) {
        const vislib::sys::ASCIIFileBuffer::LineBuffer& line = linesA[li];
        if (line.Count() <= 0)
            continue;

        try {

            if ((::strcmp(line.Word(0), "mtllib") == 0) && (line.Count() >= 2)) {
                // load material library
                this->loadMaterialLibrary(vislib::sys::Path::Concatenate(path, A2T(line.Word(1))), matNames);

            } else if ((::strcmp(line.Word(0), "usemtl") == 0) && (line.Count() >= 2)) {
                // use material (new group)

                if (obj == NULL) {
                    objs.Append(new vislib::Array<Tri>());
                    objsMats.Append(line.Word(1));
                    obj = objs.Last();

                } else if (objsMats.Last().IsEmpty()) {
                    objsMats.Last() = line.Word(1);

                } else if (!objsMats.Last().Equals(line.Word(1))) {
                    if (obj->Count() > 0) {
                        objs.Append(new vislib::Array<Tri>());
                        objsMats.Append(line.Word(1));
                        obj = objs.Last();

                    } else {
                        objsMats.Last() = line.Word(1);
                    }
                }

            } else if ((::strcmp(line.Word(0), "v") == 0) && (line.Count() >= 4)) {
                // vertex
                vec3[0] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(1)));
                vec3[1] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(2)));
                vec3[2] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(3)));
                if (vert.Count() == vert.Capacity())
                    vert.AssertCapacity(vert.Capacity() + capacityGrowth);
                vert.Add(vec3v3);

            } else if ((::strcmp(line.Word(0), "vn") == 0) && (line.Count() >= 4)) {
                // vertex normal
                vec3[0] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(1)));
                vec3[1] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(2)));
                vec3[2] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(3)));
                if (norm.Count() == norm.Capacity())
                    norm.AssertCapacity(norm.Capacity() + capacityGrowth);
                norm.Add(vec3v3);

            } else if ((::strcmp(line.Word(0), "vt") == 0) && (line.Count() >= 2)) {
                // vertex texture coordinate
                vec3[0] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(1)));
                vec3[1] =
                    (line.Count() >= 3) ? static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(2))) : 0.0f;
                if (texc.Count() == texc.Capacity())
                    texc.AssertCapacity(texc.Capacity() + capacityGrowth);
                texc.Add(vec3v2);

            } else if ((::strcmp(line.Word(0), "f") == 0) && (line.Count() >= 4)) {
                // face
                if (obj == NULL) {
                    objs.Append(new vislib::Array<Tri>());
                    objsMats.Append(vislib::StringA::EMPTY);
                    obj = objs.Last();
                }

                Tri t;
                const char* p1 = ::strchr(line.Word(1), '/');
                int idx;

                if (p1 == NULL) {
                    t.n = t.t = false;
                    t.n1 = t.n2 = t.n3 = 0;
                    t.t1 = t.t2 = t.t3 = 0;
                    idx = vislib::CharTraitsA::ParseInt(line.Word(1));
                    if (idx <= 0)
                        throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                    t.v1 = static_cast<unsigned int>(idx - 1);
                    idx = vislib::CharTraitsA::ParseInt(line.Word(2));
                    if (idx <= 0)
                        throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                    t.v2 = static_cast<unsigned int>(idx - 1);
                    for (unsigned int i = 3; i < line.Count(); i++) {
                        idx = vislib::CharTraitsA::ParseInt(line.Word(i));
                        if (idx <= 0)
                            throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                        t.v3 = static_cast<unsigned int>(idx - 1);
                        obj->Append(t);
                        t.v2 = t.v3;
                    }

                } else {
                    SIZE_T len = vislib::CharTraitsA::SafeStringLength(line.Word(1));
                    const char* p2 = ::strchr(p1 + 1, '/');
                    if (p2 == NULL)
                        throw vislib::Exception("Single slash face entry element illegal", __FILE__, __LINE__);
                    t.t = (p2 > p1 + 1);
                    t.n = (p2 + 1 < line.Word(1) + len);
                    t.n1 = t.n2 = t.n3 = 0;
                    t.t1 = t.t2 = t.t3 = 0;
                    *const_cast<char*>(p1) = '\0';
                    *const_cast<char*>(p2) = '\0';
                    p1++;
                    p2++;
                    idx = vislib::CharTraitsA::ParseInt(line.Word(1));
                    if (idx <= 0)
                        throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                    t.v1 = static_cast<unsigned int>(idx - 1);
                    if (t.t) {
                        idx = vislib::CharTraitsA::ParseInt(p1);
                        if (idx <= 0)
                            throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                        t.t1 = static_cast<unsigned int>(idx - 1);
                    }
                    if (t.n) {
                        idx = vislib::CharTraitsA::ParseInt(p2);
                        if (idx <= 0)
                            throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                        t.n1 = static_cast<unsigned int>(idx - 1);
                    }

                    len = vislib::CharTraitsA::SafeStringLength(line.Word(2));
                    p1 = ::strchr(line.Word(2), '/');
                    if (p1 == NULL)
                        throw vislib::Exception("Single slash face entry element illegal", __FILE__, __LINE__);
                    p2 = ::strchr(p1 + 1, '/');
                    if (p2 == NULL)
                        throw vislib::Exception("Single slash face entry element illegal", __FILE__, __LINE__);
                    if (t.t != (p2 > p1 + 1))
                        throw vislib::Exception("face entry inconsistancy", __FILE__, __LINE__);
                    if (t.n != (p2 + 1 < line.Word(2) + len))
                        throw vislib::Exception("face entry inconsistancy", __FILE__, __LINE__);
                    *const_cast<char*>(p1) = '\0';
                    *const_cast<char*>(p2) = '\0';
                    p1++;
                    p2++;
                    idx = vislib::CharTraitsA::ParseInt(line.Word(2));
                    if (idx <= 0)
                        throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                    t.v2 = static_cast<unsigned int>(idx - 1);
                    if (t.t) {
                        idx = vislib::CharTraitsA::ParseInt(p1);
                        if (idx <= 0)
                            throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                        t.t2 = static_cast<unsigned int>(idx - 1);
                    }
                    if (t.n) {
                        idx = vislib::CharTraitsA::ParseInt(p2);
                        if (idx <= 0)
                            throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                        t.n2 = static_cast<unsigned int>(idx - 1);
                    }

                    t.v2 = static_cast<unsigned int>(vislib::CharTraitsA::ParseInt(line.Word(2)) - 1);
                    if (t.t)
                        t.t2 = static_cast<unsigned int>(vislib::CharTraitsA::ParseInt(p1) - 1);
                    if (t.n)
                        t.n2 = static_cast<unsigned int>(vislib::CharTraitsA::ParseInt(p2) - 1);

                    for (unsigned int i = 3; i < line.Count(); i++) {
                        len = vislib::CharTraitsA::SafeStringLength(line.Word(i));
                        p1 = ::strchr(line.Word(i), '/');
                        if (p1 == NULL)
                            throw vislib::Exception("Single slash face entry element illegal", __FILE__, __LINE__);
                        p2 = ::strchr(p1 + 1, '/');
                        if (p2 == NULL)
                            throw vislib::Exception("Single slash face entry element illegal", __FILE__, __LINE__);
                        if (t.t != (p2 > p1 + 1))
                            throw vislib::Exception("face entry inconsistancy", __FILE__, __LINE__);
                        if (t.n != (p2 + 1 < line.Word(i) + len))
                            throw vislib::Exception("face entry inconsistancy", __FILE__, __LINE__);
                        *const_cast<char*>(p1) = '\0';
                        *const_cast<char*>(p2) = '\0';
                        p1++;
                        p2++;
                        idx = vislib::CharTraitsA::ParseInt(line.Word(i));
                        if (idx <= 0)
                            throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                        t.v3 = static_cast<unsigned int>(idx - 1);
                        if (t.t) {
                            idx = vislib::CharTraitsA::ParseInt(p1);
                            if (idx <= 0)
                                throw vislib::Exception(
                                    "Negative face element indices not supported", __FILE__, __LINE__);
                            t.t3 = static_cast<unsigned int>(idx - 1);
                        }
                        if (t.n) {
                            idx = vislib::CharTraitsA::ParseInt(p2);
                            if (idx <= 0)
                                throw vislib::Exception(
                                    "Negative face element indices not supported", __FILE__, __LINE__);
                            t.n3 = static_cast<unsigned int>(idx - 1);
                        }
                        obj->Append(t);
                        t.v2 = t.v3;
                        t.t2 = t.t3;
                        t.n2 = t.n3;
                    }
                }

            } else if (::strcmp(line.Word(0), "g") == 0) {
                // new group
                if ((obj == NULL) || (obj->Count() > 0)) {
                    objs.Append(new vislib::Array<Tri>());
                    objsMats.Append(vislib::StringA::EMPTY); // or should we keep the material? spec does not tell!
                    obj = objs.Last();
                }
            } else if (::strcmp(line.Word(0), "l") == 0) {
                int idxS = vislib::CharTraitsA::ParseInt(line.Word(1));
                int idxT = vislib::CharTraitsA::ParseInt(line.Word(2));
                size_t ID = 0;
                size_t listIdx = 0;
                if (line.Count() == 4) {
                    ID = static_cast<size_t>(vislib::CharTraitsA::ParseUInt64(line.Word(3)));
                    if (lineID2Idx.find(ID) == lineID2Idx.end()) {
                        size_t oldSize = lineID2Idx.size();
                        lineID2Idx[ID] = oldSize;
                        lineVerts.SetCount(lineID2Idx[ID] + 1);
                        lineVerts[lineVerts.Count() - 1].AssertCapacity(1000);
                        lineVerts[lineVerts.Count() - 1].SetCapacityIncrement(1000);
                        // new line
                        Lines lineData;
                        lineData.SetID(ID);
                        this->lines.push_back(lineData);
                    }
                    listIdx = lineID2Idx[ID];
                } else {
                    listIdx = 0;
                    if (lineVerts.Count() == 0) {
                        lineVerts.SetCount(1);
                        lineVerts[lineVerts.Count() - 1].AssertCapacity(1000);
                        lineVerts[lineVerts.Count() - 1].SetCapacityIncrement(1000);
                        Lines lineData;
                        lineData.SetID(0);
                        this->lines.push_back(lineData);
                    }
                }
                float* vertices = new float[6];
                auto s = vert[idxS - 1];
                auto t = vert[idxT - 1];
                lineVerts[listIdx].Append(s.X());
                lineVerts[listIdx].Append(s.Y());
                lineVerts[listIdx].Append(s.Z());
                lineVerts[listIdx].Append(t.X());
                lineVerts[listIdx].Append(t.Y());
                lineVerts[listIdx].Append(t.Z());
                //lineData.Set(2, vertices, vislib::graphics::ColourRGBAu8(255, 255, 255, 255));
                //this->lines.push_back(lineData);
            }

        } catch (vislib::Exception ex) {
            Log::DefaultLog.WriteError( "Error parsing line %u: %s (%s, %d)", li, ex.GetMsgA(), ex.GetFile(), ex.GetLine());
        } catch (...) { Log::DefaultLog.WriteError( "Error parsing line %u: unexpected exception", li); }
    }
    if (lineVerts.Count() > 0) {
        for (size_t loop = 0; loop < lineVerts.Count(); loop++) {
            lines[loop].Set(static_cast<unsigned int>(lineVerts[loop].Count() / 3), lineVerts[loop].PeekElements(),
                vislib::graphics::ColourRGBAu8(255, 255, 255, 255));
        }
    }

    double parseTime = (vislib::sys::PerformanceCounter::QueryMillis() - startTime) * 0.001;
    Log::DefaultLog.WriteInfo( "Parsing file completed after %f seconds\n", parseTime);

    if (!this->lines.empty()) {
        float minV[] = {FLT_MAX, FLT_MAX, FLT_MAX};
        float maxV[] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

        for (size_t i = 0; i < vert.Count(); i++) {
            if (minV[0] > vert[i].X())
                minV[0] = vert[i].X();
            if (maxV[0] < vert[i].X())
                maxV[0] = vert[i].X();

            if (minV[1] > vert[i].Y())
                minV[1] = vert[i].Y();
            if (maxV[1] < vert[i].Y())
                maxV[1] = vert[i].Y();

            if (minV[2] > vert[i].Z())
                minV[2] = vert[i].Z();
            if (maxV[2] < vert[i].Z())
                maxV[2] = vert[i].Z();
        }

        this->bbox.Set(minV[0], minV[1], minV[2], maxV[0], maxV[1], maxV[2]);
    } else {

        unsigned int* vertUsed = new unsigned int[vert.Count()];

        unsigned int oc = 0;
        for (SIZE_T i = 0; i < objs.Count(); i++) {
            if (objs[i]->Count() > 0)
                oc++;
        }
        this->objs.SetCount(oc);
        this->objs.Trim();
        if (oc == 0)
            this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        else
            this->bbox.Set(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        oc = 0;
        for (SIZE_T i = 0; i < objs.Count(); i++) {
            if (objs[i]->Count() <= 0)
                continue;
            INT_PTR matIdx = matNames.IndexOf(objsMats[i]);
            this->objs[oc].SetMaterial(((matIdx == vislib::Array<vislib::StringA>::INVALID_POS) ||
                                           (static_cast<SIZE_T>(matIdx) >= this->mats.Count()))
                                           ? NULL
                                           : &this->mats[static_cast<SIZE_T>(matIdx)]);
            this->makeMesh(this->objs[oc], *objs[i], vertUsed, vert, norm, texc);
            oc++;
        }

        delete[] vertUsed;
    }

    return true;
}


/*
 * WavefrontObjDataSource::loadMaterialLibrary
 */
void WavefrontObjDataSource::loadMaterialLibrary(
    const vislib::TString& filename, vislib::Array<vislib::StringA>& names) {
    using megamol::core::utility::log::Log;
    ASSERT(names.Count() == this->mats.Count());

    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteInfo( "Material library filename is empty");
        return;
    }
    if (!vislib::sys::File::Exists(filename)) {
        Log::DefaultLog.WriteError( "Material library \"%s\" file does not exist", vislib::StringA(filename).PeekBuffer());
        return;
    }
    vislib::sys::ASCIIFileBuffer linesA(vislib::sys::ASCIIFileBuffer::PARSING_WORDS);
    if (!linesA.LoadFile(filename)) {
        Log::DefaultLog.WriteError( "Unable to load material library \"%s\"", vislib::StringA(filename).PeekBuffer());
        return;
    }
    Log::DefaultLog.WriteInfo( "Loading material library \"%s\"", vislib::StringA(filename).PeekBuffer());

    vislib::TString path = vislib::sys::Path::GetDirectoryName(filename);
    Material* mat = NULL;

    for (SIZE_T li = 0; li < linesA.Count(); li++) {
        const vislib::sys::ASCIIFileBuffer::LineBuffer& line = linesA[li];
        if (line.Count() <= 0)
            continue;

        try {

            if ((strcmp(line.Word(0), "newmtl") == 0) && (line.Count() >= 2)) {
                INT_PTR idx = names.IndexOf(line.Word(1));
                if (idx == vislib::Array<vislib::StringA>::INVALID_POS) {
                    names.Append(line.Word(1));
                    this->mats.Append(Material());
                    mat = &this->mats.Last();
                } else {
                    mat = &this->mats[static_cast<SIZE_T>(idx)];
                }

            } else if (mat == NULL) {
                // ignoring line when there is no active material

            } else if ((strcmp(line.Word(0), "Ns") == 0) && (line.Count() >= 2)) {
                mat->SetNs(static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(1))));

            } else if ((strcmp(line.Word(0), "Ni") == 0) && (line.Count() >= 2)) {
                mat->SetNi(static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(1))));

            } else if ((strcmp(line.Word(0), "d") == 0) && (line.Count() >= 2)) {
                mat->SetD(static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(1))));

            } else if ((strcmp(line.Word(0), "Ka") == 0) && (line.Count() >= 4)) {
                mat->SetKa(static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(1))),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(2))),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(3))));

            } else if ((strcmp(line.Word(0), "Kd") == 0) && (line.Count() >= 4)) {
                mat->SetKd(static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(1))),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(2))),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(3))));

            } else if ((strcmp(line.Word(0), "Ks") == 0) && (line.Count() >= 4)) {
                mat->SetKs(static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(1))),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(2))),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(3))));

            } else if ((strcmp(line.Word(0), "Ke") == 0) && (line.Count() >= 4)) {
                mat->SetKe(static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(1))),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(2))),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line.Word(3))));

            } else if (((strcmp(line.Word(0), "bump") == 0) || (strcmp(line.Word(0), "map_bump") == 0) ||
                           (strcmp(line.Word(0), "bump_map") == 0)) &&
                       (line.Count() >= 2)) {
                mat->SetBumpMapFileName(vislib::sys::Path::Concatenate(path, line.Word(1)));

            } else if ((strncmp(line.Word(0), "map", 3) == 0) && (line.Count() >= 2)) {
                mat->SetMapFileName(vislib::sys::Path::Concatenate(path, line.Word(1)));
            }

        } catch (vislib::Exception ex) {
            Log::DefaultLog.WriteError( "Error parsing line %u: %s (%s, %d)", li, ex.GetMsgA(), ex.GetFile(), ex.GetLine());
        } catch (...) { Log::DefaultLog.WriteError( "Error parsing line %u: unexpected exception", li); }
    }

    ASSERT(names.Count() == this->mats.Count());
}


/*
 * WavefrontObjDataSource::makeMesh
 */
void WavefrontObjDataSource::makeMesh(megamol::geocalls_gl::CallTriMeshDataGL::Mesh& mesh,
    const vislib::Array<WavefrontObjDataSource::Tri>& tris, unsigned int* vu,
    const vislib::Array<vislib::math::Vector<float, 3>>& v, const vislib::Array<vislib::math::Vector<float, 3>>& n,
    const vislib::Array<vislib::math::Vector<float, 2>>& t) {
    ASSERT(tris.Count() > 0);
    ASSERT(v.Count() > 0);

    float* vd = new float[tris.Count() * 3 * 3];                      // vertices
    float* nd = (tris[0].n) ? new float[tris.Count() * 3 * 3] : NULL; // normals
    float* td = (tris[0].t) ? new float[tris.Count() * 3 * 2] : NULL; // texture coordinates
    unsigned int* fd =
        new unsigned int[tris.Count() *
                         3]; // faces. actually just 0,1,2,3,4... since everything is multiplied out already
    vislib::math::Cuboid<float> bbox(v[0].X(), v[0].Y(), v[0].Z(), v[0].X(), v[0].Y(), v[0].Z());

    for (SIZE_T ti = 0; ti < tris.Count(); ti++) {
        const Tri& tri = tris[ti];
        vd[ti * 9 + 0 + 0] = v[tri.v1].X();
        vd[ti * 9 + 0 + 1] = v[tri.v1].Y();
        vd[ti * 9 + 0 + 2] = v[tri.v1].Z();
        bbox.GrowToPoint(v[tri.v1].X(), v[tri.v1].Y(), v[tri.v1].Z());
        vd[ti * 9 + 3 + 0] = v[tri.v2].X();
        vd[ti * 9 + 3 + 1] = v[tri.v2].Y();
        vd[ti * 9 + 3 + 2] = v[tri.v2].Z();
        bbox.GrowToPoint(v[tri.v2].X(), v[tri.v2].Y(), v[tri.v2].Z());
        vd[ti * 9 + 6 + 0] = v[tri.v3].X();
        vd[ti * 9 + 6 + 1] = v[tri.v3].Y();
        vd[ti * 9 + 6 + 2] = v[tri.v3].Z();
        bbox.GrowToPoint(v[tri.v3].X(), v[tri.v3].Y(), v[tri.v3].Z());
        if (nd) {
            nd[ti * 9 + 0 + 0] = n[tri.n1].X();
            nd[ti * 9 + 0 + 1] = n[tri.n1].Y();
            nd[ti * 9 + 0 + 2] = n[tri.n1].Z();
            nd[ti * 9 + 3 + 0] = n[tri.n2].X();
            nd[ti * 9 + 3 + 1] = n[tri.n2].Y();
            nd[ti * 9 + 3 + 2] = n[tri.n2].Z();
            nd[ti * 9 + 6 + 0] = n[tri.n3].X();
            nd[ti * 9 + 6 + 1] = n[tri.n3].Y();
            nd[ti * 9 + 6 + 2] = n[tri.n3].Z();
        }
        if (td) {
            td[ti * 6 + 0 + 0] = t[tri.t1].X();
            td[ti * 6 + 0 + 1] = t[tri.t1].Y();
            td[ti * 6 + 2 + 0] = t[tri.t2].X();
            td[ti * 6 + 2 + 1] = t[tri.t2].Y();
            td[ti * 6 + 4 + 0] = t[tri.t3].X();
            td[ti * 6 + 4 + 1] = t[tri.t3].Y();
        }
        fd[ti * 3 + 0] = ti * 3;
        fd[ti * 3 + 1] = ti * 3 + 1;
        fd[ti * 3 + 2] = ti * 3 + 2;
    }

    // TODO: normal smoothing?
    // TODO: data consolidation?

    mesh.SetVertexData(
        static_cast<unsigned int>(tris.Count() * 3), vd, nd, NULL, td, true); // now don't delete vd, nd, or td
    //mesh.SetTriangleData(0, NULL, false);
    mesh.SetTriangleData(tris.Count(), fd, true);

    if (this->bbox.IsEmpty())
        this->bbox = bbox;
    else
        this->bbox.Union(bbox);
}
