/*
 * WavefrontObjDataSource.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "WavefrontObjDataSource.h"
#include "vislib/Array.h"
#include "vislib/assert.h"
#include "vislib/ASCIIFileBuffer.h"
#include "vislib/Cuboid.h"
#include "vislib/File.h"
#include "vislib/Log.h"
#include "vislib/MemmappedFile.h"
#include "vislib/Path.h"
#include "vislib/PerformanceCounter.h"
#include "vislib/PtrArray.h"
#include "vislib/ShallowVector.h"
#include "vislib/StringConverter.h"
#include "vislib/Vector.h"

using namespace megamol;
using namespace megamol::trisoup;


/*
 * WavefrontObjDataSource::WavefrontObjDataSource
 */
WavefrontObjDataSource::WavefrontObjDataSource(void) : AbstractTriMeshLoader() {
    // intentionally empty
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
    using vislib::sys::Log;

    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Filename is empty");
        return true;
    }
    if (!vislib::sys::File::Exists(filename)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Specified file does not exist");
        return false;
    }

    vislib::sys::ASCIIFileBuffer lines(vislib::sys::ASCIIFileBuffer::PARSING_WORDS);
    if (!lines.LoadFile(filename)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load file");
        return false;
    }

    this->objs.Clear();
    this->mats.Clear();

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Start loading \"%s\"\n", vislib::StringA(filename).PeekBuffer());
    double startTime = vislib::sys::PerformanceCounter::QueryMillis();

    vislib::TString path = vislib::sys::Path::GetDirectoryName(filename);

    float vec3[3];
    vislib::math::ShallowVector<float, 3> vec3v3(vec3);
    vislib::math::ShallowVector<float, 2> vec3v2(vec3);
    vislib::Array<vislib::math::Vector<float, 3> > vert;
    vislib::Array<vislib::math::Vector<float, 3> > norm;
    vislib::Array<vislib::math::Vector<float, 2> > texc;
    vislib::Array<vislib::StringA> objsMats;
    vislib::PtrArray<vislib::Array<Tri> > objs;
    vislib::Array<Tri> *obj = NULL;
    vislib::Array<vislib::StringA> matNames;

    const SIZE_T capacityGrowth = 10 * 1024;

    for (SIZE_T li = 0; li < lines.Count(); li++) {
        const vislib::sys::ASCIIFileBuffer::LineBuffer& line = lines[li];
        if (line.Count() <= 0) continue;

        try {

            if ((::strcmp(line[0], "mtllib") == 0) && (line.Count() >= 2)) {
                // load material library
                this->loadMaterialLibrary(vislib::sys::Path::Concatenate(path, A2T(line[1])), matNames);

            } else if ((::strcmp(line[0], "usemtl") == 0) && (line.Count() >= 2)) {
                // use material (new group)

                if (obj == NULL) {
                    objs.Append(new vislib::Array<Tri>());
                    objsMats.Append(line[1]);
                    obj = objs.Last();

                } else if (objsMats.Last().IsEmpty()) {
                    objsMats.Last() = line[1];

                } else if (!objsMats.Last().Equals(line[1])) {
                    if (obj->Count() > 0) {
                        objs.Append(new vislib::Array<Tri>());
                        objsMats.Append(line[1]);
                        obj = objs.Last();

                    } else {
                        objsMats.Last() = line[1];
                    }

                }

            } else if ((::strcmp(line[0], "v") == 0) && (line.Count() >= 4)) {
                // vertex
                vec3[0] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line[1]));
                vec3[1] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line[2]));
                vec3[2] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line[3]));
                if (vert.Count() == vert.Capacity()) vert.AssertCapacity(vert.Capacity() + capacityGrowth);
                vert.Add(vec3v3);

            } else if ((::strcmp(line[0], "vn") == 0) && (line.Count() >= 4)) {
                // vertex normal
                vec3[0] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line[1]));
                vec3[1] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line[2]));
                vec3[2] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line[3]));
                if (norm.Count() == norm.Capacity()) norm.AssertCapacity(norm.Capacity() + capacityGrowth);
                norm.Add(vec3v3);

            } else if ((::strcmp(line[0], "vt") == 0) && (line.Count() >= 2)) {
                // vertex texture coordinate
                vec3[0] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line[1]));
                vec3[1] = (line.Count() >= 3) ? static_cast<float>(vislib::CharTraitsA::ParseDouble(line[2])) : 0.0f;
                if (texc.Count() == texc.Capacity()) texc.AssertCapacity(texc.Capacity() + capacityGrowth);
                texc.Add(vec3v2);

            } else if ((::strcmp(line[0], "f") == 0) && (line.Count() >= 4)) {
                // face
                if (obj == NULL) {
                    objs.Append(new vislib::Array<Tri>());
                    objsMats.Append(vislib::StringA::EMPTY);
                    obj = objs.Last();
                }

                Tri t;
                const char* p1 = ::strchr(line[1], '/');
                int idx;

                if (p1 == NULL) {
                    t.n = t.t = false;
                    t.n1 = t.n2 = t.n3 = 0;
                    t.t1 = t.t2 = t.t3 = 0;
                    idx = vislib::CharTraitsA::ParseInt(line[1]);
                    if (idx <= 0) throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                    t.v1 = static_cast<unsigned int>(idx - 1);
                    idx = vislib::CharTraitsA::ParseInt(line[2]);
                    if (idx <= 0) throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                    t.v2 = static_cast<unsigned int>(idx - 1);
                    for (unsigned int i = 3; i < line.Count(); i++) {
                        idx = vislib::CharTraitsA::ParseInt(line[i]);
                        if (idx <= 0) throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                        t.v3 = static_cast<unsigned int>(idx - 1);
                        obj->Append(t);
                        t.v2 = t.v3;
                    }

                } else {
                    SIZE_T len = vislib::CharTraitsA::SafeStringLength(line[1]);
                    const char* p2 = ::strchr(p1 + 1, '/');
                    if (p2 == NULL) throw vislib::Exception("Single slash face entry element illegal", __FILE__, __LINE__);
                    t.t = (p2 > p1 + 1);
                    t.n = (p2 + 1 < line[1] + len);
                    t.n1 = t.n2 = t.n3 = 0;
                    t.t1 = t.t2 = t.t3 = 0;
                    *const_cast<char *>(p1) = '\0';
                    *const_cast<char *>(p2) = '\0';
                    p1++;
                    p2++;
                    idx = vislib::CharTraitsA::ParseInt(line[1]);
                    if (idx <= 0) throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                    t.v1 = static_cast<unsigned int>(idx - 1);
                    if (t.t) {
                        idx = vislib::CharTraitsA::ParseInt(p1);
                        if (idx <= 0) throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                        t.t1 = static_cast<unsigned int>(idx - 1);
                    }
                    if (t.n) {
                        idx = vislib::CharTraitsA::ParseInt(p2);
                        if (idx <= 0) throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                        t.n1 = static_cast<unsigned int>(idx - 1);
                    }

                    len = vislib::CharTraitsA::SafeStringLength(line[2]);
                    p1 = ::strchr(line[2], '/');
                    if (p1 == NULL) throw vislib::Exception("Single slash face entry element illegal", __FILE__, __LINE__);
                    p2 = ::strchr(p1 + 1, '/');
                    if (p2 == NULL) throw vislib::Exception("Single slash face entry element illegal", __FILE__, __LINE__);
                    if (t.t != (p2 > p1 + 1)) throw vislib::Exception("face entry inconsistancy", __FILE__, __LINE__);
                    if (t.n != (p2 + 1 < line[2] + len)) throw vislib::Exception("face entry inconsistancy", __FILE__, __LINE__);
                    *const_cast<char *>(p1) = '\0';
                    *const_cast<char *>(p2) = '\0';
                    p1++;
                    p2++;
                    idx = vislib::CharTraitsA::ParseInt(line[2]);
                    if (idx <= 0) throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                    t.v2 = static_cast<unsigned int>(idx - 1);
                    if (t.t) {
                        idx = vislib::CharTraitsA::ParseInt(p1);
                        if (idx <= 0) throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                        t.t2 = static_cast<unsigned int>(idx - 1);
                    }
                    if (t.n) {
                        idx = vislib::CharTraitsA::ParseInt(p2);
                        if (idx <= 0) throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                        t.n2 = static_cast<unsigned int>(idx - 1);
                    }

                    t.v2 = static_cast<unsigned int>(vislib::CharTraitsA::ParseInt(line[2]) - 1);
                    if (t.t) t.t2 = static_cast<unsigned int>(vislib::CharTraitsA::ParseInt(p1) - 1);
                    if (t.n) t.n2 = static_cast<unsigned int>(vislib::CharTraitsA::ParseInt(p2) - 1);

                    for (unsigned int i = 3; i < line.Count(); i++) {
                        len = vislib::CharTraitsA::SafeStringLength(line[i]);
                        p1 = ::strchr(line[i], '/');
                        if (p1 == NULL) throw vislib::Exception("Single slash face entry element illegal", __FILE__, __LINE__);
                        p2 = ::strchr(p1 + 1, '/');
                        if (p2 == NULL) throw vislib::Exception("Single slash face entry element illegal", __FILE__, __LINE__);
                        if (t.t != (p2 > p1 + 1)) throw vislib::Exception("face entry inconsistancy", __FILE__, __LINE__);
                        if (t.n != (p2 + 1 < line[i] + len)) throw vislib::Exception("face entry inconsistancy", __FILE__, __LINE__);
                        *const_cast<char *>(p1) = '\0';
                        *const_cast<char *>(p2) = '\0';
                        p1++;
                        p2++;
                        idx = vislib::CharTraitsA::ParseInt(line[i]);
                        if (idx <= 0) throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                        t.v3 = static_cast<unsigned int>(idx - 1);
                        if (t.t) {
                            idx = vislib::CharTraitsA::ParseInt(p1);
                            if (idx <= 0) throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                            t.t3 = static_cast<unsigned int>(idx - 1);
                        }
                        if (t.n) {
                            idx = vislib::CharTraitsA::ParseInt(p2);
                            if (idx <= 0) throw vislib::Exception("Negative face element indices not supported", __FILE__, __LINE__);
                            t.n3 = static_cast<unsigned int>(idx - 1);
                        }
                        obj->Append(t);
                        t.v2 = t.v3;
                        t.t2 = t.t3;
                        t.n2 = t.n3;
                    }

                }

            } else if (::strcmp(line[0], "g") == 0) {
                // new group
                if ((obj == NULL) || (obj->Count() > 0)) {
                    objs.Append(new vislib::Array<Tri>());
                    objsMats.Append(vislib::StringA::EMPTY); // or should we keep the material? spec does not tell!
                    obj = objs.Last();
                }
            }

        } catch(vislib::Exception ex) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Error parsing line %u: %s (%s, %d)",
                li, ex.GetMsgA(), ex.GetFile(), ex.GetLine());
        } catch(...) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Error parsing line %u: unexpected exception", li);
        }
    }

    double parseTime = (vislib::sys::PerformanceCounter::QueryMillis() - startTime) * 0.001;
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Parsing file completed after %f seconds\n", parseTime);

    unsigned int *vertUsed = new unsigned int[vert.Count()];

    unsigned int oc = 0;
    for (SIZE_T i = 0; i < objs.Count(); i++) {
        if (objs[i]->Count() > 0) oc++;
    }
    this->objs.SetCount(oc);
    this->objs.Trim();
    if (oc == 0) this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    else this->bbox.Set(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    oc = 0;
    for (SIZE_T i = 0; i < objs.Count(); i++) {
        if (objs[i]->Count() <= 0) continue;
        INT_PTR matIdx = matNames.IndexOf(objsMats[i]);
        this->objs[oc].SetMaterial(
            ((matIdx == vislib::Array<vislib::StringA>::INVALID_POS) || (static_cast<SIZE_T>(matIdx) >= this->mats.Count()))
            ? NULL
            : &this->mats[static_cast<SIZE_T>(matIdx)]);
        this->makeMesh(this->objs[oc], *objs[i], vertUsed, vert, norm, texc);
        oc++;
    }

    delete[] vertUsed;

    return true;
}


/*
 * WavefrontObjDataSource::loadMaterialLibrary
 */
void WavefrontObjDataSource::loadMaterialLibrary(const vislib::TString& filename, vislib::Array<vislib::StringA>& names) {
    using vislib::sys::Log;
    ASSERT(names.Count() == this->mats.Count());

    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Material library filename is empty");
        return;
    }
    if (!vislib::sys::File::Exists(filename)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Material library \"%s\" file does not exist", vislib::StringA(filename).PeekBuffer());
        return;
    }
    vislib::sys::ASCIIFileBuffer lines(vislib::sys::ASCIIFileBuffer::PARSING_WORDS);
    if (!lines.LoadFile(filename)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load material library \"%s\"", vislib::StringA(filename).PeekBuffer());
        return;
    }
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Loading material library \"%s\"", vislib::StringA(filename).PeekBuffer());

    vislib::TString path = vislib::sys::Path::GetDirectoryName(filename);
    Material *mat = NULL;

    for (SIZE_T li = 0; li < lines.Count(); li++) {
        const vislib::sys::ASCIIFileBuffer::LineBuffer& line = lines[li];
        if (line.Count() <= 0) continue;

        try {

            if ((strcmp(line[0], "newmtl") == 0) && (line.Count() >= 2)) {
                INT_PTR idx = names.IndexOf(line[1]);
                if (idx == vislib::Array<vislib::StringA>::INVALID_POS) {
                    names.Append(line[1]);
                    this->mats.Append(Material());
                    mat = &this->mats.Last();
                } else {
                    mat = &this->mats[static_cast<SIZE_T>(idx)];
                }

            } else if (mat == NULL) {
                // ignoring line when there is no active material

            } else if ((strcmp(line[0], "Ns") == 0) && (line.Count() >= 2)) {
                mat->SetNs(static_cast<float>(vislib::CharTraitsA::ParseDouble(line[1])));

            } else if ((strcmp(line[0], "Ni") == 0) && (line.Count() >= 2)) {
                mat->SetNi(static_cast<float>(vislib::CharTraitsA::ParseDouble(line[1])));

            } else if ((strcmp(line[0], "d") == 0) && (line.Count() >= 2)) {
                mat->SetD(static_cast<float>(vislib::CharTraitsA::ParseDouble(line[1])));

            } else if ((strcmp(line[0], "Ka") == 0) && (line.Count() >= 4)) {
                mat->SetKa(
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line[1])),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line[2])),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line[3])));

            } else if ((strcmp(line[0], "Kd") == 0) && (line.Count() >= 4)) {
                mat->SetKd(
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line[1])),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line[2])),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line[3])));

            } else if ((strcmp(line[0], "Ks") == 0) && (line.Count() >= 4)) {
                mat->SetKs(
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line[1])),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line[2])),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line[3])));

            } else if ((strcmp(line[0], "Ke") == 0) && (line.Count() >= 4)) {
                mat->SetKe(
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line[1])),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line[2])),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(line[3])));

            } else if (((strcmp(line[0], "bump") == 0) || (strcmp(line[0], "map_bump") == 0) || (strcmp(line[0], "bump_map") == 0)) && (line.Count() >= 2)) {
                mat->SetBumpMapFileName(vislib::sys::Path::Concatenate(path, line[1]));

            } else if ((strncmp(line[0], "map", 3) == 0) && (line.Count() >= 2)) {
                mat->SetMapFileName(vislib::sys::Path::Concatenate(path, line[1]));

            }

        } catch(vislib::Exception ex) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Error parsing line %u: %s (%s, %d)",
                li, ex.GetMsgA(), ex.GetFile(), ex.GetLine());
        } catch(...) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Error parsing line %u: unexpected exception", li);
        }
    }

    ASSERT(names.Count() == this->mats.Count());
}


/*
 * WavefrontObjDataSource::makeMesh
 */
void WavefrontObjDataSource::makeMesh(CallTriMeshData::Mesh& mesh,
        const vislib::Array<WavefrontObjDataSource::Tri>& tris,
        unsigned int* vu, const vislib::Array<vislib::math::Vector<float, 3> >& v,
        const vislib::Array<vislib::math::Vector<float, 3> >& n,
        const vislib::Array<vislib::math::Vector<float, 2> >& t) {
    ASSERT(tris.Count() > 0);
    ASSERT(v.Count() > 0);

    float *vd = new float[tris.Count() * 3 * 3];  // vertices
    float *nd = (tris[0].n) ? new float[tris.Count() * 3 * 3] : NULL;  // normals
    float *td = (tris[0].t) ? new float[tris.Count() * 3 * 2] : NULL;  // texture coordinates
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
    }

    // TODO: normal smoothing?
    // TODO: data consolidation?

    mesh.SetVertexData(static_cast<unsigned int>(tris.Count() * 3), vd, nd, NULL, td, true); // now don't delete vd, nd, or td
    mesh.SetTriangleData(0, NULL, false);

    if (this->bbox.IsEmpty()) this->bbox = bbox;
    else this->bbox.Union(bbox);
}
