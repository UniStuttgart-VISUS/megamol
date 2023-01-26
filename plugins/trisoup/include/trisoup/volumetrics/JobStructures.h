#ifndef MEGAMOLCORE_JOBSTRUCTURES_INCLUDED
#define MEGAMOLCORE_JOBSTRUCTURES_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "geometry_calls/MultiParticleDataCall.h"
#include "trisoup/trisoupVolumetricDataCall.h"
#include "vislib/Array.h"
#include "vislib/PtrArray.h"
#include "vislib/SmartPtr.h"
#include "vislib/forceinline.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/ShallowShallowTriangle.h"
#include "vislib/math/mathtypes.h"
#include "vislib/sys/CriticalSection.h"

/** forward declaration */
namespace megamol::trisoup_gl::volumetrics {
class VoluMetricJob;
}

namespace megamol {
namespace trisoup {
namespace volumetrics {

/** typdef steering the arithmetic precision of the voxelizer. */
typedef /*float*/ double VoxelizerFloat;

/** forward declaration */
class BorderVoxel;

typedef BorderVoxel* BorderVoxelElement;
typedef vislib::PtrArray<BorderVoxel> BorderVoxelArray;

/**
 * Utility class for less precise (larger-epsilon) comparison between doubles.
 * Employed for stitching.
 */
class Dowel {
private:
    /**
     * Test for equality of <double, 3> points. This operation uses the
     * IsEqual with a hard-coded, large, epsilon.
     *
     * @param rhs The right hand side operand.
     *
     * @param true, if 'rhs' and this point are equal, false otherwise.
     */
    template<class Tp>
    VISLIB_FORCEINLINE static bool IsEqual(
        const vislib::math::AbstractPoint<double, 3, Tp>& a, const vislib::math::AbstractPoint<double, 3, Tp>& b) {
        return (vislib::math::IsEqual(a.GetX(), b.GetX(), static_cast<double>(vislib::math::FLOAT_EPSILON))) &&
               (vislib::math::IsEqual(a.GetY(), b.GetY(), static_cast<double>(vislib::math::FLOAT_EPSILON))) &&
               (vislib::math::IsEqual(a.GetZ(), b.GetZ(), static_cast<double>(vislib::math::FLOAT_EPSILON)));
    }

    /**
     * Test for equality of <float, 3> points. This operation is a pass-through to
     * Point's == operator.
     *
     * @param rhs The right hand side operand.
     *
     * @param true, if 'rhs' and this point are equal, false otherwise.
     */
    template<class Tp>
    VISLIB_FORCEINLINE static bool IsEqual(
        const vislib::math::AbstractPoint<float, 3, Tp>& a, const vislib::math::AbstractPoint<float, 3, Tp>& b) {
        return a == b;
    }

public:
    /**
     * Answer whether this and rhs have at least one edge in common. Uses Dowel::IsEqual.
     *
     * @param rhs the other triangle
     *
     * @return true if this and rhs have one edge in common
     */
    VISLIB_FORCEINLINE static bool HaveCommonEdge(const vislib::math::ShallowShallowTriangle<double, 3>& a,
        const vislib::math::ShallowShallowTriangle<double, 3>& b) {
        return ((Dowel::IsEqual(a.PeekCoordinates()[0], b.PeekCoordinates()[0]) ||
                    Dowel::IsEqual(a.PeekCoordinates()[0], b.PeekCoordinates()[1]) ||
                    Dowel::IsEqual(a.PeekCoordinates()[0], b.PeekCoordinates()[2]))
                       ? 1U
                       : 0U) +
                   ((Dowel::IsEqual(a.PeekCoordinates()[1], b.PeekCoordinates()[0]) ||
                        Dowel::IsEqual(a.PeekCoordinates()[1], b.PeekCoordinates()[1]) ||
                        Dowel::IsEqual(a.PeekCoordinates()[1], b.PeekCoordinates()[2]))
                           ? 1U
                           : 0U) +
                   ((Dowel::IsEqual(a.PeekCoordinates()[2], b.PeekCoordinates()[0]) ||
                        Dowel::IsEqual(a.PeekCoordinates()[2], b.PeekCoordinates()[1]) ||
                        Dowel::IsEqual(a.PeekCoordinates()[2], b.PeekCoordinates()[2]))
                           ? 1U
                           : 0U) >=
               2U;
    }

    /**
     * Answer whether this and rhs have at least one edge in common. This operation is a pass-through to
     * Triangle's == operator.
     *
     * @param rhs the other triangle
     *
     * @return true if this and rhs have one edge in common
     */
    VISLIB_FORCEINLINE static bool HaveCommonEdge(const vislib::math::ShallowShallowTriangle<float, 3>& a,
        const vislib::math::ShallowShallowTriangle<float, 3>& b) {
        return a.HasCommonEdge(b);
    }
};

/**
 * Class used for storing information about voxels that potentially touch neighboring
 * sub-volumes and thus need to be checked for stitching. A copy of the resulting
 * geometry is retained such that the FatVoxel volume can be safely discarded.
 */
class BorderVoxel {
public:
    inline BorderVoxel() {
        // intentionally empy
    }

    inline ~BorderVoxel() {
        this->triangles.Clear();
    }

    /** Array of triangles in this voxel */
    vislib::Array<VoxelizerFloat> triangles;

    /** ABSOLUTE x coordinate in the virtual volume encompassing the whole dataset */
    unsigned int x;

    /** ABSOLUTE y coordinate in the virtual volume encompassing the whole dataset */
    unsigned int y;

    /** ABSOLUTE z coordinate in the virtual volume encompassing the whole dataset */
    unsigned int z;

    /**
     * Test for equality, i.e. whether they live at the same coordinate and nothing else(!).
     *
     * @param rhs The right side operand.
     *
     * @return true, if *this and 'rhs' are equal, false otherwise.
     */
    inline bool operator==(const BorderVoxel& rhs) {
        return (this->x == rhs.x) && (this->y == rhs.y) && (this->z == rhs.z);
    }

    /**
     * Test whether the geometry in two BorderVoxels does touch, i.e. have a common
     * edge. For stitching surfaces. If double-precision arithmetic is used, the Dowel
     * utility class is employed for a more forgiving epsilon.
     *
     * @param rhs The right side operand.
     *
     * @return true, if *this and rhs contain geometry that can be stitched.
     */
    inline bool doesTouch(const BorderVoxel& rhs) {
        // this one can only work for marching cubes since there are only edges on cube faces! (I think)
        //if ((this->x == rhs.x) || (this->y == rhs.y) || (this->z == rhs.z)) {

        //if (vislib::math::Point<float, 3>(this->x, this->y, this->z).Distance(
        //            vislib::math::Point<float, 3>(rhs.x, rhs.y, rhs.z)) < 2) {
        if (((this->x - rhs.x) * (this->x - rhs.x) + (this->y - rhs.y) * (this->y - rhs.y) +
                (this->z - rhs.z) * (this->z - rhs.z)) <= 2) {
            vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3> sst1(
                const_cast<VoxelizerFloat*>(this->triangles.PeekElements()));
            vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3> sst2(
                const_cast<VoxelizerFloat*>(rhs.triangles.PeekElements()));
            for (unsigned int i = 0; i < this->triangles.Count() / 9; i++) {
                for (unsigned int j = 0; j < rhs.triangles.Count() / 9; j++) {
                    sst1.SetPointer(const_cast<VoxelizerFloat*>(this->triangles.PeekElements() + i * 9));
                    sst2.SetPointer(const_cast<VoxelizerFloat*>(rhs.triangles.PeekElements() + j * 9));
                    if (Dowel::HaveCommonEdge(sst1, sst2)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
     * Test whether the geometry in two BorderVoxels does touch, i.e. have a common
     * edge. For stitching surfaces. If double-precision arithmetic is used, the Dowel
     * utility class is employed for a more forgiving epsilon.
     *
     * @param rhs The right side operand.
     *
     * @return true, if *this and *rhs contain geometry that can be stitched.
     */
    inline bool doesTouch(const BorderVoxelElement& rhs) {
        return this->doesTouch(*rhs);
    }
};

/**
 * Data structure for the sampling of a glyph-based dataset into a distance field.
 * Currently the whole code assumes the lowest-value (lower-left-behind) corner
 * of the voxel as the point sampled.
 */
struct FatVoxel {
    /**
     * Distance to the nearest glyph surface. Negative distances indicate
     * we are inside the geometry.
     */
    VoxelizerFloat distField;

    /** Pointer to the memory holding numTriangles * 3 * 3 VoxelizerFloats. */
    VoxelizerFloat* triangles;

    /**
     * Pointer to the memory holding numTriangles VoxelizerFloats containing the
     * volume associated with each triangle.
     */
    VoxelizerFloat* volumes;

    /**
     * Pointer to the memory holding numTriangles unsigned chars containing the
     * respective corner the tet was anchored on. Can be used for growing the
     * trivial neighbors.
     */
    unsigned char* corners;

    /** Number of triangles contained in this FatVoxel */
    unsigned char numTriangles;

    /**
     * For marching CUBES, just that: the case from the table. For marching tets,
     * the Number of < 0, i.e. inside-the-volume corners
     */
    unsigned char mcCase;

    /**
     * bit field to remember the triangles already collected when stitching
     * surfaces.
     */
    /*unsigned*/ short consumedTriangles;

    /**
     * BorderVoxel containing a copy of the geometry for those FatVoxels that
     * potentially touch other subvolumes. This memory must NOT be deleted alongside
     * the FatVoxel since it is linked for fast access/deduplication, but needs to
     * survive the volume itself.
     */
    BorderVoxelElement borderVoxel;

    /**
     * thomasbm: surface that might enclose all surfaces withing this voxel.
     * This is used to detect and remove entirely enclosed surfaces.
     */
    //class Surface *enclosingCandidate;
};

/**
 * we introduced this class to detect enclosed surfaces
 * thomasbm: we need that to avoid carrying an initialized-flag for
 * each boudning box with us...
 */
//#define PARALLEL_BBOX_COLLECT
template<class T>
class BoundingBox {
    vislib::math::Cuboid<T> box;
    bool initialized;

public:
    inline BoundingBox() : initialized(false) {}
    inline void AddPoint(vislib::math::Point<T, 3> p) {
        if (!initialized) {
            initialized = true;
            box.Set(p.X(), p.Y(), p.Z(), p.X(), p.Y(), p.Z());
        } else
            box.GrowToPoint(p);
    }
    inline void Union(const BoundingBox<T>& other) {
        //ASSERT(other.initialized);
        if (!other.initialized)
            return;
        if (!initialized) {
            initialized = true;
            box = other.box;
        } else
            box.Union(other.box);
    }
    inline bool operator==(const BoundingBox<T>& o) const {
        return (initialized == o.initialized) && (box == o.box);
    }
    inline bool IsInitialized() const {
        return this->initialized;
    }

    enum CLASSIFY_STATUS {
        CONTAINS_OTHER,
        IS_CONTAINED_BY_OTHER,
        UNSPECIFIED // contains partial intersection for now ...
    };

    inline CLASSIFY_STATUS Classify(const BoundingBox<T>& o) const {
        vislib::math::Cuboid<T> _union(box);
        ASSERT(this->initialized && o.initialized);
        _union.Union(o.box);
        T volume = _union.Volume();
        if (volume <= this->box.Volume())
            return CONTAINS_OTHER;
        if (volume <= o.box.Volume())
            return IS_CONTAINED_BY_OTHER;
        return UNSPECIFIED;
    }
};

/**
 * represents a contiguous surface inside a subvolume and associated
 * metrics.
 */
class Surface {
public:
    Surface() : border(new BorderVoxelArray()) {}

    ~Surface() {
        this->border = NULL;
    }

    /** array of coordinates forming triangles making up the surface */
    vislib::Array<VoxelizerFloat> mesh;

    /** array of Bordervoxels */
    vislib::SmartPtr<BorderVoxelArray> border;

    /** thomasbm: bounding box of the triangle mesh */
    BoundingBox<unsigned> boundingBox;

    /** surface area */
    VoxelizerFloat surface;

    /** volume encompassed by this surface */
    VoxelizerFloat volume;

    /** thomasbm: volume of the "void" space so we can add up enclosed space later ... */
    VoxelizerFloat voidVolume;

    unsigned int globalID;

    /**
     * bit field indicating on which subvolume faces the attachment of a
     * full subvolume is indicated.
     */
    unsigned char fullFaces;

    /**
     * thomasbm: a list of surfaces that might be enclosed by this surface.
     */
    //vislib::Array<Surface*> enclSurfaces;

    /**
     * Answer the equality of this and rhs. Equality actually means
     * equality of surface and volume.
     *
     * @param rhs the right hand side operand
     *
     * @return whether this and rhs are equal
     */
    inline bool operator==(const Surface& rhs) {
        return (this->surface == rhs.surface) && (this->volume == rhs.volume);
    }
};

/**
 * Struct containing the results of a marching (whatever) on an instance of SubJubData.
 */
struct SubJobResult {
    /** Array of the surfaces resulting from the Voxelization */
    vislib::Array<Surface> surfaces;

    //#ifdef _DEBUG
    /* for debug purposes only - volume density values - dimension are given by sjd->resXYZ*/
    trisoupVolumetricDataCall::Volume debugVolume;
    //#endif

    /** whether this job has completed runnning */
    bool done;

    /** ctor (yuck). mostly sets done to false just to be sure */
    SubJobResult() : done(false) {}
};

/**
 * Structure that holds all parameters so a single thread can work
 * on his subset of the total volume occupied by a particle list
 */
struct SubJobData {
    /** Volume to work on and rasterize the data of */
    vislib::math::Cuboid<VoxelizerFloat> Bounds;

    /** volume resolution, i.e. number of subdivisions with respect to bounds */
    int resX;

    /** volume resolution, i.e. number of subdivisions with respect to bounds */
    int resY;

    /** volume resolution, i.e. number of subdivisions with respect to bounds */
    int resZ;

    /** global voxel position offset (for absolute voxel positions) */
    int offsetX;

    /** global voxel position offset (for absolute voxel positions) */
    int offsetY;

    /** global voxel position offset (for absolute voxel positions) */
    int offsetZ;

    int gridX;
    int gridY;
    int gridZ;

    /** Maximum radius in the datasource. */
    VoxelizerFloat MaxRad;

    /** radius multiplication factor for calibration */
    VoxelizerFloat RadMult;

    /**
     * Edge length of a voxel (set in accordance to the radii of
     * the contained particles)
     */
    VoxelizerFloat CellSize;

    /** datacall that gives access to the particles */
    geocalls::MultiParticleDataCall* datacall;

    /** here the Job should store its results */
    SubJobResult Result;

    /** whether to persist the geometry computation takes place on (in result.mesh) */
    bool storeMesh, storeVolume;

    trisoup_gl::volumetrics::VoluMetricJob* parent;

    VISLIB_FORCEINLINE unsigned cellIndex(const vislib::math::Point<unsigned, 3>& p) {
        return cellIndex(p.X(), p.Y(), p.Z());
    }
    VISLIB_FORCEINLINE unsigned cellIndex(unsigned x, unsigned y, unsigned z) {
        return (z * this->resY + y) * this->resX + x;
    }

    /*   if ((((mN.X() < 0) && (x > 0)) || (mN.X() == 0) || ((mN.X() > 0) && (x < sjd->resX - 2))) &&
             (((mN.Y() < 0) && (y > 0)) || (mN.Y() == 0) || ((mN.Y() > 0) && (y < sjd->resY - 2))) &&
             (((mN.Z() < 0) && (z > 0)) || (mN.Z() == 0) || ((mN.Z() > 0) && (z < sjd->resZ - 2))))
         -> same as coordsInside(mN.X()+x, mN.Y()+y, mN.Z()+z) ?! */

    VISLIB_FORCEINLINE bool coordsInside(const vislib::math::Point<int, 3>& p) {
        return coordsInside(p.X(), p.Y(), p.Z());
    }
    VISLIB_FORCEINLINE bool coordsInside(int x, int y, int z) {
        return (x >= 0 && x < this->resX - 1) && (y >= 0 && y < this->resY - 1) && (z >= 0 && z < this->resZ - 1);
    }

    VISLIB_FORCEINLINE bool isBorder(unsigned x, unsigned y, unsigned z) {
        return (x == 0) || (x == this->resX - 2) || (y == 0) || (y == this->resY - 2) || (z == 0) ||
               (z == this->resZ - 2);
    }
};


} /* end namespace volumetrics */
} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_JOBSTRUCTURES_INCLUDED */
