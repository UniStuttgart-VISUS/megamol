/*
 * Mol2Data.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MOL2DATA_H_INCLUDED
#define MEGAMOLCORE_MOL2DATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <climits>
#include <cfloat>

namespace megamol {
namespace core {
namespace vismol2 {

// local configuration defines
#define POINT_RELATIVE  1
#define POINT_FORMAT    0

#if POINT_FORMAT == 0

   typedef float coords_t;
#  define COORDS_TYPE	GL_FLOAT
#  define COORDS_GL(x,y)	x ## f ## y
#  define COORDS_SCALE	FLOAT_SCALE
#  define COORDS_OFFSET 0
#  define COORDS_GFX_MULT 1
#  define COORDS_ROUND(x) (x)
   typedef float psize_t;
#  define PSIZE_TYPE	GL_FLOAT
#  define PSIZE_GL(x,y)	x ## f ## y
#  define PSIZE_SCALE	FLOAT_SCALE
#  define PSIZE_ROUND(x) (x)
#  define ASCII_LOAD_SCALE 1
#  define ASCII_LOAD_OFFSET 0
#  define MOLQUAT_ROUND(x) (x)//((x + 1.0) / 2.0)
#  define MOLDIST_ROUND(x) (x)
#elif POINT_FORMAT == 1

   typedef short coords_t;
#  define COORDS_TYPE	GL_SHORT
#  define COORDS_GL(x,y)	x ## s ## y
#  define COORDS_SCALE	SHRT_MAX
#  define COORDS_OFFSET 0
#  define COORDS_GFX_MULT 1
#  define COORDS_ROUND(x) ((x)+0.5)
   typedef short psize_t;
#  define PSIZE_TYPE	GL_SHORT
#  define PSIZE_GL(x,y)	x ## s ## y
#  define PSIZE_SCALE	SHRT_MAX
#  define PSIZE_ROUND(x) ((x)+0.5)
/* For loading to discretized data assume input data is scaled to BBox */
#  define ASCII_LOAD_SCALE ((float)COORDS_SCALE/LOADBOX_SIZE)
#  define ASCII_LOAD_OFFSET COORDS_OFFSET

#elif POINT_FORMAT == 2
   typedef unsigned char coords_t;
#  define COORDS_TYPE	GL_UNSIGNED_BYTE
#  define COORDS_GL(x,y)	x ## ub ## y
#  define COORDS_SCALE	((UCHAR_MAX-1)/2)
#  define COORDS_OFFSET ((UCHAR_MAX+1)/2)
#  define COORDS_GFX_MULT UCHAR_MAX
#  define COORDS_ROUND(x) ((x)+0.5)
   typedef unsigned char psize_t;
#  define PSIZE_TYPE	GL_UNSIGNED_BYTE
#  define PSIZE_GL(x,y)	x ## ub ## y
#  define PSIZE_SCALE	UCHAR_MAX
#  define PSIZE_ROUND(x) ((x)+0.5)
#  define ASCII_LOAD_SCALE ((float)COORDS_SCALE/LOADBOX_SIZE)
#  define ASCII_LOAD_OFFSET COORDS_OFFSET
#  define MOLQUAT_ROUND(x) (x * 255)//(((x + 1.0) * 128))
#  define MOLDIST_ROUND(x) (x * 255)
#endif

    typedef unsigned char color_t;
#define COLOR_TYPE	GL_UNSIGNED_BYTE
#define COLOR_GL(x,y)	x ## ub ## y
#define COLOR_SCALE	UCHAR_MAX	/* TODO: unused so far */
/* TODO: number of channels / parameters -> lookup texture */


#if POINT_RELATIVE
#  define POINTCOORD(pos,coord,scale)	((pos)+(((float)(coord))-COORDS_OFFSET)*(scale))
#  define POINTSIZE(size,scale,maxp)	((size)*(scale))
#else
#  define POINTCOORD(pos,coord,scale)	(((float)(coord))-COORDS_OFFSET)
#  define POINTSIZE(size,scale,maxp)	((size)*(maxp))
#endif


    typedef struct position_s {
        float x, y, z;
    } position_t;

    typedef struct point_s {
        coords_t x, y, z;		/* relative, scaled with cluster->scale */
        psize_t  s;			/* diameter, rel., scaled w. cluster->scale */
        color_t r, g, b, a;
        coords_t q1, q2, q3, q4;
        color_t type;
        /* target values for linear interpolation */
        coords_t tpx, tpy, tpz;			/* target coordinates */
        coords_t tq1, tq2, tq3, tq4;	/* target quaternions */
        color_t  tcr, tcg, tcb, tca;	/* target color */
        signed char clusTimeDist;
        unsigned char clusShape;
        psize_t  tsize;					/* target diameter */
        int interpolType;
        unsigned long id;
        float r1, r2, r3, tr1, tr2, tr3; /* ellipsoid radii */

        int semClustID;
        float clustGrowth; /* number of molecules a cluster is growing/shrinking in this time slice */

        float devEllPos[9];
    } point_t;


    typedef struct molecule_s {
        float dist, radius1, radius2, radius3;
        float colindex, colindex2;
        float cyllength;
    } molecule_t;

    typedef struct molecule_upload_s {
        coords_t dist, radius1, radius2, radius3;
        coords_t colindex;
        coords_t cyllength;
        coords_t padding0, padding1; // quadruples...
    } molecule_upload_t;

    #define MOL_COUNTER signed short

    typedef struct cluster_s cluster_t;

    struct cluster_s {
        /* children */
        cluster_t *clusters;
        /* points of children */
        point_t   *points;
        /* size of the previous arrays */
        // HAZARD this was long which is NOT a good idea with 64bits because there it is twice as big
        // so we cannot load 32-bit-saved pclds :/ (MEH)
        int       size;
    #if POINT_RELATIVE
        /* children point and size data is scaled by this float */
        float      scale;
    #endif
    };

    typedef struct pointcloud_s {
        cluster_t start;
        int      maxreclevel;
        int      delta_n;		/* diff. rendering vs. cluster selection */
        float    genpointsize;	/* generic point size (max data point size) */
        unsigned int myTime;		/* time of this data set */
        //struct mol21Frame *mol21;	/* additional data */
    } pointcloud_t;

    //void calc_mol_size_stuff();

    ///* returns number of allocated points */
    //long readData(const char *filename, pointcloud_t *pcld, int maxloadlevel);

    //void saveData(const char *filename, pointcloud_t *data);

    ///* returns number of allocated points */
    //long buildhierarchy(pointcloud_t *data);

} /* end namespace vismol2 */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MOL2DATA_H_INCLUDED */
