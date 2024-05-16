/**************************************************************************\
* **  2004 Thomas Klein <thomas.klein@informatik.uni-stuttgart.de>      ** *
****************************************************************************
*                                                                          *
* File: datRaw.h                                                           *
*                                                                          *
* Implements a loader for files in the dat-raw-format.                     *
*                                                                          *
* File format(s):                                                          *
*                                                                          *
* A dat-raw dataset consist of two (or more) files:                        *
*   - the dat-file that stores meta data describing the raw data           *
*   - the raw-file(s) that store(s) the actual data in binary format       *
*                                                                          *
*   The raw-file stores the binary data as a M-dimensional array of        *
*   N-dimensional tupels. Each element of the tupel has the same type      *
*   and the first dimension varies fastest.                                *
*                                                                          *
*   The dat-file describes the raw-file(s) using the following tags:       *
*     OBJECTFILENAME  - the name(s) of the raw-file(s)                     *
*                       For a single raw-file this is just the file name,  *
*                       for multiple raw-files, forming for example a      *
*                       time-series, a the numbering is controlled by a    *
*                       format string. More details below.                 *
*     FORMAT          - The format (or data type) of a single element.     *
*                       Currently supported are:                           *
*                        - CHAR (8bit signed int)                          *
*                        - UCHAR (8bit unsigned int)                       *
*                        - SHORT (16bit signed int)                        *
*                        - USHORT (16bit unsigned int)                     *
*                        - INT (32bit signed int)                          *
*                        - UINT (32bit unsigned int)                       *
*                        - LONG (64bit signed int)                         *
*                        - ULONG (64bit unsigned int)                      *
*                        - HALF (16bit float format)                       *
*                          (1 sign bit + 5bit exponent + 10b mantissa)     *
*                        - FLOAT (32bit IEEE single float format)          *
*                        - LONG (64bit IEEE double float format)           *
*     GRIDTYPE        - The type of grid the data is organized in.         *
*                       Currently only the UNIFORM type is supported.      *
*     COMPONENTS      - number N of components per tupel (int value)       *
*     DIMENSIONS      - dimensionality M of the grid (int value)           *
*     TIMESTEPS       - number of time steps/ number of raw files          *
*     BYTEORDER       - byte order of the raw-file is stored in; either    *
*                       LITTLE_ENDIAN (default) or BIG_ENDIAN              *
*     DATAOFFSET      - byte offset in the ra-file(s) where the actual     *
*                       data starts                                        *
*     RESOLUTION      - resolution of the grid, i.e. number of elements in *
*                       each dimension. (M int values, X Y Z ...)          *
*     SLICETHICKNESS  - size of the grid cells in each direction/dimension *
*                       (M float values, (dX dY dZ ...)                    *
*                                                                          *
* The raw data for a data set consisting of multiple timesteps can be      *
* either stored consecutively in a single file  or in a separate file per  *
* time step. In the second case, the ObjectFilename must contain a         *
* conversion specification similar to the printf format string. It starts  *
* with the '%' character followed by an optional padding flag, field       *
* width, skip, and stride modifier and has to end with the conversion      *
* specifier 'd'. The padding flags ('0', '-', ' ') and the minimum field   *
* width have the same meaning as in the *printf specification. The skip    *
* flag (a '+' followed by a positive decimal value) gives the enumeration  *
* of the first data file. The default skip offset is 0, thus the first     *
* time step is assumed to be stored in the file enumerated with 0.         *
* The stride (a '*' followed by a positive decimal value) specifies the    *
* offset between two consecutive file enumerations.                        *
*                                                                          *
* Example: data%03+1*2d.raw specifies the data files                       *
*          data001.raw, data003.raw, data005.raw, data007.raw, ...         *
*                                                                          *
* Example of a dat-file:                                                   *
*                                                                          *
*    OBJECTFILENAME: test%06+2*5d.raw                                      *
*    FORMAT: DOUBLE                                                        *
*    GRIDTYPE: UNIFORM                                                     *
*    COMPONENTS: 3                                                         *
*    DIMENSIONS: 3                                                         *
*    TIMESTEPS: 5                                                          *
*    BYTEORDER: LITTLE_ENDIAN                                              *
*    RESOLUTION: 50 60 70                                                  *
*    SLICETHICKNESS: 0.200000 0.300000 0.400000                            *
*                                                                          *
* This describes a 5 time steps time series of a 3D vector field, sampled  *
* on a 50x60x70 grid consisting of 0.2x0.3x0.4 cuboids and stored as 64bit *
* IEEE double-precision values. The data files are named test000002.raw,   *
* test000007.raw, test000012.raw, test000017.raw, and test000022.raw       *
* (skip=2, stride=5, padded to 6 digits).                                  *
*                                                                          *
\**************************************************************************/

#ifndef __DATRAW_H_
#define __DATRAW_H_

#include <stdio.h>
#include <stdlib.h>

#include <zlib.h>

#ifdef _WIN32
typedef __int8 DR_CHAR;
typedef unsigned __int8 DR_UCHAR;
typedef __int16 DR_SHORT;
typedef unsigned __int16 DR_USHORT;
typedef __int32 DR_INT;
typedef unsigned __int32 DR_UINT;
typedef __int64 DR_LONG;
typedef unsigned __int64 DR_ULONG;
typedef unsigned __int16 DR_HALF;
typedef float DR_FLOAT;
typedef double DR_DOUBLE;
#else
#include <inttypes.h>
typedef int8_t DR_CHAR;
typedef uint8_t DR_UCHAR;
typedef int16_t DR_SHORT;
typedef uint16_t DR_USHORT;
typedef int32_t DR_INT;
typedef uint32_t DR_UINT;
typedef int64_t DR_LONG;
typedef uint64_t DR_ULONG;
typedef uint16_t DR_HALF;
typedef float DR_FLOAT;
typedef double DR_DOUBLE;
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern int datRaw_LogLevel;

enum DatRawLogLevel { DR_LOG_ERROR = 0, DR_LOG_WARNING, DR_LOG_INFO };

enum DatRawDataFormat {
    DR_FORMAT_NONE = 0,
    DR_FORMAT_CHAR,
    DR_FORMAT_UCHAR,
    DR_FORMAT_SHORT,
    DR_FORMAT_USHORT,
    DR_FORMAT_INT,
    DR_FORMAT_UINT,
    DR_FORMAT_LONG,
    DR_FORMAT_ULONG,
    DR_FORMAT_HALF,
    DR_FORMAT_FLOAT,
    DR_FORMAT_DOUBLE,
    DR_FORMAT_RAW
};

enum DatRawGridType { DR_GRID_NONE = 0, DR_GRID_CARTESIAN, DR_GRID_RECTILINEAR, DR_GRID_TETRAHEDRAL };

enum DatRawByteOrder { DR_BIG_ENDIAN, DR_LITTLE_ENDIAN };

typedef struct {
    /* required fields */
    char* name;
    int format;
    int numComponents;
    int timeDependent;
    /* output field, must be initialized with NULL when reading or with data
     * when writing
     */
    void* data;
} DatRawOptionalField;

typedef struct {
    char* descFileName;
    char* dataFileName;
    int dimensions;
    int timeSteps;
    int gridType;
    int numComponents;
    int dataFormat;
    /* only for cartesian and rectilinear grids */
    int* resolution;
    float* sliceDist; /* For tetrahedral grids, resolution entries per
                         dimension are concatenated. */
    float* origin;    /* addition for the scivis contest */
    /* only for tetrahedral grids */
    int numVertices;
    int numTetrahedra;
    /* internal data; do not change! */
    int multiDataFiles;
    gzFile fd_dataFile;
    int currentStep;
    int byteOrder;
    int dataOffset;
} DatRawFileInfo;


/* Applies default values to all fields in 'info'. */
void datRaw_initHeader(DatRawFileInfo* info);

/*
   reads the header data from file and fills info structure
   returns 0 if an error occured.
   optionalFields can be NULL, or a NULL terminated array of pointers to
   optional meta fields descriptions.
*/
int datRaw_readHeader(const char* file, DatRawFileInfo* info, DatRawOptionalField** optionalFields);

/*
    returns the size (in byte) of one data element (one component)
    of data format 'format'
*/
int datRaw_getFormatSize(int format);

/*
    returns the size (in byte) of one data record of the data described in 'info'
    asumming a data format of 'format'
*/
size_t datRaw_getRecordSize(const DatRawFileInfo* info, int format);

/*
    returns the number of elements in the data described by 'info' for one timestep
*/
size_t datRaw_getElementCount(const DatRawFileInfo* info);

/*
    returns the size of the buffer necessary to hold one timestep of
    the data specified by 'info' stored as 'format'
    i.e. recordSize * elementCount
*/
size_t datRaw_getBufferSize(const DatRawFileInfo* info, int format);

/*
   reads the header data from file, fills info structure, and loads all(!)
   timesteps from the raw file into *buffer. If *buffer is NULL, memory is
   allocated that fits the data size.
   optionalFields can be NULL, or a NULL terminated array of pointers to
   optional meta fields descriptions.
   This function returns 0, if an error occured
*/
int datRaw_load(
    const char* file, DatRawFileInfo* info, DatRawOptionalField** optionalFields, void** buffer, int format);

/*
    loads the data of the next time step into *buffer. If *buffer is NULL,
    memory is allocated that fits the data size.
    Returns -1, if there is no next timestep to load and 0, if an error occured.
*/
int datRaw_getNext(DatRawFileInfo* info, void** buffer, int format);

/*
    loads the data of the previous time step into *buffer. If *buffer is
    NULL, memory is allocated that fits the data size.
    Returns -1, if there is no previous timestep to load and 0, if an error
    occured.
*/
int datRaw_getPrevious(DatRawFileInfo* info, void** buffer, int format);

/*
   loads the i-th time step from the raw file(s) indicated by info into *buffer.
   If *buffer is NULL, memory is allocated that fits the data size.
   Returns -1, if there is no i-th timestep to load and 0, if  an error occured.
 */
int datRaw_loadStep(DatRawFileInfo* info, int n, void** buffer, int format);

/*
    write just the header file specified in info
   optionalFields can be NULL, or a NULL terminated array of pointers to
   optional meta fields descriptions.
*/
int datRaw_writeHeaderFile(const DatRawFileInfo* info, DatRawOptionalField** optionalFields);

/*
   write the buffer content to a file with the options specified in info,
   assumes all timesteps are stored consecutively in buffer. Optionally compress
   the data using gzip.  The data is always written in the byte order of the
   current machine.
   optionalFields can be NULL, or a NULL terminated array of pointers to
   optional meta fields descriptions.
*/
int datRaw_write(
    const DatRawFileInfo* info, DatRawOptionalField** optionalFields, void* buffer, int bufferFormat, int compress);

/*
   write the buffer content to a raw-file, if it is part of a multi-file a new
   file is created, otherwise the data is appended to the raw-file (in this
   case, the data has to be written sequentially starting with timestep 0).
   Optionally compress the data using gzip.  The data is always written in the
   byte order of the current machine.
*/
int datRaw_writeTimestep(const DatRawFileInfo* info, void* buffer, int bufferFormat, int compress, int timeStep);

/*
   print the contents of the info file to stderr
*/
void datRaw_printInfo(const DatRawFileInfo* info);


/*
  close the raw data file
*/
void datRaw_close(DatRawFileInfo* info);

/*
  create description struct
  Note: Depending on the grid type, variable arguments are expected in
        their order of appearance in the info structure
*/
int datRaw_createInfo(DatRawFileInfo* info, const char* descFileName, const char* dataFileName, int dimensions,
    int timeSteps, int gridType, int numComponents, int dataFormat, ...);

/*
  check whether info matches values;  0 == don't care
*/
int datRaw_checkInfo(const DatRawFileInfo* info, const char* descFileName, const char* dataFileName, int dimensions,
    int timeSteps, int gridType, int numComponents, int dataFormat, ...);


int datRaw_copyInfo(DatRawFileInfo* newinfo, DatRawFileInfo* oldinfo);
void datRaw_freeInfo(DatRawFileInfo* info);

const char* datRaw_getDataFormatName(int val);
const char* datRaw_getGridTypeName(int val);

char* getMultifileFilename(DatRawFileInfo* info, int timeStep);

#ifdef __cplusplus
}
#endif

#endif
