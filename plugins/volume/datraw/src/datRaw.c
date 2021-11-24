/***************************************************************************
* **  2004 Thomas Klein <thomas.klein@informatik.uni-stuttgart.de>     ** *
***************************************************************************
*                                                                         *
* File: datRaw.c                                                          *
*                                                                         *
* Implents a loader for files in the dat-raw-format.                      *
*                                                                         *
***************************************************************************/

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <malloc.h>
#include <assert.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>

#include "zlib.h"
#include "datRaw.h"
#include "datRaw_log.h"
#include "datRaw_half.h"

#define DATRAW_MAX(x, y)  ((x) > (y) ? (x) : (y))

static char *dupstr(const char *s)
{
    char *dup;

    dup = (char*)malloc((strlen(s) + 1) * sizeof(char));

    if (dup) {
        strcpy(dup, s);
    }

    return dup;
}

static char *dirname(char *path)
{
    static char* dot = ".";
    char *s;

    s = path != NULL ? strrchr(path, '/') : NULL;

    if (!s) {
        return dot;
    } else if (s == path) {
        return path;
    } else if (s[1] == '\0') {
        while (s != path && *s != '/') s--;
        s[1] = '\0';
        return path;
    } else {
        *s = '\0';
        return path;
    }
}

static struct {
    char *tag;
    unsigned int value;
    unsigned int size;
    char *formatString;
}
datRawDataFormats[] = {
    {"CHAR", DR_FORMAT_CHAR, sizeof(DR_CHAR), "%hhd"},
    {"UCHAR", DR_FORMAT_UCHAR, sizeof(DR_UCHAR), "%hhu"},
    {"SHORT", DR_FORMAT_SHORT, sizeof(DR_SHORT), "%hd"},
    {"USHORT", DR_FORMAT_USHORT, sizeof(DR_USHORT), "%hu"},
    {"INT", DR_FORMAT_INT, sizeof(DR_INT), "%d"},
    {"UINT", DR_FORMAT_UINT, sizeof(DR_UINT), "%u"},
    {"LONG", DR_FORMAT_LONG, sizeof(DR_LONG), "%ld"},
    {"ULONG", DR_FORMAT_ULONG, sizeof(DR_ULONG), "%lu"},
    {"HALF", DR_FORMAT_HALF, sizeof(DR_HALF), NULL},
    {"FLOAT", DR_FORMAT_FLOAT, sizeof(DR_FLOAT), "%f"},
    {"DOUBLE", DR_FORMAT_DOUBLE, sizeof(DR_DOUBLE), "%lf"}
};

static struct {
    char *tag;
    unsigned int value;
}
datRawGridTypes[] = {
    {"EQUIDISTANT", DR_GRID_CARTESIAN},
    {"CARTESIAN", DR_GRID_CARTESIAN},
    {"UNIFORM", DR_GRID_CARTESIAN},
    {"RECTILINEAR", DR_GRID_RECTILINEAR},
    {"TETRAHEDRA", DR_GRID_TETRAHEDRAL}
};

#define DATRAW_GET_VALUE_FROM_TAG(VAL, LIST, TAG) \
{ \
    int __i; \
    for (__i = 0; __i < sizeof(LIST)/sizeof((LIST)[0]); __i++) { \
    if (!strcmp((LIST)[__i].tag, TAG)) { \
    (VAL) = (LIST)[__i].value; \
    break; \
    } \
    } \
}

#define DATRAW_GET_TAG_FROM_VALUE(TAG, LIST, VAL) \
{ \
    int __i; \
    for (__i = 0; __i < sizeof(LIST)/sizeof((LIST)[0]); __i++) { \
    if ((LIST)[__i].value == (VAL)) { \
    (TAG) = (LIST)[__i].tag; \
    break; \
    } \
    } \
}
#define DATRAW_GET_TAG_FROM_VALUE_FUNC(LIST) \
    const char* datRaw_get##LIST##Name(int val)\
{\
    char *__name = "UNKNOWN";\
    DATRAW_GET_TAG_FROM_VALUE(__name, datRaw##LIST##s, val)\
    return __name;\
}

DATRAW_GET_TAG_FROM_VALUE_FUNC(DataFormat)
DATRAW_GET_TAG_FROM_VALUE_FUNC(GridType)

static int datRaw_getByteOrder()
{
    DR_USHORT word = 0x0001;
    DR_UCHAR *byte = (DR_UCHAR*)&word;
    return *byte ? DR_LITTLE_ENDIAN : DR_BIG_ENDIAN;
}

static void swapByteOrder64(void *data, size_t size) 
{
    size_t i;
    DR_ULONG v, sv;
    DR_ULONG *idata = (DR_ULONG*)data;
    for (i = 0; i < size; i++) {
        v = idata[i];
        sv = (v & 0x00000000000000FFULL);
        sv = ((v & 0x000000000000FF00ULL) >> 0x08) | (sv << 0x08);
        sv = ((v & 0x0000000000FF0000ULL) >> 0x10) | (sv << 0x08);
        sv = ((v & 0x00000000FF000000ULL) >> 0x18) | (sv << 0x08);
        sv = ((v & 0x000000FF00000000ULL) >> 0x20) | (sv << 0x08);
        sv = ((v & 0x0000FF0000000000ULL) >> 0x28) | (sv << 0x08);
        sv = ((v & 0x00FF000000000000ULL) >> 0x30) | (sv << 0x08);
        sv = ((v & 0xFF00000000000000ULL) >> 0x38) | (sv << 0x08);
        idata[i] = sv;
    }
}

static void swapByteOrder32(void *data, size_t size) 
{
    size_t i;
    DR_UINT v, sv;
    DR_UINT *idata = (DR_UINT*)data;
    for (i = 0; i < size; i++) {
        v = idata[i];
        sv = (v & 0x000000FF);
        sv = ((v & 0x0000FF00) >> 0x08) | (sv << 0x08);
        sv = ((v & 0x00FF0000) >> 0x10) | (sv << 0x08);
        sv = ((v & 0xFF000000) >> 0x18) | (sv << 0x08);
        idata[i] = sv;
    }
}

static void swapByteOrder16(void *data, size_t size) 
{
    size_t i;
    DR_USHORT v, sv;
    DR_USHORT *idata = (DR_USHORT*)data;
    for (i = 0; i < size; i++) {
        v = idata[i];
        sv = (v & 0x00FF);
        sv = ((v & 0xFF00) >> 0x08) | (sv << 0x08);
        idata[i] = sv;
    }
}

static void swapByteOrder(void *data, size_t size, int type)
{
    switch (type) {
        case DR_FORMAT_CHAR:
        case DR_FORMAT_UCHAR:
            break;
        case DR_FORMAT_SHORT:
        case DR_FORMAT_USHORT:
        case DR_FORMAT_HALF:
            swapByteOrder16(data, size);
            break;
        case DR_FORMAT_INT:
        case DR_FORMAT_UINT:
        case DR_FORMAT_FLOAT:
            swapByteOrder32(data, size);
            break;
        case DR_FORMAT_DOUBLE:
            swapByteOrder64(data, size);
            break;
        default:
            datRaw_logError("Unknow data type specified\n");
            break;
    }
}

int datRaw_getFormatSize(int format)
{
    int i;
    for (i = 0; i < sizeof(datRawDataFormats)/sizeof(datRawDataFormats[0]); i++) {
        if (datRawDataFormats[i].value == format) {
            return datRawDataFormats[i].size;
        }
    }
    datRaw_logError("Unknown data format %d\n", format);
    return 0;
}

static const char* datRaw_getFormatString(int format)
{
    int i;
    for (i = 0; i < sizeof(datRawDataFormats)/sizeof(datRawDataFormats[0]); i++) {
        if (datRawDataFormats[i].value == format) {
            return datRawDataFormats[i].formatString;
        }
    }
    datRaw_logError("Unknown data format %d\n", format);
    return NULL;
}

static char *datRaw_makeupper(char *s, char delim)
{
    char *tmp = s;
    if (!tmp) {
        return NULL;
    }
    while (*tmp != delim && *tmp != '\0') {
        *tmp = toupper(*tmp);
        tmp++;
    }
    return *tmp == '\0' ? NULL : tmp;
}


static void datRaw_isolateLine(char **buf, const int maxBuf,
        char **outEndPos, char *outEndChar) {
    char *p = *buf;
    int c = 0;

    while ((c < maxBuf) && *p && (*p != '\n') && (*p != '\r')) {
        c++;
        p++;
    }

    *outEndPos = p;
    *outEndChar = *p;

    if (c != maxBuf) {
        /* proceed to next line */
        if (*p == '\n') {
            /* unix */
            p++; 
        }
        else if (*p == '\r') {
            /* dos or mac */
            p++;
            if (*p == '\n') {
                /* dos */
                p++;
            }
        }
        *buf = p;
        **outEndPos = '\0';
    } else {
        *buf = NULL;
    }
}

static int isMultifileDescription(const char *filename)
{
    const char *s = filename;
    while (*s) {
        if (s[0] == '%' && s[1] != '%') {
            return 1;
        }
        s++;
    }
    return 0;
}

static char *parseMultifileDescription(const char *filename, int *width,
                                       int *skip, int *stride) 
{
    char *p, *s, *q;

    if (!(p = dupstr(filename))) {
        datRaw_logError("DatRaw: Failed to allocate file description string\n");
        return 0;
    }

    s = NULL;
    q = p;
    while (*q) {
        if (q[0] == '%' && q[1] != '%') {
            if (s) {
                datRaw_logError("DatRaw: Multi file description contains more than "
                    "one varying\n");
                return NULL;
            } else {
                s = q;
            }
        }
        q++;
    }
    s++;

    if (!s || *s == '\0') {
        datRaw_logError("DatRaw: Strange input dataFileName: %s\n",
            filename);
        free(p);
        return NULL;
    }

    if (*s == '0' || *s == '-' || *s == ' ') {
        s++;
    }
    *width = 0;
    while (isdigit(*s)) {
        *width = *width*10 + (*s - '0');
        s++;
    }
    *skip = 0;
    if (*s == '+') {
        q = s;
        s++;
        if (!isdigit(*s)) {
            datRaw_logError("DatRaw: Error in multi file description.\n"
                "Invalid skip in '%s'", filename);
            free(p);
            return NULL;
        }
        while (isdigit(*s)) {
            *skip = *skip*10 + (*s - '0');
            s++;
        }
        memmove(q, s, strlen(s) + 1);
        s = q;
    }
    *stride = 0;
    if (*s == '*') {
        q = s;
        s++;
        if (!isdigit(*s)) {
            datRaw_logError("DatRaw: Error in multi file description.\n"
                "Invalid stride in '%s'", filename);
            free(p);
            return NULL;
        }
        while (isdigit(*s)) {
            *stride = *stride*10 + (*s - '0');
            s++;
        }
        memmove(q, s, strlen(s) + 1);
        s = q;
    } else {
        *stride = 1;
    }
    if (*s != 'd') {
        datRaw_logError("DatRaw: Error in multi file description '%s'\n", filename);
        free(p);
        return NULL;
    }

    return p;
}

char *getMultifileFilename(DatRawFileInfo *info, int timeStep)
{
    int maxFilename, minWidth, offset, stride;
    char *filenameTemplate, *filename;

    filenameTemplate = parseMultifileDescription(info->dataFileName, &minWidth, &offset, &stride);

    if (!filenameTemplate) {
        return 0;
    }

    maxFilename = (int)strlen(filenameTemplate) +
        DATRAW_MAX((int)log10(info->timeSteps), minWidth) + 2;

    if (!(filename = (char*)malloc(maxFilename * sizeof(char)))) {
        datRaw_logError("DatRaw: Failed to allocate memory for filename\n");
        return 0;
    }

    sprintf(filename, filenameTemplate, offset + stride*timeStep);

    free(filenameTemplate);

    return filename;
}

static int datRaw_parseHeaderFile(
                                  const char *datfile,
                                  DatRawFileInfo *info,
                                  DatRawOptionalField **optionalFields)
{
    int fd;
    char *inputLine;
    char buf[256];
    char *datFileText;
    char *input, *sep, *s1, *s2, *p;
    char *endPos, endChar;
    char *resolutionLine = NULL;
    char *originLine = NULL;
    int lineCount = 0;
    int error  = 0;
    int i;
    int axis;
    size_t size;
#ifdef _WIN32
    struct _stat fstatus;
#else
    struct stat fstatus;
#endif
    if (!info) {
        datRaw_logError("DatRaw: NULL description record!\n");
        return 0;
    }

    /* 
    set defaults 
    */
    datRaw_initHeader(info);

    if (!(info->descFileName = dupstr(datfile))) {
        datRaw_logError("DatRaw: Failed to allocate memory for description record!\n");
        return 0;
    }

#ifdef _WIN32
    if ((fd = _open(datfile, O_RDONLY | O_BINARY)) == -1) {
#else
    if ((fd = open(datfile, O_RDONLY)) == -1) {
#endif
        perror("DatRaw: Could not open header file");
        return 0;
    }

#ifdef _WIN32
    if (_fstat(fd, &fstatus)) {
#else
    if (fstat(fd, &fstatus)) {
#endif
        perror("DatRaw: Failed to get header file properties");
        return 0;
    }

    if (! (datFileText = (char*)malloc(fstatus.st_size + 1))) {
        datRaw_logError("DatRaw: Failed to allocate memory for header file text!\n");
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return 0;
    }

#ifdef _WIN32
    if (_read(fd, datFileText, fstatus.st_size) != fstatus.st_size) {
#else
    if (read(fd, datFileText, fstatus.st_size) != fstatus.st_size) {
#endif
        perror("DatRaw: Could not read header file");
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return 0;
    }

#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif

    /* Ensure that the text is zero-terminated. */
    datFileText[fstatus.st_size] = 0;

    /*
    * independent fields
    */
    input = datFileText;
    while (input) {

        lineCount++;

        inputLine = input;
        datRaw_isolateLine(&input, fstatus.st_size - (input - datFileText),
            &endPos, &endChar);

        /* remove comments */
        if ((p = strchr(inputLine, '#'))) {
            *p = '\0';
        }

        /* change everything preceding the separator to upper case */
        if (!(sep = datRaw_makeupper(inputLine, ':'))) {

            /* no ':' in input line check for whitespace else exit */
            p = inputLine;
            while (*p != '\0') {
                if (!isspace(*p)) {
                    datRaw_logError("DatRaw: Error reading header file: %s\nLine %d: %s...\n", datfile, lineCount, inputLine);
                    return 0;
                }
                p++;
            }

            if (*endPos != endChar) {
                /* Restore original text if datRaw_isolateLine changed it. */
                *endPos = endChar;
            }

            continue;
        }

        if (strstr(inputLine, "OBJECTFILENAME")) {
            if (info->dataFileName != NULL) {
                datRaw_logError("DatRaw Error: Multiple data files specified\n");
                error = 1;
            } else {
                /* eat whitespace */
                sep++;
                while (sep && *sep && isspace(*sep)) sep++;
                if (!sep || !*sep) {
                    datRaw_logError("DatRaw Error: Invalid data filename!\n");
                    error = 1;
                } else {
                    if (sscanf(sep, "%[^\n]\n", buf) != 1) {
                        error = 1;
                    } else {
                        if (isMultifileDescription(buf)) {
                            info->multiDataFiles = 1;
                        }
                        if (buf[0] == '/') {
                            if (!(info->dataFileName = dupstr(buf))) {
                                datRaw_logError("DatRaw: Failed to allocate memory for description record!\n");
                                return 0;
                            }
                        } else {
                            if (!(s2 = dupstr(info->descFileName))) {
                                datRaw_logError("DatRaw: Failed to allocate memory for description record!\n");
                                return 0;
                            }
                            s1 = dirname(s2);
                            if (!(info->dataFileName =
                                (char*)malloc(strlen(s1)+strlen(buf) + 2))) {
                                    datRaw_logError("DatRaw: Failed to allocate memory for description record!\n");
                                    return 0;
                            }
                            strcpy(info->dataFileName, s1);
                            strcat(info->dataFileName, "/");
                            strcat(info->dataFileName, buf);
                            free(s2);
                        }
                    }
                }
            }
        } else if (strstr(inputLine, "FORMAT")) {
            if (info->dataFormat != DR_FORMAT_NONE) {
                datRaw_logError("DatRaw Error: Multiple data formats found\n");
                error = 1;
            } else if (sscanf(sep+1, "%s", buf) != 1) {
                error = 1;
            } else {
                datRaw_makeupper(buf, '\0');
                DATRAW_GET_VALUE_FROM_TAG(info->dataFormat, datRawDataFormats, buf)
                    error = (info->dataFormat == DR_FORMAT_NONE);
            }
        } else if (strstr(inputLine, "GRIDTYPE")) {
            if (info->gridType != DR_GRID_NONE) {
                datRaw_logError("DatRaw Error: Multiple grid types found\n");
                error = 1;
            } else if (sscanf(sep+1, "%s", buf) != 1) {
                error = 1;
            } else {
                datRaw_makeupper(buf, '\0');
                DATRAW_GET_VALUE_FROM_TAG(info->gridType, datRawGridTypes, buf)
                    error = (info->gridType == DR_GRID_NONE);
            }
        } else if (strstr(inputLine, "COMPONENTS")) {
            if (info->dimensions != -1) {
                datRaw_logError("DatRaw Error: # of components given multiple times\n");
                error = 1;
            } else if (sscanf(sep+1, "%d", &info->numComponents) != 1) {
                error = 1;
            }
        } else if (strstr(inputLine, "DIMENSIONS")) {
            if (info->dimensions != -1) {
                datRaw_logError("DatRaw Error: Multiple dimensions given\n");
                error = 1;
            } else if (sscanf(sep+1, "%d", &info->dimensions) != 1) {
                error = 1;
            }
        } else if (strstr(inputLine, "TIMESTEPS")) {
            if (info->timeSteps!= -1) {
                datRaw_logError("DatRaw Error: Multiple time steps given\n");
                error = 1;
            } else if (sscanf(sep+1, "%d", &info->timeSteps) != 1) {
                error = 1;
            }
        } else if (strstr(inputLine, "BYTEORDER")) { 
            if (info->byteOrder != -1) {
                datRaw_logError("DatRaw Error: Multiple byte orders given\n");
                error = 1;
            } else if (sscanf(sep+1, "%s", buf) != 1) {
                error = 1;
            } else {
                datRaw_makeupper(buf, '\0');
                if (!strcmp(buf, "BIG_ENDIAN")) {
                    info->byteOrder = DR_BIG_ENDIAN;
                } else if (!strcmp(buf, "LITTLE_ENDIAN")) {
                    info->byteOrder = DR_LITTLE_ENDIAN;
                } else {
                    error = 1;
                }
            }
        } else if (strstr(inputLine, "DATAOFFSET")) {
            if (info->dataOffset != -1) {
                datRaw_logError("DatRaw Error: Multiple data offsets given\n");
                error = 1;
            } else if (sscanf(sep+1, "%d", &info->dataOffset) != 1) {
                error = 1;
            }
        } else if (strstr(inputLine, "RESOLUTION")) {
            /*
             * Remember the resolution line, because we need to know it before
             * we can allocate sliceDist in case of a rectilinear grid.
             */
            if (resolutionLine == NULL) {
                resolutionLine = sep + 1;
            } else {
                datRaw_logError("DatRaw Error: Multiple resolution lines given\n");
                error = 1;
            }
        } else if (strstr(inputLine, "SLICETHICKNESS")) {
        } else if (strstr(inputLine, "ORIGIN")) {
            /*
             * Remember the origin line, because we need to know it before
             * we can allocate sliceDist in case of a rectilinear grid.
             */
            if (originLine == NULL) {
                originLine = sep + 1;
            } else {
                datRaw_logError("DatRaw Error: Multiple origin lines given\n");
                error = 1;
            }
        } else {
            datRaw_logInfo("DatRaw: Datfile - line %d ignored: %s...\n", lineCount, inputLine);
        }

        if (*endPos != endChar) {
            /* Restore original text if datRaw_isolateLine changed it. */
            *endPos = endChar;
        }

        if (error) {
            datRaw_logError("DatRaw: Error reading header file: %s\nLine %d: %s...\n", datfile, lineCount, inputLine);
            return 0;
        }
    }

    /* fatal errors */
    if (info->dataFileName == NULL) {
        datRaw_logError("DatRaw: Error reading header file: %s\nRaw data file missing!\n", datfile);
        return 0;
    }

    if (info->dataFormat == DR_FORMAT_NONE) {
        datRaw_logError("DatRaw: Error reading header file: %s\nData format missing!\n", datfile);
        return 0;
    }

    /* warnings */
    if (info->numComponents == 0) {
        datRaw_logWarning("DatRaw Warning: Number of components not given in %s\n-> assuming 1 (scalar data)\n", datfile);
        info->numComponents = 1;
    }

    if (info->byteOrder == -1) {
        if (datRaw_getFormatSize(info->dataFormat) > 1) {
            datRaw_logWarning("DatRaw Warning: Byte order of data not given in %s\n-> assuming LITTLE ENDIAN\n",
                datfile);
        }
        info->byteOrder = DR_LITTLE_ENDIAN;
    }

    if (info->gridType == DR_GRID_NONE) {
        datRaw_logWarning("DatRaw Warning: Grid type not given in %s\n-> assuming CARTESIAN!\n", datfile);
        info->gridType = DR_GRID_CARTESIAN;
    }

    if (info->timeSteps == -1) {
        datRaw_logWarning("DatRaw Warning: Number of time steps not given in %s\n-> assuming 1\n", datfile);
        info->timeSteps = 1;
    }

    if (info->dimensions == -1) {
        datRaw_logWarning("DatRaw: Number of dimensions missing in %s\n-> assuming 3\n", datfile);
        info->dimensions = 3;
    }

    if (info->dataOffset == -1) {
        datRaw_logWarning("DatRaw: Dataoffset missing in %s\n-> assuming 0\n", datfile);
        info->dataOffset = 0;
    }

    if (!(info->resolution = (int*)malloc(info->dimensions * sizeof(int)))) {
        datRaw_logError("DatRaw: Failed to allocate memory for description record!\n");
            return 0;
    }

    /*
     * Allocate resolution. This is a special case of a dependent file because
     * of rectilinear grids.
     */
    if (resolutionLine != NULL) {
        endPos = resolutionLine;
        while ((*endPos != 0) && (*endPos != '\r') && (*endPos != '\n')) {
            ++endPos;
        }
        endChar = *endPos;
        *endPos = 0;
        i = 0;
        p = s1 = dupstr(resolutionLine);
        *endPos = endChar;
        if (!p) {
            datRaw_logError("DatRaw: Error parsing dat-file - Failed to allocate temp. storage!\n");
            return 0;
        }
        /*while ((s2 = strsep(&s1," \t\n"))) {*/ /* not ansi :-( */
        while ((s2 = strtok(s1, " ,\t\n"))) {
            if (*s2 == '\0') {
                continue;
            }
            if (i >= info->dimensions) {
                error = 1;
                break;
            }
            if (sscanf(s2, "%d", &info->resolution[i]) != 1) {
                error = 1;
                break;
            }
            i++;
            s1 = NULL;
        }
        free(p);
    }

    /*
     * Allocate origin
     */
    if (!(info->origin = (float*)malloc(info->dimensions * sizeof(int)))) {
        datRaw_logError("DatRaw: Failed to allocate memory for description record!\n");
        return 0;
    }
    for (i = 0; i < info->dimensions; i++) {
        info->origin[i] = 0.0f;
    }

    if (originLine != NULL) {
        endPos = originLine;
        while ((*endPos != 0) && (*endPos != '\r') && (*endPos != '\n')) {
            ++endPos;
        }
        endChar = *endPos;
        *endPos = 0;
        i = 0;
        p = s1 = dupstr(originLine);
        *endPos = endChar;
        if (!p) {
            datRaw_logError("DatRaw: Error parsing dat-file - Failed to allocate temp. storage!\n");
            return 0;
        }
        /*while ((s2 = strsep(&s1," \t\n"))) {*/ /* not ansi :-( */
        while ((s2 = strtok(s1, " ,\t\n"))) {
            if (*s2 == '\0') {
                continue;
            }
            if (i >= info->dimensions) {
                error = 1;
                break;
            }
            if (sscanf(s2, "%f", &info->origin[i]) != 1) {
                error = 1;
                break;
            }
            i++;
            s1 = NULL;
        }
        free(p);
    }

    if (info->gridType == DR_GRID_RECTILINEAR) {
        if (resolutionLine == NULL) {
            datRaw_logError("DatRaw: The volume resolution is a required parameter for rectilinear grids!\n");
            return 0;
        }

        size = 0;
        for (i = 0; i < info->dimensions; ++i) {
            size += info->resolution[i];
        }
        if (!(info->sliceDist = (float *) malloc(size * sizeof(float)))) {
            datRaw_logError("DatRaw: Failed to allocate memory for description record!\n");
            return 0;
        }

        for (i = 0; i < size; ++i) {
            info->sliceDist[i] = -1.0f;
        }

    } else {
        if (!(info->sliceDist = (float*)malloc(info->dimensions * sizeof(float)))) {
            datRaw_logError("DatRaw: Failed to allocate memory for description record!\n");
            return 0;
        }

        for (i = 0; i < info->dimensions; i++) {
            info->sliceDist[i] = -1.0f;
        }
    }


    /*
     * dependent fields
     */
    lineCount = 0;
    input = datFileText;
    while (input) {

        lineCount++;

        inputLine = input;
        datRaw_isolateLine(&input, fstatus.st_size - (input - datFileText),
            &endPos, &endChar);

        /* remove comments */
        if ((p = strchr(inputLine, '#'))) {
            *p = '\0';
        }

        /* change everything preceding the separator to upper case */
        if (!(sep = datRaw_makeupper(inputLine, ':'))) {
            /* This can be only whitespace, otherwise we already exited above */
            if (*endPos != endChar) {
                /* Restore original text if datRaw_isolateLine changed it. */
                *endPos = endChar;
            }
            continue;
        }

        switch(info->gridType) {
        case DR_GRID_CARTESIAN:
            if (strstr(inputLine, "SLICETHICKNESS")) {
                if (info->sliceDist[0] != -1.0f) {
                    datRaw_logError("DatRaw Error: Multiple slice distance lines given\n");
                    error = 1;
                } else {
                    i = 0;
                    p = s1 = dupstr(sep + 1);
                    if (!p) {
                        datRaw_logError("DatRaw: Error parsing dat-file - Failed to allocate temp. storage!\n");
                        return 0;
                    }
                    /*while ((s2 = strsep(&s1," \t\n"))) {*/ /* not ansi :-( */
                    while ((s2 = strtok(s1, " ,\t\n"))) {
                        if (*s2 == '\0') {
                            continue;
                        }
                        if (i >= info->dimensions) {
                            error = 1;
                            break;
                        }
                        if (sscanf(s2, "%f", &info->sliceDist[i]) != 1) {
                            error = 1;
                            break;
                        }
                        i++;
                        s1 = NULL;
                    }
                    free(p);
                }
            } 
            break;

        case DR_GRID_RECTILINEAR:
            if (strstr(inputLine, "SLICETHICKNESS")) {
                if (sscanf(inputLine, "SLICETHICKNESS[%d]", &axis) != 1) {
                    datRaw_logError("SLICKETHICKNESS must be subscripted with the axis in case of a RECTILINEAR grid!\n");
                    return 0;
                }

                size = 0;
                for (i = 0; i < axis; ++i) {
                    size += info->resolution[i];
                }

                if (info->sliceDist[size] != -1.0f) {
                    datRaw_logError("DatRaw Error: Multiple slice distance lines given\n");
                    error = 1;
                } else {
                    i = 0;
                    p = s1 = dupstr(sep + 1);
                    if (!p) {
                        datRaw_logError("DatRaw: Error parsing dat-file - Failed to allocate temp. storage!\n");
                        return 0;
                    }
                    /*while ((s2 = strsep(&s1," \t\n"))) {*/ /* not ansi :-( */
                    while ((s2 = strtok(s1, " ,\t\n"))) {
                        if (*s2 == '\0') {
                            continue;
                        }
                        if (i >= info->resolution[axis]) {
                            error = 1;
                            break;
                        }
                        if (sscanf(s2, "%f", &info->sliceDist[size + i]) != 1) {
                            error = 1;
                            break;
                        }
                        i++;
                        s1 = NULL;
                    }
                    free(p);
                }
            }
            break;

        case DR_GRID_TETRAHEDRAL:
            if (strstr(inputLine, "VERTICES")) {
                if (sscanf(sep+1, "%d", &info->numVertices) != 1) {
                    error = 1;
                }
            } else if (strstr(inputLine, "TETRAHEDRA")) {
                if (sscanf(sep+1, "%d", &info->numTetrahedra) != 1) {
                    error = 1;
                }
            }
            break;
        default:
            datRaw_logError("DatRaw: Error in reading header file %s\nUnknown Grid type!", datfile);
            break;
        }


        if (optionalFields) {
            int n = 0;
            for (; optionalFields[n]; ++n) {
                DatRawOptionalField* pOptionalField = optionalFields[n];
                int elementSize = datRaw_getFormatSize(pOptionalField->format);
                const char* formatString = datRaw_getFormatString(pOptionalField->format);
                /* sanity checks */
                if (!pOptionalField->name ||
                    elementSize <= 0 ||
                    pOptionalField->numComponents <= 0)
                {
                    datRaw_logError("DatRaw Error: Optional data field "
                        "description %d invalid\n", n);
                    error = 1;
                    break;
                }

                if (strstr(inputLine, pOptionalField->name)) {
                    int numElements = 0;
                    if (pOptionalField->data) {
                        datRaw_logError("DatRaw Error: Multiple ""%s"" lines "
                            "given or optional data field description not "
                            "correctly initialized \n", pOptionalField->name);
                        error = 1;
                        break;
                    }

                    numElements = pOptionalField->numComponents;
                    if (pOptionalField->timeDependent) {
                        numElements *= info->timeSteps;
                    }

                    if (!(pOptionalField->data = malloc(numElements*elementSize))) {
                        datRaw_logError("DatRaw: Failed to allocate memory for description record!\n");
                        return 0;
                    }

                    i = 0;
                    p = s1 = dupstr(sep + 1);
                    if (!p) {
                        datRaw_logError("DatRaw: Error parsing dat-file - Failed to allocate temp. storage!\n");
                        return 0;
                    }
                    /*while ((s2 = strsep(&s1," \t\n"))) {*/ /* not ansi :-( */
                    while ((s2 = strtok(s1, " ,\t\n"))) {
                        if (*s2 == '\0') {
                            continue;
                        }

                        if (i >= numElements) {
                            error = 1;
                            break;
                        }

                        if (sscanf(s2, formatString,
                            (char*)pOptionalField->data + i*elementSize) != 1) {
                                error = 1;
                                break;
                        }
                        i++;
                        s1 = NULL;
                    }
                    free(p);
                }
            }
            if (error)
            {
                break;
            }
        }

        if (*endPos != endChar) {
            /* Restore original text if datRaw_isolateLine changed it. */
            *endPos = endChar;
        }

        if (error) {
            datRaw_logError("DatRaw: Error in reading header file %s\nLine %d: %s...\n", 
                datfile, lineCount, inputLine);
            return 0;
        }
    }

    free(datFileText);

    /* warnings and errors */
    switch(info->gridType) {
    case DR_GRID_CARTESIAN:
    case DR_GRID_RECTILINEAR:
        for (i = 0; i < info->dimensions; i++) {
            if (info->sliceDist[i] <= 0.0) {
                datRaw_logWarning("DatRaw Warning: %d. slice distance in %s invalid\n-> set to 1! \n", i, datfile);
                info->sliceDist[i] = 1.f;
            }
            if (info->resolution[i] <= 0) {
                datRaw_logError("DatRaw Error: resolution for %d. axes in %s invalid\n! \n", i, datfile);
                return 0;
            }
        }
        break;
    case DR_GRID_TETRAHEDRAL:
        if (info->numVertices <= 0) {
            datRaw_logWarning("DatRaw: Number of vertices missing or invalid in %s\n-> assuming 0\n", datfile);
            info->numVertices = 0;
        }
        if (info->numTetrahedra <= 0) {
            datRaw_logWarning("DatRaw: Number of simplices missing or invalid in %s\n-> assuming 0\n", datfile);
            info->numTetrahedra = 0;
        }
        break;
    default:
        break;
    }
    return 1;
}

void datRaw_initHeader(DatRawFileInfo *info) {
    if (info != NULL) {
        info->timeSteps = -1;
        info->descFileName = NULL;
        info->dataFileName = NULL;
        info->multiDataFiles = 0;
        info->dimensions = -1;
        info->gridType = DR_GRID_NONE;
        info->numComponents = 0;
        info->dataFormat = DR_FORMAT_NONE;
        /* only for cartesian and rectilinear grids */
        info->resolution = NULL;
        info->sliceDist = NULL;
        info->origin = NULL;
        /* only for tetrahedral grids */
        info->numVertices = -1;
        info->numTetrahedra = -1;
        info->byteOrder = -1;
        info->dataOffset = -1;

        info->fd_dataFile = NULL;
        info->currentStep = -1;
    }
}

int datRaw_readHeader(
                      const char *file,
                      DatRawFileInfo *info,
                      DatRawOptionalField **optionalFields)
{
    return datRaw_parseHeaderFile(file, info, optionalFields);    
}

void datRaw_printInfo(const DatRawFileInfo *info)
{
    char *grid = "NOT SET";
    char *format = "NOT SET";
    int i, j, k;

    if (!info) {
        return;
    }

    DATRAW_GET_TAG_FROM_VALUE(grid, datRawGridTypes, info->gridType)
        DATRAW_GET_TAG_FROM_VALUE(format, datRawDataFormats, info->dataFormat)

        fprintf(stdout,
        "Dat File info:\n"
        "Header File: %s\n"
        "Data File  : %s\n"
        "MultiFile  : %s\n"
        "Grid       : %s\n"
        "Dimensions : %d\n"
        "Steps      : %d\n"
        "Components : %d\n"
        "Format     : %s\n",
        info->descFileName,
        info->dataFileName,
        info->multiDataFiles ? "yes" : "no",
        grid,
        info->dimensions,
        info->timeSteps,
        info->numComponents,
        format);

    switch (info->gridType) {
    case DR_GRID_CARTESIAN:
        fprintf(stdout, "SliceDist  : ");
        for (i = 0; i < info->dimensions; i++) {
            fprintf(stdout, "%.4f ", info->sliceDist[i]);
        }
        fprintf(stdout, "\n");
        fprintf(stdout, "Resolution : ");
        for (i = 0; i < info->dimensions; i++) {
            fprintf(stdout, "%d ", info->resolution[i]);
        }
        fprintf(stdout, "\n");
        break; 
    case DR_GRID_RECTILINEAR:
        for (i = 0, j = 0; i < info->dimensions; ++i) {
            fprintf(stdout, "SliceDist[%d]: ", i);
            for (k = 0; k < info->resolution[i]; ++j, ++k) {
                fprintf(stdout, "%.4f ", info->sliceDist[j]);
            }
            fprintf(stdout, "\n");
        }
        fprintf(stdout, "Resolution : ");
        for (i = 0; i < info->dimensions; i++) {
            fprintf(stdout, "%d ", info->resolution[i]);
        }
        fprintf(stdout, "\n");
        break;
    case DR_GRID_TETRAHEDRAL:
        fprintf(stdout, "Vertices   : %d\nTetrahedra : %d\n",
            info->numVertices, info->numTetrahedra);
        break;
    default:
        break;
    }
}

#define DAT_RAW_CONVERT_BLOCK(SRC_FORMAT, DST_FORMAT) \
{ \
    long int j; \
    DR_##SRC_FORMAT *s = (DR_##SRC_FORMAT *)src; \
    DR_##DST_FORMAT *d = (DR_##DST_FORMAT *)dst; \
    for (j = 0; j < count; j++) { \
    *d++ = (DR_##DST_FORMAT)*s++; \
    } \
}

#define DAT_RAW_CONVERT_TO_HALF_BLOCK(SRC_FORMAT) \
case (DR_FORMAT_HALF): \
{ \
    long int j; \
    DR_##SRC_FORMAT *s = (DR_##SRC_FORMAT *)src; \
    DR_HALF *d = (DR_HALF *)dst; \
    for (j = 0; j < count; j++) { \
    *d++ = floatToHalf((DR_FLOAT)*s++); \
    } \
} \
    break;

#define DAT_RAW_CONVERT_FROM_HALF_BLOCK(DST_FORMAT) \
case (DR_FORMAT_##DST_FORMAT): \
{ \
    long int j; \
    DR_HALF *s = (DR_HALF *)src; \
    DR_##DST_FORMAT *d = (DR_##DST_FORMAT *)dst; \
    for (j = 0; j < count; j++) { \
    *d++ = (DR_##DST_FORMAT)halfToFloat(*s++); \
    /*fprintf(stderr, "0H%x -> %f\n", *(s - 1), (float)*(d-1));*/ \
    } \
} \
    break;

#define DAT_RAW_CONVERT_DST_BLOCK(DST_FORMAT, SRC_FORMAT) \
case (DR_FORMAT_##DST_FORMAT): \
    DAT_RAW_CONVERT_BLOCK(SRC_FORMAT, DST_FORMAT) \
    break;

#define DAT_RAW_CONVERT_SRC_BLOCK(SRC_FORMAT) \
case (DR_FORMAT_##SRC_FORMAT): \
    switch(dstFormat) { \
    DAT_RAW_CONVERT_DST_BLOCK(CHAR, SRC_FORMAT) \
    DAT_RAW_CONVERT_DST_BLOCK(UCHAR, SRC_FORMAT) \
    DAT_RAW_CONVERT_DST_BLOCK(SHORT, SRC_FORMAT) \
    DAT_RAW_CONVERT_DST_BLOCK(USHORT, SRC_FORMAT) \
    DAT_RAW_CONVERT_DST_BLOCK(INT, SRC_FORMAT) \
    DAT_RAW_CONVERT_DST_BLOCK(UINT, SRC_FORMAT) \
    DAT_RAW_CONVERT_DST_BLOCK(LONG, SRC_FORMAT) \
    DAT_RAW_CONVERT_DST_BLOCK(ULONG, SRC_FORMAT) \
    DAT_RAW_CONVERT_DST_BLOCK(FLOAT, SRC_FORMAT) \
    DAT_RAW_CONVERT_DST_BLOCK(DOUBLE, SRC_FORMAT) \
    DAT_RAW_CONVERT_TO_HALF_BLOCK(SRC_FORMAT) \
    /* ADD NEW BASIC TYPES BEFORE THIS LINE !!!! */ \
        default: \
        datRaw_logError("Unknown dst datatype in datRaw_convertBlock1\n"); \
        break; \
} \
    break;

#define DAT_RAW_CONVERT_HALF_BLOCK \
case (DR_FORMAT_HALF): \
    switch(dstFormat) { \
    DAT_RAW_CONVERT_FROM_HALF_BLOCK(CHAR) \
    DAT_RAW_CONVERT_FROM_HALF_BLOCK(UCHAR) \
    DAT_RAW_CONVERT_FROM_HALF_BLOCK(SHORT) \
    DAT_RAW_CONVERT_FROM_HALF_BLOCK(USHORT) \
    DAT_RAW_CONVERT_FROM_HALF_BLOCK(INT) \
    DAT_RAW_CONVERT_FROM_HALF_BLOCK(UINT) \
    DAT_RAW_CONVERT_FROM_HALF_BLOCK(LONG) \
    DAT_RAW_CONVERT_FROM_HALF_BLOCK(ULONG) \
    DAT_RAW_CONVERT_FROM_HALF_BLOCK(FLOAT) \
    DAT_RAW_CONVERT_FROM_HALF_BLOCK(DOUBLE) \
    /* ADD NEW BASIC TYPES BEFORE THIS LINE !!!! */ \
        default: \
        datRaw_logError("Unknown dst datatype in datRaw_convertBlock2\n"); \
        break; \
} \
    break;

static void datRaw_convertBlock(DR_UCHAR *src, int srcFormat, DR_UCHAR *dst,
                                int dstFormat, int count)
{
    switch (srcFormat) {
        DAT_RAW_CONVERT_SRC_BLOCK(CHAR)
            DAT_RAW_CONVERT_SRC_BLOCK(UCHAR)
            DAT_RAW_CONVERT_SRC_BLOCK(SHORT)
            DAT_RAW_CONVERT_SRC_BLOCK(USHORT)
            DAT_RAW_CONVERT_SRC_BLOCK(INT)
            DAT_RAW_CONVERT_SRC_BLOCK(UINT)
            DAT_RAW_CONVERT_SRC_BLOCK(LONG)
            DAT_RAW_CONVERT_SRC_BLOCK(ULONG)
            DAT_RAW_CONVERT_SRC_BLOCK(FLOAT)
            DAT_RAW_CONVERT_SRC_BLOCK(DOUBLE)
            DAT_RAW_CONVERT_HALF_BLOCK
            /* ADD NEW BASIC TYPES BEFORE THIS LINE !!!! */
        default:
            datRaw_logError("Unknown src datatype in datRaw_convertBlock\n");
            break;
    }
}

size_t datRaw_getRecordSize(const DatRawFileInfo *info, int format)
{
    size_t dataRecSize = 1;

    if (info) {

        if (format == DR_FORMAT_RAW) {
            dataRecSize *= datRaw_getFormatSize(info->dataFormat);
        } else {
            dataRecSize *= datRaw_getFormatSize(format);
        }

        if (dataRecSize == 0) {
            return 0;
        }

        dataRecSize *= info->numComponents;

        if (dataRecSize == 0) {
            return 0;
        }

        return dataRecSize;
    }
    return 0;
}

size_t datRaw_getElementCount(const DatRawFileInfo *info)
{
    int i;
    size_t size;

    switch (info->gridType) {
    case DR_GRID_CARTESIAN:
    case DR_GRID_RECTILINEAR:
        size = 1;
        for (i = 0; i < info->dimensions; i++) {
            size *= info->resolution[i];
        }
        break;
    case DR_GRID_TETRAHEDRAL:
        size = 0; 
        datRaw_logError("DatRaw: tetrahedral grids not implemented, yet.\n");
        break;
    default:
        datRaw_logError("DatRaw Error: Unknow grid type %d\n", info->gridType);
        size = 0;
        break;
    }
    return size;
}

size_t datRaw_getBufferSize(const DatRawFileInfo *info, int format)
{
    size_t size;

    if (info) {

        size = datRaw_getRecordSize(info, format);

        if (size <= 0) {
            return 0;
        }

        size *= datRaw_getElementCount(info);

        if (size <= 0) {
            return 0;
        }

        return size;
    }
    return 0;
}

int datRaw_load(const char *datfile,
                DatRawFileInfo *info,
                DatRawOptionalField **optionalFields,
                void **buffer,
                int format)
{
    size_t memSize;
    size_t storageSize;
    int i, offset, stride, allocateMem;
    void *tmpBuffer = NULL;
    void *buf = NULL;
    char *filename = NULL, *filenameTemplate = NULL;

    /* read header */
    if(!datRaw_parseHeaderFile(datfile, info, optionalFields)) {
        return 0;
    };

    if (!buffer) {
        return 0;
    }
    allocateMem = !(*buffer);

    storageSize = datRaw_getBufferSize(info, info->dataFormat);
    memSize = datRaw_getBufferSize(info, format);

    if (memSize <= 0 || storageSize <= 0) {
        return 0;
    }

    if (allocateMem && !(*buffer = malloc(memSize * info->timeSteps))) {
        datRaw_logError("DatRaw Error: Could not allocate buffer memory (%ld byte)\n", memSize * info->timeSteps);
        return 0;
    }

    if (!info->multiDataFiles) {
        info->fd_dataFile = gzopen(info->dataFileName, "rb");

        if (!info->fd_dataFile) {
            datRaw_logError("DatRaw: Error opening data file \"%s\"!\n", info->dataFileName);
            if (allocateMem) {
                free(*buffer);
                *buffer = NULL;
            }
            return 0;
        }
        if (info->dataOffset > 0 && gzseek(info->fd_dataFile, info->dataOffset,
            SEEK_SET) != info->dataOffset) {
                datRaw_logError("DatRaw: Error reading data file %s (skipping failed)!\n", info->dataFileName);
                if (allocateMem) {
                    free(*buffer);
                    *buffer = NULL;
                }
                gzclose(info->fd_dataFile);
                info->fd_dataFile = NULL;
                return 0;
        }
    } else {
        int maxFilename, minWidth; 
        filenameTemplate = parseMultifileDescription(info->dataFileName, &minWidth, &offset, &stride);

        maxFilename = (int)strlen(filenameTemplate) +
            DATRAW_MAX((int)log10(info->timeSteps), minWidth) + 2;

        if (!(filename = (char*)malloc(maxFilename * sizeof(char)))) {
            datRaw_logError("DatRaw: Failed to allocate memory for filename\n");
            if (allocateMem) {
                free(*buffer); 
                *buffer = NULL;
            }
            return 0;
        }
    }

    if (format == info->dataFormat || format == DR_FORMAT_RAW) {
        buf = *buffer;
    } else {
        if (!(tmpBuffer = malloc(storageSize))) {
            datRaw_logError("DatRaw Error: Could not allocate tmp. buffer memory (%ld byte)\n", storageSize);
            if (!info->multiDataFiles) {
                gzclose(info->fd_dataFile);
            }
            if (allocateMem) {
                free(*buffer);
                *buffer = NULL;
            }
            return 0;
        }
        buf = tmpBuffer;
    }

    for (i = 0; i < info->timeSteps; i++) {

        if (info->multiDataFiles) {
            sprintf(filename, filenameTemplate, offset + stride*i);

            info->fd_dataFile = gzopen(filename, "rb");

            if (!info->fd_dataFile) {
                datRaw_logError("DatRaw: Error opening data file \"%s\"!\n", filename);
                if (allocateMem) {
                    free(*buffer);
                    *buffer = NULL;
                }
                free(tmpBuffer);
                return 0;
            }

            if (info->dataOffset > 0 && gzseek(info->fd_dataFile, info->dataOffset,
                SEEK_SET) != info->dataOffset) {
                    datRaw_logError("DatRaw: Error reading data file %s (skipping failed)!\n", info->dataFileName);
                    if (allocateMem) {
                        free(*buffer);
                        *buffer = NULL;
                    }
                    free(tmpBuffer);
                    gzclose(info->fd_dataFile);
                    info->fd_dataFile = NULL;
                    return 0;
            }

        }

        if (gzread(info->fd_dataFile, buf, storageSize) != storageSize) {
            if (gzeof(info->fd_dataFile)) {
                datRaw_logError("DatRaw: Error reading data file %s (EOF reached)!\n", info->dataFileName);
            } 
            else {
                int err;
                gzerror(info->fd_dataFile, &err);
                if (err) {
                    datRaw_logError("DatRaw: Error reading data file %s\n", info->dataFileName);
                }
            }
            gzclearerr(info->fd_dataFile);
            if (allocateMem) {
                free(*buffer);
                *buffer = NULL;
            }
            free(tmpBuffer);
            gzclose(info->fd_dataFile);
            info->fd_dataFile = NULL;
            return 0;
        }
        if (info->byteOrder != datRaw_getByteOrder()) {
            swapByteOrder(buf, 
                datRaw_getElementCount(info)*info->numComponents,
                info->dataFormat);
        }
        if (format != info->dataFormat && format != DR_FORMAT_RAW) {
            datRaw_convertBlock(buf, info->dataFormat, (DR_UCHAR*)*buffer + i*memSize,
                format, datRaw_getElementCount(info)*info->numComponents);
        } else {
            buf = (DR_UCHAR*)buf + memSize;
        }

        if (info->multiDataFiles) {
            gzclose(info->fd_dataFile);
        }
    }

    if (format != info->dataFormat && format != DR_FORMAT_RAW) {
        free(tmpBuffer);
    }

    if (!info->multiDataFiles) {
        gzclose(info->fd_dataFile);
    } else {
        free(filename);
        free(filenameTemplate);
    }

    info->fd_dataFile = NULL;

    return 1;
}


/*
loads the data of the next time step into *buffer. If *buffer is NULL,
memory is allocated that fits the data size.
Returns -1, if there is no next timestep to load and 0, if an error occured.
*/
int datRaw_getNext(DatRawFileInfo *info, void **buffer, int format)
{
    size_t storageSize, memSize;
    DR_UCHAR *buf;
    int allocateMem;

    if (!buffer) {
        return 0;
    }
    allocateMem = !(*buffer);

    if (info->currentStep + 1 >= info->timeSteps) {
        return -1;
    }

    info->currentStep++;

    if (!info->multiDataFiles) {
        if (info->fd_dataFile == NULL && info->currentStep == 0) {
            if (!(info->fd_dataFile = gzopen(info->dataFileName, "rb"))) {
                datRaw_logError("DatRaw: Error opening data file \"%s\"!\n",
                    info->dataFileName);
                return 0;
            }
            if (info->dataOffset > 0 && gzseek(info->fd_dataFile, info->dataOffset,
                SEEK_SET) != info->dataOffset) {
                    datRaw_logError("DatRaw: Error reading data file %s (skipping failed)!\n", info->dataFileName);
                    gzclose(info->fd_dataFile);
                    info->fd_dataFile = NULL;
                    return 0;
            }
        }
    } else {
        int maxFilename, minWidth, offset, stride;
        char *filenameTemplate, *filename;

        filenameTemplate = parseMultifileDescription(info->dataFileName, &minWidth, &offset, &stride);

        if (!filenameTemplate) {
            return 0;
        }

        maxFilename = (int)strlen(filenameTemplate) +
            DATRAW_MAX((int)log10(info->timeSteps), minWidth) + 2;

        if (!(filename = (char*)malloc(maxFilename * sizeof(char)))) {
            datRaw_logError("DatRaw: Failed to allocate memory for filename\n");
            return 0;
        }

        sprintf(filename, filenameTemplate, offset + stride*info->currentStep);

        free(filenameTemplate);

        if (!(info->fd_dataFile = gzopen(filename, "rb"))) {
            datRaw_logError("DatRaw: Error opening data file \"%s\"!\n",
                filename);
            free(filename);
            return 0;
        }
        free(filename);
        if (info->dataOffset > 0 && gzseek(info->fd_dataFile, info->dataOffset,
            SEEK_SET) != info->dataOffset) {
                datRaw_logError("DatRaw: Error reading data file %s (skipping failed)!\n", info->dataFileName);
                gzclose(info->fd_dataFile);
                info->fd_dataFile = NULL;
                return 0;
        }
    }

    storageSize = datRaw_getBufferSize(info, info->dataFormat);
    memSize = datRaw_getBufferSize(info, format);

    if (memSize <= 0 || storageSize <= 0) {
        return 0;
    }

    if (allocateMem && !(*buffer = malloc(memSize))) {
        datRaw_logError("DatRaw Error: Could not allocate buffer memory (%ld byte)\n", memSize);
        return 0;
    }

    if (format == info->dataFormat || format == DR_FORMAT_RAW) {
        buf = *buffer;
    } else {
        if (!(buf = malloc(storageSize))) {
            datRaw_logError("DatRaw Error: Could not allocate tmp. buffer memory (%ld byte)\n", storageSize);
            if (info->multiDataFiles) {
                gzclose(info->fd_dataFile);
            }
            info->fd_dataFile = NULL;
            if (allocateMem) {
                free(*buffer);
                *buffer = NULL;
            }
            return 0;
        }
    }

    if (gzread(info->fd_dataFile, buf, storageSize) != storageSize) {
        if (gzeof(info->fd_dataFile)) {
            datRaw_logError("DatRaw: Error reading data file %s (EOF reached)!\n", info->dataFileName);
        } 
        else {
            int err;
            gzerror(info->fd_dataFile, &err);
            if (err) {
                datRaw_logError("DatRaw: Error reading data file %s\n", info->dataFileName);
            }
        }
        gzclearerr(info->fd_dataFile);
        if (format != info->dataFormat && format != DR_FORMAT_RAW) {
            free(buf);
        }
        if (info->multiDataFiles) {
            gzclose(info->fd_dataFile);
            info->fd_dataFile = NULL;
        }
        if (allocateMem) {
            free(*buffer);
            *buffer = NULL;
        }
        return 0;
    }

    if (info->byteOrder != datRaw_getByteOrder()) {
        swapByteOrder(buf, 
            datRaw_getElementCount(info) * info->numComponents,
            info->dataFormat);
    }


    if (format != info->dataFormat && format != DR_FORMAT_RAW) {
        datRaw_convertBlock(buf, info->dataFormat, (DR_UCHAR*)*buffer, format, 
            datRaw_getElementCount(info) * info->numComponents);
        free(buf);
    }

    if (info->multiDataFiles) {
        gzclose(info->fd_dataFile);
        info->fd_dataFile = NULL;
    }

    return 1;
}

/*
loads the data of the previous time step into *buffer. If *buffer is
NULL, memory is allocated that fits the data size.
Returns -1, if there is no previous timestep to load and 0, if an error
occured.
*/
int datRaw_getPrevious(DatRawFileInfo *info, void **buffer, int format)
{
    size_t storageSize, memSize;
    int allocateMem;
    DR_UCHAR *buf;

    if (info->currentStep <= 0) {
        return -1;
    }

    if (!buffer) {
        return 0;
    }

    allocateMem = !(*buffer);

    info->currentStep--;

    storageSize = datRaw_getBufferSize(info, info->dataFormat);
    if (storageSize <= 0) {
        return 0;
    }

    if (!info->multiDataFiles) {
        if (info->fd_dataFile == NULL) {
            datRaw_logError("DatRaw: Cant't get previous timestep from closed file\n");
            return 0;
        }
        if (gzseek(info->fd_dataFile, info->currentStep*storageSize, SEEK_SET) < 0) {
            if (gzeof(info->fd_dataFile)) {
                datRaw_logError("DatRaw: Error reading data file %s (EOF reached)!\n", 
                    info->dataFileName);
            } else {
                int err;
                gzerror(info->fd_dataFile, &err);
                if (err) {
                    datRaw_logError("DatRaw: Error reading data file %s\n", info->dataFileName);
                }
            }
            gzclearerr(info->fd_dataFile);
            return 0;
        }
    } else {
        int maxFilename, minWidth, offset, stride;
        char *filenameTemplate, *filename;

        filenameTemplate = parseMultifileDescription(info->dataFileName, &minWidth, &offset, &stride);

        if (!filenameTemplate) {
            return 0;
        }

        maxFilename = (int)strlen(filenameTemplate) +
            DATRAW_MAX((int)log10(info->timeSteps), minWidth) + 2;

        if (!(filename = (char*)malloc(maxFilename * sizeof(char)))) {
            datRaw_logError("DatRaw: Failed to allocate memory for filename\n");
            return 0;
        }

        sprintf(filename, filenameTemplate, offset + stride*info->currentStep);

        free(filenameTemplate); 

        if (!(info->fd_dataFile = gzopen(filename, "rb"))) {
            datRaw_logError("DatRaw: Error opening data file \"%s\"!\n",
                filename);
            free(filename);
            return 0;
        }
        free(filename);
        if (info->dataOffset > 0 && gzseek(info->fd_dataFile, info->dataOffset,
            SEEK_SET) != info->dataOffset) {
                datRaw_logError("DatRaw: Error reading data file %s (skipping failed)!\n", info->dataFileName);
                gzclose(info->fd_dataFile);
                info->fd_dataFile = NULL;
                return 0;
        }
    }

    memSize = datRaw_getBufferSize(info, format);

    if (memSize <= 0) {
        return 0;
    }

    if (allocateMem && !(*buffer = malloc(memSize))) {
        datRaw_logError("DatRaw Error: Could not allocate buffer memory (%ld byte)\n", memSize);
        return 0;
    }


    if (format == info->dataFormat || format == DR_FORMAT_RAW) {
        buf = *buffer;
    } else {
        if (!(buf = malloc(storageSize))) {
            datRaw_logError("DatRaw Error: Could not allocate tmp. buffer memory (%ld byte)\n", storageSize);
            if (info->multiDataFiles) {
                gzclose(info->fd_dataFile);
            }
            info->fd_dataFile = NULL;
            if (allocateMem) {
                free(*buffer);
                *buffer = NULL;
            }
            return 0;
        }
    }

    if (gzread(info->fd_dataFile, buf, storageSize) != storageSize) {
        if (gzeof(info->fd_dataFile)) {
            datRaw_logError("DatRaw: Error reading data file %s (EOF reached)!\n", info->dataFileName);
        } else {
            int err;
            gzerror(info->fd_dataFile, &err);
            if (err) {
                datRaw_logError("DatRaw: Error reading data file %s\n", info->dataFileName);
            }
        }
        gzclearerr(info->fd_dataFile);
        if (format != info->dataFormat && format != DR_FORMAT_RAW) {
            free(buf);
        }
        if (info->multiDataFiles) {
            gzclose(info->fd_dataFile);
            info->fd_dataFile = NULL;
        }
        if (allocateMem) {
            free(*buffer);
            *buffer = NULL;
        }
        return 0;
    }

    if (info->byteOrder != datRaw_getByteOrder()) {
        swapByteOrder(buf, 
            datRaw_getElementCount(info) * info->numComponents,
            info->dataFormat);
    }

    if (format != info->dataFormat && format != DR_FORMAT_RAW) {
        datRaw_convertBlock(buf, info->dataFormat, (DR_UCHAR*)*buffer, format, 
            datRaw_getElementCount(info) * info->numComponents);
        free(buf);
    }

    if (info->multiDataFiles) {
        gzclose(info->fd_dataFile);
        info->fd_dataFile = NULL;
    }
    return 1;
}


int datRaw_loadStep(DatRawFileInfo *info, int n, void **buffer, int format)
{
    size_t storageSize, memSize;
    int allocateMem;
    DR_UCHAR *buf;

    if (n < 0 || n >= info->timeSteps) {
        datRaw_logError("Invalid time step specified\n");
        return -1;
    }

    if (!buffer) {
        return 0;
    }

    allocateMem = !(*buffer);

    storageSize = datRaw_getBufferSize(info, info->dataFormat);

    if (storageSize <= 0) {
        return 0;
    }

    if (!info->multiDataFiles) {
        if (info->fd_dataFile == NULL && info->currentStep == -1) {
            if (!(info->fd_dataFile = gzopen(info->dataFileName, "rb"))) {
                datRaw_logError("DatRaw: Error opening data file \"%s\"!\n",
                    info->dataFileName);
                return 0;
            }
        }
        if (gzseek(info->fd_dataFile, info->dataOffset+n*storageSize, SEEK_SET) < 0) {
            if (gzeof(info->fd_dataFile)) {
                datRaw_logError("DatRaw: Error reading data file %s (EOF reached)!\n", 
                    info->dataFileName);
            } else {
                int err;
                gzerror(info->fd_dataFile, &err);
                if (err) {
                    datRaw_logError("DatRaw: Error reading data file %s\n", info->dataFileName);
                } else {
                    datRaw_logError("DatRaw: Error seeking in data (too few bytes in file?)\n");
                }
            }
            gzclearerr(info->fd_dataFile);
            return 0;
        }
    } else {
        int maxFilename, minWidth, offset, stride;
        char *filenameTemplate, *filename;

        filenameTemplate = parseMultifileDescription(info->dataFileName, &minWidth, &offset, &stride);

        if (!filenameTemplate) {
            return 0;
        }

        maxFilename = (int)strlen(filenameTemplate) +
            DATRAW_MAX((int)log10(info->timeSteps), minWidth) + 2;

        if (!(filename = (char*)malloc(maxFilename * sizeof(char)))) {
            datRaw_logError("DatRaw: Failed to allocate memory for filename\n");
            return 0;
        }

        sprintf(filename, filenameTemplate, offset + stride*n);

        free(filenameTemplate);

        if (!(info->fd_dataFile = gzopen(filename, "rb"))) {
            datRaw_logError("DatRaw: Error opening data file \"%s\"!\n",
                filename);
            free(filename);
            return 0;
        }
        free(filename);
        if (info->dataOffset > 0 && gzseek(info->fd_dataFile, info->dataOffset,
            SEEK_SET) != info->dataOffset) {
                datRaw_logError("DatRaw: Error reading data file %s (skipping failed)!\n", info->dataFileName);
                gzclose(info->fd_dataFile);
                info->fd_dataFile = NULL;
                return 0;
        }
    }

    memSize = datRaw_getBufferSize(info, format);

    if (memSize <= 0) {
        return 0;
    }

    if (allocateMem && !(*buffer = malloc(memSize))) {
        datRaw_logError("DatRaw Error: Could not allocate buffer memory (%ld byte)\n", memSize);
        return 0;
    }

    if (format == info->dataFormat || format == DR_FORMAT_RAW) {
        buf = *buffer;
    } else {
        if (!(buf = malloc(storageSize))) {
            datRaw_logError("DatRaw Error: Could not allocate tmp. buffer memory (%ld byte)\n", storageSize);
            if (info->multiDataFiles) {
                gzclose(info->fd_dataFile);
            }
            info->fd_dataFile = NULL;
            if (allocateMem) {
                free(*buffer);
                *buffer = NULL;
            }
            return 0;
        }
    }

    if (gzread(info->fd_dataFile, buf, storageSize) != storageSize) {
        if (gzeof(info->fd_dataFile)) {
            datRaw_logError("DatRaw: Error reading data file %s (EOF reached)!\n", info->dataFileName);
        } else {
            int err;
            gzerror(info->fd_dataFile, &err);
            if (err) {
                datRaw_logError("DatRaw: Error reading data file %s\n", info->dataFileName);
            } else {
                datRaw_logError("DateRaw: Error reading data (too few bytes in file?)\n");
            }
        }
        gzclearerr(info->fd_dataFile);
        if (format != info->dataFormat && format != DR_FORMAT_RAW) {
            free(buf);
        }
        if (info->multiDataFiles) {
            gzclose(info->fd_dataFile);
            info->fd_dataFile = NULL;
        }
        if (allocateMem) {
            free(*buffer);
            *buffer = NULL;
        }
        return 0;
    }

    if (info->byteOrder != datRaw_getByteOrder()) {
        swapByteOrder(buf, 
            datRaw_getElementCount(info) * info->numComponents,
            info->dataFormat);
    }

    if (format != info->dataFormat && format != DR_FORMAT_RAW) {
        datRaw_convertBlock(buf, info->dataFormat, (DR_UCHAR*)*buffer, format, 
            datRaw_getElementCount(info) * info->numComponents);
        free(buf);
    }

    if (info->multiDataFiles) {
        gzclose(info->fd_dataFile);
        info->fd_dataFile = NULL;
    }
    info->currentStep = n;
    return 1;
}


void datRaw_close(DatRawFileInfo *info)
{
    if (info->fd_dataFile) {
        gzclose(info->fd_dataFile);
        info->fd_dataFile = NULL;
        info->currentStep = -1;
    }
}

static int checkFileInfo(const DatRawFileInfo *info)
{
    int i, result = 1;
    int size;
    char *tag = NULL;

    DATRAW_GET_TAG_FROM_VALUE(tag, datRawGridTypes, info->gridType)
        result = result && tag != NULL;
    DATRAW_GET_TAG_FROM_VALUE(tag, datRawDataFormats, info->dataFormat)
        result = result && tag != NULL;

    result =  result &&
        info->descFileName != NULL &&
        info->dataFileName != NULL &&
        info->dimensions > 0 &&
        info->timeSteps > 0;

    result = result && (info->byteOrder == DR_BIG_ENDIAN ||
        info->byteOrder == DR_LITTLE_ENDIAN);

    result = result && (info->dataOffset >= 0);

    switch(info->gridType) {
    case DR_GRID_CARTESIAN:
        for (i = 0; i < info->dimensions; i++) {
            if (info->sliceDist[i] <= 0.0 || info->resolution[i] <= 0) {
                result = 0;
                break;
            }
        }
        break;
    case DR_GRID_RECTILINEAR:
        size = 0;
        for (i = 0; i < info->dimensions; i++) {
            if (info->resolution[i] <= 0) {
                result = 0;
                break;
            }
            size += info->resolution[i];
        }
        for (i = 0; i < size; ++i) {
            if (info->sliceDist[i] <= 0) {
                result = 0;
                break;
            }
        }
        break;
    case DR_GRID_TETRAHEDRAL:
        result = result && 
            info->numVertices > info->dimensions && 
            info->numTetrahedra >= 0 ;
        break;
    default:
        result = 0;
        break;
    }

    return result;
}


int datRaw_writeHeaderFile(
                           const DatRawFileInfo *info,
                           DatRawOptionalField **optionalFields)
{
    char *grid, *format;
    int i, j, k;
    FILE *hf;

    if (!checkFileInfo(info)) {
        datRaw_logError("DatRaw: invalid file info, please check!\n");
        return 0;
    }

    hf = fopen(info->descFileName, "w");

    if (!hf) {
        datRaw_logError("DatRaw: Could not create header file\n");
        return 0;
    }

    grid = format = NULL;

    DATRAW_GET_TAG_FROM_VALUE(grid, datRawGridTypes, info->gridType)
    DATRAW_GET_TAG_FROM_VALUE(format, datRawDataFormats, info->dataFormat)

    fprintf(hf, "OBJECTFILENAME: %s\n", info->dataFileName);
    fprintf(hf, "FORMAT: %s\n", format);
    fprintf(hf, "GRIDTYPE: %s\n", grid);
    fprintf(hf, "COMPONENTS: %d\n", info->numComponents);
    fprintf(hf, "DIMENSIONS: %d\n", info->dimensions);
    fprintf(hf, "TIMESTEPS: %d\n", info->timeSteps);
    fprintf(hf, "BYTEORDER: %s\n", 
        datRaw_getByteOrder() == DR_BIG_ENDIAN ? "BIG_ENDIAN" : "LITTLE_ENDIAN");
    switch (info->gridType) {
    case DR_GRID_CARTESIAN:
        fprintf(hf, "RESOLUTION: %d", info->resolution[0]);
        for (i = 1; i < info->dimensions; i++) {
            fprintf(hf, " %d", info->resolution[i]);
        }
        fprintf(hf, "\n");
        fprintf(hf, "SLICETHICKNESS: %f", info->sliceDist[0]);
        for (i = 1; i < info->dimensions; i++) {
            fprintf(hf, " %f", info->sliceDist[i]);
        }
        fprintf(hf, "\n");
        break;
    case DR_GRID_RECTILINEAR:
        fprintf(hf, "RESOLUTION: %d", info->resolution[0]);
        for (i = 1; i < info->dimensions; i++) {
            fprintf(hf, " %d", info->resolution[i]);
        }
        fprintf(hf, "\n");
        for (i = 0, j = 0; i < info->dimensions; ++i) {
            fprintf(hf, "SLICETHICKNESS[%d]: ", i);
            for (k = 0; k < info->resolution[i]; ++j, ++k) {
                fprintf(hf, " %f", info->sliceDist[j]);
            }

            fprintf(hf, "\n");
        }
        break;
    case DR_GRID_TETRAHEDRAL:
        fprintf(hf, "VERTICES: %d\n", info->numVertices);
        fprintf(hf, "TETRAHEDRA: %d\n", info->numTetrahedra);
        break;
    default:
        break;
    }

    if (optionalFields) {
        int n = 0;
        for (; optionalFields[n]; ++n) {
            const DatRawOptionalField* pOptionalField = optionalFields[n];
            int elementSize = datRaw_getFormatSize(pOptionalField->format);
            const char* formatString = datRaw_getFormatString(pOptionalField->format);
            int numElements = 0;
            /* sanity checks */
            if (!pOptionalField->name ||
                elementSize <= 0 ||
                pOptionalField->numComponents <= 0 ||
                !pOptionalField->data)
            {
                datRaw_logError("DatRaw Error: Optional data field "
                    "description %d invalid\n", n);
                fclose(hf);
                return 0;
            }

            fprintf(hf, "%s:", pOptionalField->name);

            numElements = pOptionalField->numComponents;
            if (pOptionalField->timeDependent) {
                numElements *= info->timeSteps;
            }

            for (i = 0; i < numElements; ++i) {
                fprintf(hf, " ");
                switch (pOptionalField->format) {
                    case DR_FORMAT_CHAR:
                        fprintf(hf, formatString,
                            ((DR_CHAR*)pOptionalField->data)[i]);
                        break;
                    case DR_FORMAT_UCHAR:
                        fprintf(hf, formatString,
                            ((DR_UCHAR*)pOptionalField->data)[i]);
                        break;
                    case DR_FORMAT_SHORT:
                        fprintf(hf, formatString,
                            ((DR_SHORT*)pOptionalField->data)[i]);
                        break;
                    case DR_FORMAT_USHORT:
                        fprintf(hf, formatString,
                            ((DR_USHORT*)pOptionalField->data)[i]);
                        break;
                    case DR_FORMAT_INT:
                        fprintf(hf, formatString,
                            ((DR_INT*)pOptionalField->data)[i]);
                        break;
                    case DR_FORMAT_UINT:
                        fprintf(hf, formatString,
                            ((DR_UINT*)pOptionalField->data)[i]);
                        break;
                    case DR_FORMAT_LONG:
                        fprintf(hf, formatString,
                            ((DR_LONG*)pOptionalField->data)[i]);
                        break;
                    case DR_FORMAT_ULONG:
                        fprintf(hf, formatString,
                            ((DR_ULONG*)pOptionalField->data)[i]);
                        break;
                    case DR_FORMAT_FLOAT:
                        fprintf(hf, formatString,
                            ((DR_FLOAT*)pOptionalField->data)[i]);
                        break;
                    case DR_FORMAT_DOUBLE:
                        fprintf(hf, formatString,
                            ((DR_DOUBLE*)pOptionalField->data)[i]);
                        break;
                    default:
                        datRaw_logError("DatRaw Error: Optional data field "
                            "description %d invalid\n", n);
                        fclose(hf);
                        return 0;
                }
            }
            fprintf(hf, "\n");
        }
    }

    fclose(hf);

    return 1;
}

static FILE *wfopen(const char *path, const char *mode, int compress)
{
    if (compress) {
        return gzopen(path, mode);
    } else {
        return fopen(path, mode);
    }
}

static size_t wfwrite(const void *ptr, size_t size, size_t  nmemb,
                      FILE *stream, int compress)
{
    if (compress) {
        return gzwrite(stream, ptr, size*nmemb) / size;
    } else {
        return fwrite(ptr, size, nmemb, stream);
    }
}

static int wfseek(FILE *stream, long offset, int whence, int compress)
{
    if (compress) {
        return gzseek(stream, offset, whence);
    } else {
        return fseek(stream, offset, whence);
    }
}

static int wfclose(FILE *stream, int compress)
{
    if (compress) {
        return gzclose(stream);
    } else {
        return fclose(stream);
    }
}

static int datRaw_writeRawFileSingle(const DatRawFileInfo *info, void *buffer,
                                     int bufferFormat, int compress)
{
    FILE *rf;
    size_t storageSize;

    rf = wfopen(info->dataFileName, "wb", compress);
    if (!rf) {
        datRaw_logError("DatRaw: Could not open raw data file \"%s\" for writting\n",
            info->dataFileName);
        return 0;
    }

    storageSize = datRaw_getBufferSize(info, info->dataFormat) * info->timeSteps;

    if (bufferFormat == info->dataFormat || bufferFormat == DR_FORMAT_RAW) {
        if (storageSize <= 0 || wfwrite(buffer, storageSize, 1, rf, compress) != 1) {
            datRaw_logError("DatRaw: Error writing raw data file \"%s\"", info->dataFileName);
            wfclose(rf, compress);
            return 0;
        }
    }
    else {
        void *buf;

        if (!(buf = malloc(storageSize))) {
            datRaw_logError("DatRaw Error: Could not allocate tmp. buffer "
                "memory (%lu byte)\n", storageSize);
            wfclose(rf, compress);
            return 0;
        }
        datRaw_convertBlock(buffer, bufferFormat, buf, info->dataFormat, 
            datRaw_getElementCount(info) *
            info->numComponents *
            info->timeSteps);

        if (storageSize <= 0 || wfwrite(buf, storageSize, 1, rf, compress) != 1) {
            datRaw_logError("DatRaw: Error writing raw data file \"%s\"", info->dataFileName);
            wfclose(rf, compress);
            return 0;
        }
        free(buf);
    }
    wfclose(rf, compress);
    return 1;
}

static int datRaw_writeRawFileMulti(const DatRawFileInfo *info, void *buffer,
                                    int bufferFormat, int compress)
{
    FILE *rf;
    char *filename, *filenameTemplate;
    int i, maxFilename, minWidth, offset, stride;
    size_t blocksize;
    size_t storageSize;
    void *buf = NULL ;

    filenameTemplate = parseMultifileDescription(info->dataFileName, &minWidth, &offset, &stride);

    if (!filenameTemplate) {
        return 0;
    }

    maxFilename = (int)strlen(filenameTemplate) +
        DATRAW_MAX((int)log10(info->timeSteps), minWidth) + 2;

    if (!(filename = (char*)malloc(maxFilename * sizeof(char)))) {
        datRaw_logError("DatRaw: Failed to allocate memory for filename\n");
        return 0;
    }

    blocksize = datRaw_getBufferSize(info, bufferFormat);

    if (blocksize <= 0) {
        datRaw_logError("DatRaw: Buffersize invalid!\n");
        return 0;
    }

    storageSize = datRaw_getBufferSize(info, info->dataFormat);

    if (bufferFormat != info->dataFormat && bufferFormat != DR_FORMAT_RAW ) {
        if (!(buf = malloc(storageSize))) {
            datRaw_logError("DatRaw Error: Could not allocate tmp. "
                "buffer memory (%lu byte)\n", storageSize);
            return 0;
        }
    }

    for (i = 0; i < info->timeSteps; i++) {

        sprintf(filename, filenameTemplate, offset + stride*info->currentStep);

        rf = wfopen(filename, "wb", compress);
        if (!rf) {
            datRaw_logError("DatRaw: Could not open raw data file for writing\"%s\"\n",
                filename);
            free(buf);
            free(filename);
            free(filenameTemplate);
            return 0;
        }


        if (bufferFormat == info->dataFormat || bufferFormat == DR_FORMAT_RAW) {
            if (wfwrite((DR_UCHAR*)buffer + (i * blocksize), blocksize, 1, rf,
                compress) != 1) {
                    datRaw_logError("DatRaw: Error writing raw data file \"%s\"",
                        filename);
                    wfclose(rf, compress);
                    free(buf);
                    free(filename);
                    free(filenameTemplate);
                    return 0;
            }
        }
        else {
            datRaw_convertBlock((DR_UCHAR*)buffer + i*blocksize, bufferFormat, buf, info->dataFormat, 
                datRaw_getElementCount(info) * info->numComponents);


            if (wfwrite(buf, storageSize, 1, rf, compress) != 1) {
                datRaw_logError("DatRaw: Error writing raw data file \"%s\"",
                    filename);
                wfclose(rf, compress);
                free(filename);
                free(filenameTemplate);
                free(buf);
                return 0;
            }
        }
        wfclose(rf, compress);
    }

    free(filename);
    free(filenameTemplate);
    free(buf);

    return 1;
}

int datRaw_write(
                 const DatRawFileInfo *info,
                 DatRawOptionalField **optionalFields,
                 void *buffer,
                 int bufferFormat,
                 int compress)
{
    if(!datRaw_writeHeaderFile(info, optionalFields)) {
        return 0;
    }

    if (info->multiDataFiles) {
        return datRaw_writeRawFileMulti(info, buffer, bufferFormat, compress);
    }
    else {
        return datRaw_writeRawFileSingle(info, buffer, bufferFormat, compress);
    }
}

/*
write the buffer content to a raw-file, if it is part of a multi-file a new
file is created, otherwise the data is appended to the raw-file (in this
case, the data has to be written sequentially starting with timestep 0)
*/
int datRaw_writeTimestep(const DatRawFileInfo *info, void *buffer,
                         int bufferFormat, int compress, int timeStep)
{
    FILE *rf;
    size_t storageSize;

    if (info->multiDataFiles) {
        int maxFilename, minWidth, offset, stride;
        char *filenameTemplate, *filename;

        filenameTemplate = parseMultifileDescription(info->dataFileName, &minWidth, &offset, &stride);

        if (!filenameTemplate) {
            return 0;
        }

        maxFilename = (int)strlen(filenameTemplate) +
            DATRAW_MAX((int)log10(info->timeSteps), minWidth) + 2;

        if (!(filename = (char*)malloc(maxFilename * sizeof(char)))) {
            datRaw_logError("DatRaw: Failed to allocate memory for filename\n");
            return 0;
        }

        sprintf(filename, filenameTemplate, offset + stride*timeStep);

        if (!(rf = wfopen(filename, "wb", compress))) {
            datRaw_logError("DatRaw: Could not open raw data file \"%s\"\n",
                filename);
        }
        free(filenameTemplate);
        free(filename);
    } else {
        if (timeStep == 0) {
            rf = wfopen(info->dataFileName, "wb", compress);
        } else {
            rf = wfopen(info->dataFileName, "ab", compress);
        }
        if (!rf) {
            datRaw_logError("DatRaw: Could not open raw data file \"%s\"\n",
                info->dataFileName);
            return 0;
        }
    }

    storageSize = datRaw_getBufferSize(info, info->dataFormat);

    if (!info->multiDataFiles) {
        if (wfseek(rf, timeStep*storageSize, SEEK_SET, compress)) {
            perror("Failed to reposition output stream\n");
            return 0;
        }
    }

    if (bufferFormat == info->dataFormat || bufferFormat == DR_FORMAT_RAW) {
        if (wfwrite((DR_UCHAR*)buffer, storageSize, 1, rf, compress) != 1) {
            datRaw_logError("DatRaw: Error writing raw data file \"%s\"", info->dataFileName);
            wfclose(rf, compress);
            return 0;
        }
    }
    else {
        void *buf = NULL ;
        if (!(buf = malloc(storageSize))) {
            datRaw_logError("DatRaw Error: Could not allocate tmp. "
                "buffer memory (%lu byte)\n", storageSize);
            wfclose(rf, compress);
            return 0;
        }
        datRaw_convertBlock(buffer, bufferFormat, buf, info->dataFormat, 
            datRaw_getElementCount(info) * info->numComponents);

        if (wfwrite(buf, storageSize, 1, rf, compress) != 1) {
            datRaw_logError("DatRaw: Error writing raw data file \"%s\"", info->dataFileName);
            wfclose(rf, compress);
            free(buf);
            return 0;
        }

        free(buf);
    }
    wfclose(rf, compress);
    return 1;
}


int datRaw_createInfo( DatRawFileInfo *info,
                      const char *descFileName,
                      const char *dataFileName,
                      int   dimensions,
                      int   timeSteps,
                      int   gridType,
                      int   numComponents,
                      int   dataFormat,
                      ...)
{
    const int *resolution;
    const float *sliceDist;

    va_list argp;
    size_t size;
    int i;

    /* initialize save values */
    info->descFileName = NULL;
    info->dataFileName = NULL;
    info->dimensions = dimensions;
    info->timeSteps = timeSteps;
    info->gridType = gridType;
    info->numComponents = numComponents;
    info->dataFormat = dataFormat;
    info->multiDataFiles = 0;
    info->sliceDist = NULL;
    info->resolution = NULL;
    info->numVertices = 0;
    info->numTetrahedra = 0;
    info->currentStep = -1;
    info->fd_dataFile = NULL;
    info->dataOffset = 0;
    info->byteOrder = datRaw_getByteOrder();

    /* check for obviously wrong arguments */
    /* todo: add some more info */
    if (!descFileName || 
        !dataFileName || 
        dimensions <= 0 || 
        timeSteps <= 0) {
            datRaw_logError("DatRaw: Some values were plain wrong ;-)\n");
            return 0;
    }

    if (!(info->descFileName = (char*)malloc(strlen(descFileName) + 1)) ||
        !(info->dataFileName = (char*)malloc(strlen(dataFileName) + 1))) {
            datRaw_logError("DatRaw: Failed to allocate memory for description record!\n");
            return 0;
    }
    strcpy(info->descFileName, descFileName);
    strcpy(info->dataFileName, dataFileName);

    if (isMultifileDescription(dataFileName)) {
        info->multiDataFiles = 1;
    }

    switch(info->gridType) {
    case DR_GRID_CARTESIAN:
        va_start(argp, dataFormat);
        resolution = va_arg(argp, const int*);
        sliceDist = va_arg(argp, const float*);
        va_end(argp);
        if (!(info->sliceDist = (float*)malloc(info->dimensions * sizeof(float))) || 
            !(info->resolution = (int*)malloc(info->dimensions * sizeof(int)))) {
                datRaw_logError("DatRaw: Failed to allocate memory for description record!\n");
                return 0;
        }
        memcpy(info->sliceDist, sliceDist, info->dimensions * sizeof(float));
        memcpy(info->resolution, resolution, info->dimensions * sizeof(int));
        break;
    case DR_GRID_RECTILINEAR:
        va_start(argp, dataFormat);
        resolution = va_arg(argp, const int*);
        sliceDist = va_arg(argp, const float*);
        va_end(argp);
        if (!(info->resolution = (int*)malloc(info->dimensions * sizeof(int)))) {
            datRaw_logError("DatRaw: Failed to allocate memory for description record!\n");
            return 0;
        }
        memcpy(info->resolution, resolution, info->dimensions * sizeof(int));

        size = 0;
        for (i = 0; i < info->dimensions; ++i) {
            size += info->resolution[i];
        }
        size *= sizeof(float);

        if (!(info->sliceDist = (float *) malloc(size))) {
            datRaw_logError("DatRaw: Failed to allocate memory for description record!\n");
            return 0;
        }
        memcpy(info->sliceDist, sliceDist, size);
        break;
    case DR_GRID_TETRAHEDRAL:
        va_start(argp, dataFormat);
        info->numVertices = va_arg(argp, int);
        info->numTetrahedra = va_arg(argp, int);
        va_end(argp);
        break;
    default:
        break;
    }

    return checkFileInfo(info);
}

/*
check whether info matches values; 0 == don't care
*/
int datRaw_checkInfo(const DatRawFileInfo *info, 
                     const char *descFileName,
                     const char *dataFileName,
                     int   dimensions,
                     int   timeSteps,
                     int   gridType,
                     int   numComponents,
                     int   dataFormat,
                     ...)
{
    const int *resolution;
    const float *sliceDist;
    int i, numTetrahedra, numVertices, size;
    va_list argp;

    /* initialize save values */
    if (descFileName && strcmp(info->descFileName, descFileName)) {
        return 0;
    } 
    if (dataFileName && strcmp(info->dataFileName, dataFileName)) {
        return 0;
    }
    if (dimensions && info->dimensions != dimensions) {
        return 0;
    }
    if (timeSteps && info->timeSteps != timeSteps) {
        return 0;
    }
    if (gridType && info->gridType != gridType) {
        return 0;
    }
    if (numComponents && info->numComponents != numComponents) {
        return 0;
    }
    if (dataFormat && info->dataFormat != dataFormat) {
        return 0;
    }

    switch(info->gridType) {
    case DR_GRID_CARTESIAN:
        va_start(argp, dataFormat);
        resolution = va_arg(argp, const int*);
        sliceDist = va_arg(argp, const float*);
        va_end(argp);
        for (i = 0; i < info->dimensions; i++) {
            if (resolution && info->resolution[i] != resolution[i]) {
                return 0;
            }
            if (sliceDist && info->sliceDist[i] != sliceDist[i]) {
                return 0;
            }
        }
        break;
    case DR_GRID_RECTILINEAR:
        va_start(argp, dataFormat);
        resolution = va_arg(argp, const int*);
        sliceDist = va_arg(argp, const float*);
        va_end(argp);
        for (i = 0; i < info->dimensions; i++) {
            if (resolution && info->resolution[i] != resolution[i]) {
                return 0;
            }
        }

        size = 0;
        for (i = 0; i < info->dimensions; ++i) {
            size += info->resolution[i];
        }

        for (i = 0; i < size; ++i) {
            if (sliceDist && info->sliceDist[i] != sliceDist[i]) {
                return 0;
            }
        }
        break;
    case DR_GRID_TETRAHEDRAL:
        va_start(argp, dataFormat);
        numVertices = va_arg(argp, int);
        numTetrahedra = va_arg(argp, int);
        va_end(argp);
        if (numVertices && info->numVertices != numVertices) {
            return 0;
        }
        if (numTetrahedra && info->numTetrahedra != numTetrahedra) {
            return 0;
        }
        break;
    default:
        break;
    }

    return 1;
}

int datRaw_copyInfo(DatRawFileInfo *newinfo, DatRawFileInfo *oldinfo)
{
    int i;

    *newinfo = *oldinfo;
    if (!(newinfo->descFileName = dupstr(oldinfo->descFileName)) ||
        !(newinfo->dataFileName = dupstr(oldinfo->dataFileName)) ||
        !(newinfo->sliceDist = (float*)malloc(oldinfo->dimensions*sizeof(float))) ||
        !(newinfo->resolution = (int*)malloc(oldinfo->dimensions*sizeof(int))))
    {
        datRaw_logError("DatRaw: Failed to copy description record\n");
        return 0;
    }
    for (i = 0; i < oldinfo->dimensions; i++) {
        newinfo->sliceDist[i] = oldinfo->sliceDist[i];
        newinfo->resolution[i] = oldinfo->resolution[i];
    }
    return 1;
}

void datRaw_freeInfo(DatRawFileInfo *info)
{
    int i;

    switch(info->gridType) {
        case DR_GRID_CARTESIAN:
        case DR_GRID_RECTILINEAR:
            free(info->sliceDist);
            free(info->resolution);
            break;
        case DR_GRID_TETRAHEDRAL:
        default:
            break;
    }
    free(info->dataFileName);
    free(info->descFileName);
}

