/*
 * MultiIndexListDataCall.h
 *
 * Copyright (C) 2016 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_DATATOOLS_MULTIINDEXLISTDATACALL_H_INCLUDED
#define MEGAMOL_DATATOOLS_MULTIINDEXLISTDATACALL_H_INCLUDED
#pragma once

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include <cstdint>

namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Call transports multiple index lists, e.g. to subselect groups/clusters within data transported by a parallel MultiParticleListDataCall.
     */
    class MultiIndexListDataCall : public core::AbstractGetDataCall {
    public:
        /** Type for index entries */
        typedef uint32_t index_t;
        /** Type for index list lengths */
        typedef uint32_t length_t;

        typedef struct _index_list_t {
            /** Pointer to the first index entry */
            index_t const* data;
            /** Number of index entry 'data' points to */
            length_t length;

            index_t const* begin() const {
                return data;
            }
            index_t const* end() const {
                return data + length;
            }

        } index_list_t;

        /** Possible call functions/intends */
        enum CallFunctionName : int {
            GET_DATA = 0,
            GET_EXTENT = 1, /* temporal */
            GET_HASH = 2
        };

        /** Factory metadata */
        static const char *ClassName(void) { return "MultiIndexListDataCall"; }
        static const char *Description(void) { return "Call transports multiple index lists, e.g. to subselect groups/clusters within data transported by a parallel MultiParticleListDataCall."; }
        static unsigned int FunctionCount(void) { return 3; }
        static const char * FunctionName(unsigned int idx) {
            switch (idx) {
            case GET_DATA: return "GetData";
            case GET_EXTENT: return "GetExtent";
            case GET_HASH: return "GetHash";
            }
            return nullptr;
        }

        /** ctor */
        MultiIndexListDataCall();
        /** dtor */
        virtual ~MultiIndexListDataCall();

        /** Array of index lists */
        inline index_list_t const* Lists() const { return lsts; }
        /** Number of index lists 'Lists' returns */
        inline size_t Count() const { return lsts_len; }
        /** Current frame ID (zero-based) */
        inline unsigned int FrameID() const { return frameID; }
        /** Number of frames available (set by GetExtent) */
        inline unsigned int FrameCount() const { return frameCnt; }

        /** Sets lists data. No deep copy. Caller must keep memory alive */
        inline void Set(index_list_t const* lsts, size_t cnt) {
            this->lsts = lsts;
            this->lsts_len = cnt;
        }
        /** Sets current frame ID (zero-based) */
        inline void SetFrameID(unsigned int fid) {
            frameID = fid;
        }
        /** Sets number of frames available */
        inline void SetFrameCount(unsigned int cnt) {
            frameCnt = cnt;
        }

    private:

        /* data */
        index_list_t const *lsts;
        size_t lsts_len;
        unsigned int frameCnt;
        unsigned int frameID;

    };

    typedef core::factories::CallAutoDescription<MultiIndexListDataCall> MultiIndexListDataCallDescription;

}
}
}

#endif /* MEGAMOL_DATATOOLS_MULTIINDEXLISTDATACALL_H_INCLUDED */
