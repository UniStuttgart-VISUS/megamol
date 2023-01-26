/*
 * ParticleFilterMapDataCall.h
 *
 * Copyright (C) 2016 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetDataCall.h"
#include <cstdint>

namespace megamol {
namespace datatools {

class ParticleFilterMapDataCall : public core::AbstractGetDataCall {
public:
    typedef uint32_t index_t;

    enum CallFunctionName : int {
        GET_DATA = 0,
        GET_EXTENT = 1, /* temporal */
        GET_HASH = 2
    };

    static const char* ClassName(void) {
        return "ParticleFilterMapDataCall";
    }
    static const char* Description(void) {
        return "Map data, i.e. an array of length of the filtered particles with the indices into the original "
               "unfiltered particles data";
    }
    static unsigned int FunctionCount(void) {
        return 3;
    }
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case GET_DATA:
            return "GetData";
        case GET_EXTENT:
            return "GetExtent";
        case GET_HASH:
            return "GetHash";
        }
        return nullptr;
    }

    ParticleFilterMapDataCall(void);
    ~ParticleFilterMapDataCall(void) override;

    inline index_t* Data() const {
        return idx;
    }
    inline size_t Size() const {
        return idx_len;
    }
    inline unsigned int FrameID() const {
        return frameID;
    }
    inline unsigned int FrameCount() const {
        return frameCnt;
    }

    inline void Set(index_t* data, size_t cnt) {
        idx = data;
        idx_len = cnt;
    }
    inline void SetFrameID(unsigned int fid) {
        frameID = fid;
    }
    inline void SetFrameCount(unsigned int cnt) {
        frameCnt = cnt;
    }

private:
    index_t* idx;
    size_t idx_len;
    unsigned int frameCnt;
    unsigned int frameID;
};

typedef core::factories::CallAutoDescription<ParticleFilterMapDataCall> ParticleFilterMapDataCallDescription;

} /* end namespace datatools */
} /* end namespace megamol */
