#include "datatools/MultiParticleDataAdaptor.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::datatools;

MultiParticleDataAdaptor::MultiParticleDataAdaptor(geocalls::MultiParticleDataCall& data)
        : data(data)
        , list(nullptr)
        , count(0) {
    using geocalls::MultiParticleDataCall;
    unsigned int plc = data.GetParticleListCount();

    list = new list_data[plc];

    for (unsigned int pli = 0; pli < plc; pli++) {
        list[pli].next = list + (pli + 1);

        auto& pl = data.AccessParticles(pli);
        if (!((((pl.GetVertexDataType() == MultiParticleDataCall::Particles::VERTDATA_NONE) ||
                  (pl.GetVertexDataType() == MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) ||
                  (pl.GetVertexDataType() == MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR))) &&
                ((pl.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_NONE) ||
                    (pl.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB) ||
                    (pl.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA) ||
                    (pl.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_FLOAT_I)))) {
            // data type we want to skip
            list[pli].count = 0;
            continue;
        }

        list[pli].count = static_cast<size_t>(pl.GetCount());
        if (list[pli].count < 0)
            list[pli].count = 0;
        if (list[pli].count == 0)
            continue;

        list[pli].pos_data = reinterpret_cast<const uint8_t*>(pl.GetVertexData());
        list[pli].pos_data_step = pl.GetVertexDataStride();
        if (pl.GetVertexDataType() == MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
            if (list[pli].pos_data_step < 12)
                list[pli].pos_data_step = 12;
        } else if (pl.GetVertexDataType() == MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
            if (list[pli].pos_data_step < 16)
                list[pli].pos_data_step = 16;
        }

        list[pli].col_data = reinterpret_cast<const uint8_t*>(pl.GetColourData());
        list[pli].col_data_step = pl.GetColourDataStride();
        if (pl.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
            if (list[pli].col_data_step < 4)
                list[pli].col_data_step = 4;
        }
        if (pl.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB) {
            if (list[pli].col_data_step < 12)
                list[pli].col_data_step = 12;
        } else if (pl.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA) {
            if (list[pli].col_data_step < 16)
                list[pli].col_data_step = 16;
        }

        count += list[pli].count;
    }

    if (plc > 0)
        list[plc - 1].next = nullptr;
}

MultiParticleDataAdaptor::~MultiParticleDataAdaptor() {
    if (list != nullptr) {
        delete[] list;
#if defined(DEBUG) || defined(_DEBUG)
        list = nullptr;
#endif
    }
#if defined(DEBUG) || defined(_DEBUG)
    count = 0;
#endif
}
