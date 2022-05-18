#include "DumpIColorHistogramModule.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"
#include <algorithm>
#include <fstream>
#include <vector>

using namespace megamol;


datatools::DumpIColorHistogramModule::DumpIColorHistogramModule(void)
        : core::Module()
        , inDataSlot("indata", "accessing the original data")
        , dumpBtnSlot("dump", "Dumps the data")
        , timeSlot("time", "The time to dump") {

    this->inDataSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->dumpBtnSlot.SetParameter(new core::param::ButtonParam());
    this->dumpBtnSlot.SetUpdateCallback(&DumpIColorHistogramModule::dump);
    this->MakeSlotAvailable(&this->dumpBtnSlot);

    this->timeSlot.SetParameter(new core::param::FloatParam(0.0f, 0.0f));
    this->MakeSlotAvailable(&this->timeSlot);
}


datatools::DumpIColorHistogramModule::~DumpIColorHistogramModule(void) {
    this->Release();
}


bool datatools::DumpIColorHistogramModule::create(void) {
    return true;
}


void datatools::DumpIColorHistogramModule::release(void) {}

void writeSVG(const std::vector<int>& buckets, float rangeMin, float rangeMax) {
    int width = 2048;
    int height = 768;

    std::ofstream svg("test.svg");
    svg << "<?xml version=\"1.0\" standalone=\"no\"?>" << std::endl
        << "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"" << std::endl
        << " \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">" << std::endl
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" viewBox=\"0 0 " << width << " " << height << "\">"
        << std::endl;

    svg << "<style type=\"text/css\" >" << std::endl
        << "<![CDATA[" << std::endl
        << ".axis { fill:none; stroke:black; stroke-width:1}" << std::endl
        << ".bar { fill:none; stroke:black; stroke-width:2}" << std::endl
        << ".bg {fill:white; stroke:none}" << std::endl
        << "]]>" << std::endl
        << "</style>" << std::endl;
    svg << "<rect class=\"bg\" width=\"" << width << "\" height=\"" << height << "\"/>" << std::endl;

    // Render legend
    svg << "<path class=\"axis\" d=\"M" << 0.1 * width << " " << 0.1 * height << " ";
    svg << "L" << 0.1 * width << " " << 0.9 * height << " ";
    svg << "L" << 0.9 * width << " " << 0.9 * height << "\"/>" << std::endl;

    // Render buckets
    float distance = 0.8f / static_cast<float>(buckets.size());
    // find maximum
    int m = -1000;
    for (int i = 0; i < static_cast<int>(buckets.size()); ++i) {
        m = std::max<int>(m, buckets[i]);
    }

    float scale = 0.8f / static_cast<float>(m);
    float x = 0.1f + 20.0f / (float)width;
    for (int i = 0; i < static_cast<int>(buckets.size()); ++i) {
        svg << "<path class=\"bar\" d=\"M" << x * width << " " << 0.9 * height << " L" << x * width << " "
            << (0.8 - (buckets[i] * scale) + 0.1) * height << "\"/>" << std::endl;
        float val = rangeMin + i / (float)(buckets.size() - 1) * (rangeMax - rangeMin);
        svg << "<text x=\"" << (x * width) << "\" y=\"" << 0.9 * height + 15
            << "\" fill=\"black\" text-anchor=\"middle\">" << val << "</text>" << std::endl;

        x += distance;
    }

    for (int i = 0; i <= 10; ++i) {
        svg << "<text x=\"" << (0.09 * width) << "\" y=\"" << (0.1 + i * (0.8 / 10.0)) * height
            << "\" fill=\"black\" text-anchor=\"end\">" << (10 - i) / 10.0 * m << "</text>" << std::endl;
    }

    svg << "</svg>" << std::endl;
}

bool datatools::DumpIColorHistogramModule::dump(::megamol::core::param::ParamSlot& param) {
    geocalls::MultiParticleDataCall* dat = this->inDataSlot.CallAs<geocalls::MultiParticleDataCall>();
    if (dat == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("No data connected");
        return true;
    }

    dat->SetFrameID(static_cast<unsigned int>(this->timeSlot.Param<core::param::FloatParam>()->Value()), true);
    if (!(*dat)(0)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("No data received");
        return true;
    }

    const unsigned int bucket_cnt = 25;
    ::std::vector<int> buckets(bucket_cnt, 0);

    for (unsigned int pli = 0; pli < dat->GetParticleListCount(); pli++) {
        const geocalls::MultiParticleDataCall::Particles& pl = dat->AccessParticles(pli);
        if (pl.GetColourDataType() != geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I)
            continue;
        const unsigned char* cp = static_cast<const unsigned char*>(pl.GetColourData());
        unsigned int stride = ::std::max<unsigned int>(4, pl.GetColourDataStride());
        float colRng = pl.GetMaxColourIndexValue() - pl.GetMinColourIndexValue();
        if (colRng <= 0.00001f)
            colRng = 1.0f;
        for (unsigned int pi = 0; pi < pl.GetCount(); pi++, cp += stride) {
            const float* f = reinterpret_cast<const float*>(cp);
            float fn = (*f - pl.GetMinColourIndexValue()) / colRng;
            int i = ::vislib::math::Clamp<int>(static_cast<int>(fn * bucket_cnt), 0, bucket_cnt - 1);
            buckets[i] = buckets[i] + 1;
        }
    }

    dat->Unlock();

    writeSVG(buckets, 0.0f, 1.0f);

    return true; // reset dirty flag
}
