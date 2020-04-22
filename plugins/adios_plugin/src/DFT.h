#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "fftw3.h"

#include "mmstd_datatools/table/TableDataCall.h"

namespace megamol {
namespace adios {

class FFTWArrayC {
public:
    FFTWArrayC(size_t n) : data_(fftwf_alloc_complex(n)) {}

    FFTWArrayC(FFTWArrayC const& rhs) = delete;
    FFTWArrayC& operator=(FFTWArrayC const& rhs) = delete;

    ~FFTWArrayC() { fftwf_free(data_); }

    operator fftwf_complex*() { return data_; }

    fftwf_complex& operator[](size_t idx) { return data_[idx]; }

private:
    fftwf_complex* data_;
};

class FFTWArrayR {
public:
    FFTWArrayR(size_t n) : data_(fftwf_alloc_real(n)) {}

    FFTWArrayR(FFTWArrayR const& rhs) = delete;
    FFTWArrayR& operator=(FFTWArrayR const& rhs) = delete;

    ~FFTWArrayR() { fftwf_free(data_); }

    operator float*() { return data_; }

private:
    float* data_;
};

class FFTWPlan1D {
public:
    FFTWPlan1D(int n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned int flags) {
        plan_ = fftwf_plan_dft_1d(n, in, out, sign, flags);
    }

    FFTWPlan1D(int n, float* in, fftwf_complex* out, unsigned int flags) {
        plan_ = fftwf_plan_dft_r2c_1d(n, in, out, flags);
    }

    FFTWPlan1D(int n, fftwf_complex* in, float* out, unsigned int flags) {
        plan_ = fftwf_plan_dft_c2r_1d(n, in, out, flags);
    }

    FFTWPlan1D(FFTWPlan1D const& rhs) = delete;
    FFTWPlan1D& operator=(FFTWPlan1D const& rhs) = delete;

    ~FFTWPlan1D() { fftwf_destroy_plan(plan_); }

    void Execute() { fftwf_execute(plan_); }

private:
    fftwf_plan plan_;
};

class DFT : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "DFT"; }

    /** Return module class description */
    static const char* Description(void) { return "Discrete Fourier transform on adios content"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    /** Ctor */
    DFT(void);

    /** Dtor */
    virtual ~DFT(void);

protected:
    /** Lazy initialization of the module */
    bool create(void) override;

    /** Resource release */
    void release(void) override;

private:
    bool getDataCallback(core::Call& c);
    bool getHeaderCallback(core::Call& c);

    void fillInfoVector(stdplugin::datatools::table::TableDataCall::ColumnInfo const* infos, size_t num_columns) {
        for (size_t col = 0; col < num_columns; ++col) {
            infos_[col * 2 + 0] = infos[col];
            infos_[col * 2 + 0].SetName(infos_[col * 2 + 0].Name() + "_re");
            infos_[col * 2 + 1] = infos[col];
            infos_[col * 2 + 1].SetName(infos_[col * 2 + 1].Name() + "_im");
        }
    }

    core::CalleeSlot data_out_slot_;
    core::CallerSlot data_in_slot_;

    //std::shared_ptr<adiosDataMap> data_map_;

    std::vector<float> data_;
    std::vector<stdplugin::datatools::table::TableDataCall::ColumnInfo> infos_;
    size_t out_num_rows_;
    size_t out_num_columns_;

    size_t data_hash_;

}; // class DFT

} // namespace adios
} // namespace megamol