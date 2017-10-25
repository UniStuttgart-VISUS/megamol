#ifndef PBS_PBSDATACALL_H_INCLUDED
#define PBS_PBSDATACALL_H_INCLUDED

#include <vector>
#include <memory>

#include "pbs.h"
#include "mmcore/AbstractGetData3DCall.h"

#include "PBSStorage.h"

namespace megamol {
namespace pbs {

class PBS_API PBSDataCall : public core::AbstractGetData3DCall {
public:
    class NStorage : virtual public PBSStorage {
    public:
        NStorage() { };

        NStorage(const PBSStorage& rhs) {
            this->SetX(rhs.GetX());
            this->SetY(rhs.GetY());
            this->SetZ(rhs.GetZ());
            this->SetNX(rhs.GetNX());
            this->SetNY(rhs.GetNY());
            this->SetCR(rhs.GetCR());
            this->SetCG(rhs.GetCG());
            this->SetCB(rhs.GetCB());
        }

        inline virtual std::weak_ptr<std::vector<pbs_normal_t>> GetNX(void) const override {
            return this->nx;
        }

        inline virtual void SetNX(const std::weak_ptr<std::vector<pbs_normal_t>> nx) override {
            this->nx = nx;
        }

        inline virtual std::weak_ptr<std::vector<pbs_normal_t>> GetNY(void) const override {
            return this->ny;
        }

        inline virtual void SetNY(const std::weak_ptr<std::vector<pbs_normal_t>> ny) override {
            this->ny = ny;
        }
    private:
        std::weak_ptr<std::vector<pbs_normal_t>> nx, ny;
    };

    class CStorage : virtual public PBSStorage {
    public:
        CStorage() { };

        CStorage(const PBSStorage& rhs) {
            this->SetX(rhs.GetX());
            this->SetY(rhs.GetY());
            this->SetZ(rhs.GetZ());
            this->SetNX(rhs.GetNX());
            this->SetNY(rhs.GetNY());
            this->SetCR(rhs.GetCR());
            this->SetCG(rhs.GetCG());
            this->SetCB(rhs.GetCB());
        }

        inline virtual std::weak_ptr<std::vector<pbs_color_t>> GetCR(void) const override {
            return this->cr;
        }

        inline virtual void SetCR(const std::weak_ptr<std::vector<pbs_color_t>> cr) override {
            this->cr = cr;
        }

        inline virtual std::weak_ptr<std::vector<pbs_color_t>> GetCG(void) const override {
            return this->cg;
        }

        inline virtual void SetCG(const std::weak_ptr<std::vector<pbs_color_t>> cg) override {
            this->cg = cg;
        }

        inline virtual std::weak_ptr<std::vector<pbs_color_t>> GetCB(void) const override {
            return this->cb;
        }

        inline virtual void SetCB(const std::weak_ptr<std::vector<pbs_color_t>> cb) override {
            this->cb = cb;
        }
    private:
        std::weak_ptr<std::vector<pbs_color_t>> cr, cg, cb;
    };

    class PNCStorage : public NStorage, public CStorage {
    public:
        PNCStorage(const PBSStorage& rhs) {
            this->SetX(rhs.GetX());
            this->SetY(rhs.GetY());
            this->SetZ(rhs.GetZ());
            this->SetNX(rhs.GetNX());
            this->SetNY(rhs.GetNY());
            this->SetCR(rhs.GetCR());
            this->SetCG(rhs.GetCG());
            this->SetCB(rhs.GetCB());
        }

        PNCStorage(void) { };

        ~PNCStorage(void) { };
    private:
    };

    /*class PNStorage : public NStorage {
    public:
        PNStorage(void);

        ~PNStorage(void);
    private:
    };

    class PCStorage : public CStorage {
    public:
        PCStorage(void);

        ~PCStorage(void);
    private:
    };*/

    enum PBSType {
        PNC,
        PN,
        PC
    };

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "PBSDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call to transport PBS data";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return AbstractGetData3DCall::FunctionCount();
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return its name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        return AbstractGetData3DCall::FunctionName(idx);
    }

    /** ctor */
    PBSDataCall(void);

    /** dtor */
    virtual ~PBSDataCall(void);

    inline std::weak_ptr<PBSStorage> GetData(void) const {
        return this->data;
    }

    inline void SetData(const std::shared_ptr<PBSStorage>&& data) {
        this->data = data;
    }

    inline std::weak_ptr<double> GetGlobalBBox(void) const {
        return this->g_bbox;
    }

    inline void SetGlobalBBox(const std::weak_ptr<double> &bbox) {
        this->g_bbox = bbox;
    }

    inline std::weak_ptr<double> GetLocalBBox(void) const {
        return this->l_bbox;
    }

    inline void SetLocalBBox(const std::weak_ptr<double> &bbox) {
        this->l_bbox = bbox;
    }
protected:
private:
    std::shared_ptr<PBSStorage> data;

    std::weak_ptr<double> g_bbox;

    std::weak_ptr<double> l_bbox;
}; /* end class PBSDataCall */

} /* end namespace pbs */
} /* end namespace megamol */

#endif /* end ifndef PBS_PBSDATACALL_H_INCLUDED */