#ifndef PBS_PBSSTORAGE_H_INCLUDED
#define PBS_PBSSTORAGE_H_INCLUDED

#include <vector>
#include <memory>

namespace megamol {
namespace pbs {

class PBSStorage {
public:
    typedef double pbs_coord_t;
    typedef float pbs_normal_t;
    typedef unsigned char pbs_color_t;

    PBSStorage() { };

    virtual ~PBSStorage(void) { };

    inline unsigned int GetNumElements(void) const {
        return this->num_elements;
    }

    inline void SetNumElements(const unsigned int num_elements) {
        this->num_elements = num_elements;
    }

    inline std::weak_ptr<std::vector<bool>> GetRenderableFlags(void) const {
        return this->renderable_flags;
    }

    inline void SetRenderableFlags(const std::shared_ptr<std::vector<bool>> renderable_flags) {
        this->renderable_flags = renderable_flags;
    }

    inline virtual std::weak_ptr<std::vector<pbs_coord_t>> GetX(void) const {
        return this->x;
    }

    inline virtual void SetX(const std::weak_ptr<std::vector<pbs_coord_t>> x) {
        this->x = x;
    }

    inline virtual std::weak_ptr<std::vector<pbs_coord_t>> GetY(void) const {
        return this->y;
    }

    inline virtual void SetY(const std::weak_ptr<std::vector<pbs_coord_t>> y) {
        this->y = y;
    }

    inline virtual std::weak_ptr<std::vector<pbs_coord_t>> GetZ(void) const {
        return this->z;
    }

    inline virtual void SetZ(const std::weak_ptr<std::vector<pbs_coord_t>> z) {
        this->z = z;
    }

    inline virtual std::weak_ptr<std::vector<pbs_normal_t>> GetNX(void) const {
        return std::weak_ptr<std::vector<pbs_normal_t>>();
    }

    inline virtual void SetNX(const std::weak_ptr<std::vector<pbs_normal_t>> nx) {
        // intentionally empty
    }

    inline virtual std::weak_ptr<std::vector<pbs_normal_t>> GetNY(void) const {
        return std::weak_ptr<std::vector<pbs_normal_t>>();
    }

    inline virtual void SetNY(const std::weak_ptr<std::vector<pbs_normal_t>> ny) {
        // intentionally empty
    }

    inline virtual std::weak_ptr<std::vector<pbs_color_t>> GetCR(void) const {
        return std::weak_ptr<std::vector<pbs_color_t>>();
    }

    inline virtual void SetCR(std::weak_ptr<std::vector<pbs_color_t>> cr) {
        // intentionally empty
    }

    inline virtual std::weak_ptr<std::vector<pbs_color_t>> GetCG(void) const {
        return std::weak_ptr<std::vector<pbs_color_t>>();
    }

    inline virtual void SetCG(std::weak_ptr<std::vector<pbs_color_t>> cg) {
        // intentionally empty
    }

    inline virtual std::weak_ptr<std::vector<pbs_color_t>> GetCB(void) const {
        return std::weak_ptr<std::vector<pbs_color_t>>();
    }

    inline virtual void SetCB(std::weak_ptr<std::vector<pbs_color_t>> cb) {
        // intentionally empty
    }
private:
    unsigned int num_elements;

    std::weak_ptr<std::vector<pbs_coord_t>> x, y, z;

    std::weak_ptr<std::vector<bool>> renderable_flags;
}; /* end class PBSStorage */

} /* end namespace pbs */
} /* end namespace megamol */

#endif /* end ifndef PBS_PBSSTORAGE_H_INCLUDED */