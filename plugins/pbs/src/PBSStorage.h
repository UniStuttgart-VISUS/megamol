#ifndef PBS_PBSSTORAGE_H_INCLUDED
#define PBS_PBSSTORAGE_H_INCLUDED

#include <vector>
#include <memory>

namespace megamol {
namespace pbs {

class PBSStorage {
public:
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

    inline virtual std::weak_ptr<std::vector<double>> GetX(void) const {
        return this->x;
    }

    inline virtual void SetX(const std::weak_ptr<std::vector<double>> x) {
        this->x = x;
    }

    inline virtual std::weak_ptr<std::vector<double>> GetY(void) const {
        return this->y;
    }

    inline virtual void SetY(const std::weak_ptr<std::vector<double>> y) {
        this->y = y;
    }

    inline virtual std::weak_ptr<std::vector<double>> GetZ(void) const {
        return this->z;
    }

    inline virtual void SetZ(const std::weak_ptr<std::vector<double>> z) {
        this->z = z;
    }

    inline virtual std::weak_ptr<std::vector<float>> GetNX(void) const {
        return std::weak_ptr<std::vector<float>>();
    }

    inline virtual void SetNX(const std::weak_ptr<std::vector<float>> nx) {
        // intentionally empty
    }

    inline virtual std::weak_ptr<std::vector<float>> GetNY(void) const {
        return std::weak_ptr<std::vector<float>>();
    }

    inline virtual void SetNY(const std::weak_ptr<std::vector<float>> ny) {
        // intentionally empty
    }

    inline virtual std::weak_ptr<std::vector<unsigned int>> GetCR(void) const {
        return std::weak_ptr<std::vector<unsigned int>>();
    }

    inline virtual void SetCR(std::weak_ptr<std::vector<unsigned int>> cr) {
        // intentionally empty
    }

    inline virtual std::weak_ptr<std::vector<unsigned int>> GetCG(void) const {
        return std::weak_ptr<std::vector<unsigned int>>();
    }

    inline virtual void SetCG(std::weak_ptr<std::vector<unsigned int>> cg) {
        // intentionally empty
    }

    inline virtual std::weak_ptr<std::vector<unsigned int>> GetCB(void) const {
        return std::weak_ptr<std::vector<unsigned int>>();
    }

    inline virtual void SetCB(std::weak_ptr<std::vector<unsigned int>> cb) {
        // intentionally empty
    }
private:
    unsigned int num_elements;

    std::weak_ptr<std::vector<double>> x, y, z;

    std::weak_ptr<std::vector<bool>> renderable_flags;
}; /* end class PBSStorage */

} /* end namespace pbs */
} /* end namespace megamol */

#endif /* end ifndef PBS_PBSSTORAGE_H_INCLUDED */