#ifndef MEGAMOLCORE_TEXTURE2DARRAY_H_INCLUDED
#define MEGAMOLCORE_TEXTURE2DARRAY_H_INCLUDED

#include "mmcore/utility/gl/Texture.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * \class Texture2DArray
 *
 * \brief Encapsulates 2D texture array functionality.
 *
 * \author Michael Becher
 */
class Texture2DArray : public Texture {
public:
    Texture2DArray(std::string id, TextureLayout const& layout, GLvoid* data, bool generateMipmap = false);
    Texture2DArray(const Texture2DArray&) =
        delete; // TODO: think of meaningful copy operation...maybe copy texture context to new texture object?
    Texture2DArray(Texture2DArray&& other) = delete;
    Texture2DArray& operator=(const Texture2DArray& rhs) = delete;
    Texture2DArray& operator=(Texture2DArray&& rhs) = delete;
    //~Texture2DArray();

    void bindTexture() const;

    void updateMipmaps();

    TextureLayout getTextureLayout() const;

    unsigned int getWidth() const;
    unsigned int getHeigth() const;
    unsigned int getLayers() const;

private:
    unsigned int m_width;
    unsigned int m_height;
    unsigned int m_layers;
};

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_TEXTURE2DARRAY_H_INCLUDED
