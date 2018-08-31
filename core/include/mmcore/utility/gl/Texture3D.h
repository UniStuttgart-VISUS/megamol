#ifndef MEGAMOLCORE_TEXTURE3D_H_INCLUDED
#define MEGAMOLCORE_TEXTURE3D_H_INCLUDED

#include "mmcore/utility/gl/Texture.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * \class Texture3D
 *
 * \brief Encapsulates basic 3D texure functionality.
 *
 * This class encapsulates basic 3D functionality including creation of a 3D texture,
 * texture updates and texture binding.
 *
 * \author Michael Becher
 */
class Texture3D : public Texture {
public:
    Texture3D(std::string id, TextureLayout const& layout, GLvoid* data);
    Texture3D(const Texture3D&) = delete;
    Texture3D(Texture3D&& other) = delete;
    Texture3D& operator=(const Texture3D& rhs) = delete;
    Texture3D& operator=(Texture3D&& rhs) = delete;

    /**
     * \brief Bind the texture.
     */
    void bindTexture() const;

    void updateMipmaps();

    /**
     * \brief Reload the texture.
     * \param data Pointer to the new texture data.
     */
    void reload(TextureLayout const& layout, GLvoid* data);

    TextureLayout getTextureLayout() const;

    unsigned int getWidth();
    unsigned int getHeight();
    unsigned int getDepth();

private:
    unsigned int m_width;
    unsigned int m_height;
    unsigned int m_depth;
};

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_TEXTURE3D_H_INCLUDED
