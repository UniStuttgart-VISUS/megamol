#ifndef Texture2D_hpp
#define Texture2D_hpp

#include "mmcore/utility/gl/Texture.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * \class Texture2D
 *
 * \brief Encapsulates 2D texture functionality.
 *
 * \author Michael Becher
 */
class Texture2D : public Texture {
public:
    /**
     * \brief Constructor that creates and loads a 2D texture.
     *
     * \param id A identifier given to the texture object
     * \param layout A TextureLayout struct that specifies size, format and parameters for the texture
     * \param data Pointer to the actual texture data.
     * \param generateMipmap Specifies whether a mipmap will be created for the texture
     */
    Texture2D(std::string id, TextureLayout const& layout, GLvoid* data, bool generateMipmap = false);
    Texture2D(const Texture2D&) = delete;
    Texture2D(Texture2D&& other) = delete;
    Texture2D& operator=(const Texture2D& rhs) = delete;
    Texture2D& operator=(Texture2D&& rhs) = delete;

    /**
     * \brief Bind the texture.
     */
    void bindTexture() const;

    void updateMipmaps();

    /**
     * \brief Reload the texture with any new format, type and size.
     *
     * \param layout A TextureLayout struct that specifies size, format and parameters for the texture
     * \param data Pointer to the actual texture data.
     * \param generateMipmap Specifies whether a mipmap will be created for the texture
     */
    void reload(TextureLayout const& layout, GLvoid* data, bool generateMipmap = false);

    TextureLayout getTextureLayout() const;

    unsigned int getWidth() const;

    unsigned int getHeight() const;

private:
    unsigned int m_width;
    unsigned int m_height;
};

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif // !Texture2D_hpp
