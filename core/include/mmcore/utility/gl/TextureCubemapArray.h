#ifndef TextureCubemapArray_hpp
#define TextureCubemapArray_hpp

#include "mmcore/utility/gl/Texture.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * \class TextureCubemapArray
 *
 * \brief Encapsulates cubemap texture array functionality.
 *
 * \author Michael Becher
 */
class TextureCubemapArray : public Texture {
public:
    TextureCubemapArray(std::string id, GLint internal_format, unsigned int width, unsigned int height,
        unsigned int layers, GLenum format, GLenum type, GLsizei levels, GLvoid* data, bool generateMipmap = false);
    TextureCubemapArray(const TextureCubemapArray&) =
        delete; // TODO: think of meaningful copy operation...maybe copy texture context to new texture object?
    TextureCubemapArray(TextureCubemapArray&& other) = delete;
    TextureCubemapArray& operator=(const TextureCubemapArray& rhs) = delete;
    TextureCubemapArray& operator=(TextureCubemapArray&& rhs) = delete;
    //~Texture2DArray();

    /**
     * \brief Reload the texture with a new size but unchanged format and type.
     * \param width Specifies the new width of the texture in pixels.
     * \param height Specifies the new height of the texture in pixels.
     * \param layers Specifies the new number of layers in the texture array.
     * \param data Pointer to the new texture data.
     * \return Returns true if the texture was succesfully created, false otherwise
     */
    bool reload(
        unsigned int width, unsigned int height, unsigned int layers, GLvoid* data, bool generateMipmap = false);

    void bindTexture() const;

    void updateMipmaps();

    void texParameteri(GLenum pname, GLenum param);

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

#endif // !TextureCubemapArray_hpp
