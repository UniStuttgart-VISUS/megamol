#ifndef Texture3DView_hpp
#define Texture3DView_hpp

#include "mmcore/utility/gl/Texture.h"
#include "mmcore/utility/gl/Texture3D.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * \class TextureCubemapArray
 *
 * \brief Encapsulates 3D texture view functionality.
 *
 * \author Michael Becher
 */
class Texture3DView : public Texture {
public:
    Texture3DView(std::string id, Texture3D const& source_texture, TextureLayout const& layout, GLuint minlevel,
        GLuint numlevels, GLuint minlayer, GLuint numlayers);

    void bindTexture() const;

    void bindImage(GLuint location, GLenum access) const;

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

#endif // !Texture3DView_hpp
