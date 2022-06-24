#ifndef MOLSURFMAPCLUSTER_PICTUREDATA_INCLUDED
#define MOLSURFMAPCLUSTER_PICTUREDATA_INCLUDED

#include "glowl/glowl.h"
#include "image_calls/Image2DCall.h"
#include "vislib/graphics/BitmapImage.h"
#include "vislib_gl/graphics/gl/OpenGLTexture2D.h"
#include <string>

namespace megamol {
namespace molsurfmapcluster {

struct PictureData {
    std::string path;
    std::string pdbid;
    uint32_t width;
    uint32_t height;
    bool render;
    bool popup;
    float minValue;
    float maxValue;
    //vislib::graphics::gl::OpenGLTexture2D* texture;
    std::unique_ptr<glowl::Texture2D> texture;
    vislib::graphics::BitmapImage* image;
    std::vector<float> valueImage;
};

} // namespace molsurfmapcluster
} // namespace megamol

#endif
