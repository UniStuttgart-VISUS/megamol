#ifndef MOLSURFMAPCLUSTER_PICTUREDATA_INCLUDED
#define MOLSURFMAPCLUSTER_PICTUREDATA_INCLUDED

#include "image_calls/Image2DCall.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/graphics/BitmapImage.h"
#include <string>

namespace megamol {
namespace MolSurfMapCluster {

    struct PictureData {
        std::string path;
        std::string pdbid;
        uint32_t width;
        uint32_t height;
        bool render;
        bool popup;
        vislib::graphics::gl::OpenGLTexture2D* texture;
        vislib::graphics::BitmapImage* image;
    };

}
} // namespace megamol

#endif
