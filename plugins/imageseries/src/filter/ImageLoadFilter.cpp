#include "ImageLoadFilter.h"

#include "vislib/graphics/BitmapCodecCollection.h"

namespace megamol::ImageSeries::filter {

ImageLoadFilter::ImageLoadFilter(Input input) : input(input) {}

ImageLoadFilter::ImageLoadFilter(
    std::shared_ptr<vislib::graphics::BitmapCodecCollection> codecs, std::string filename, ImageMetadata metadata) {
    input.codecs = codecs;
    input.filename = filename;
    input.metadata = metadata;
}

ImageLoadFilter::ImagePtr ImageLoadFilter::operator()() {
    const char* filename = input.filename.c_str();
    try {
        util::PerfTimer timer("ImageLoadFilter", input.filename);

        auto img = std::make_shared<vislib::graphics::BitmapImage>();
        if (!input.codecs->LoadBitmapImage(*img, filename)) {
            throw vislib::Exception("No suitable codec found", __FILE__, __LINE__);
        }
        return img;
    } catch (vislib::Exception& ex) {
        // TODO thread-safe log?
        //Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load image '%s': %s (%s, %d)\n", filename, ex.GetMsgA(),
        //    ex.GetFile(), ex.GetLine());
    } catch (...) {
        // TODO thread-safe log?
        //Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load image '%s': unexpected exception\n", filename);
    }

    return nullptr;
}

ImageMetadata ImageLoadFilter::getMetadata() const {
    ImageMetadata metadata = input.metadata;
    metadata.hash = util::computeHash(input.filename);
    metadata.filename = input.filename;
    return metadata;
}

} // namespace megamol::ImageSeries::filter
