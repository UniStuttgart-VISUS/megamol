#ifndef MOLSURFMAPCLUSTER_TEXTURELOADER_INCLUDED
#define MOLSURFMAPCLUSTER_TEXTURELOADER_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "HierarchicalClustering.h"
#include "PNGPicLoader.h"
#include "vislib\sys\Log.h"

namespace megamol {
namespace MolSurfMapCluster {
namespace TextureLoader {

bool static loadPNGtoTexture(PNGPicLoader::PNGPIC* picture, GLubyte** texture) {

    unsigned int row_bytes = png_get_rowbytes(picture->png, picture->info);
    *texture = (unsigned char*)malloc(row_bytes * picture->height);

    png_bytepp row_pointers = picture->rows;

    for (int i = 0; i < picture->height; i++) {
        // note that png is ordered top to
        // bottom, but OpenGL expect it bottom to top
        // so the order or swapped
        memcpy(*texture + (row_bytes * (picture->height - 1 - i)), row_pointers[i], row_bytes);
    }

    return true;
}

bool static loadTexture(HierarchicalClustering::CLUSTERNODE* node) {
    // Lade Texturen auf Grapfikkarte
    glEnable(GL_TEXTURE_2D); // OpenGL Texturen aktivieren

    if (node->pic->texture == nullptr) {
        GLubyte* png;
        loadPNGtoTexture(node->pic, &png);

        node->pic->texture = new vislib::graphics::gl::OpenGLTexture2D();
        if (node->pic->texture->Create(node->pic->width, node->pic->height, false, png, GL_RGB) != GL_NO_ERROR) {
            vislib::sys::Log::DefaultLog.WriteError("Could not load \"%s\" texture.", node->pic->name);
            return false;
        }
        node->pic->texture->SetFilter(GL_LINEAR, GL_LINEAR);
    }
    return true;
}

void static loadTexturesToRender(HierarchicalClustering* clustering) {
    // Render Pictures
    for (HierarchicalClustering::CLUSTERNODE* leaf : *clustering->getLeaves()) {
        if (leaf->pic->render || leaf->pic->popup) {
            // Load Texture
            loadTexture(leaf);
        } else {
            if (leaf->pic->texture != nullptr) {
                // Destroy texture
                delete leaf->pic->texture;
                leaf->pic->texture = nullptr;
            }
        }
    }
}

bool static loadTextures(HierarchicalClustering::CLUSTERNODE* node, HierarchicalClustering* clustering) {
    // Search for Pictrues to be rendered
    for (HierarchicalClustering::CLUSTERNODE* leaf : *clustering->getLeaves()) {
        leaf->pic->render = false;
    }

    node->pic->render = true;
    std::vector<HierarchicalClustering::CLUSTERNODE*>* queue = clustering->getClusterNodesOfNode(node);
    // Add the actual level and the level after
    while (queue->size() > 0) {
        auto tmp = queue->back();
        queue->pop_back();

        // Set Picture to render
        tmp->pic->render = true;

        // Add new pictures which should be rendered
        if (tmp->clusterparent == node) {
            std::vector<HierarchicalClustering::CLUSTERNODE*>* tmpcluster = clustering->getClusterNodesOfNode(tmp);
            queue->insert(queue->end(), tmpcluster->begin(), tmpcluster->end());
        }
    }

    // Add the level befor when not root node
    if (node != clustering->getRoot()) {
        for (HierarchicalClustering::CLUSTERNODE* node : *clustering->getClusterNodesOfNode(node->clusterparent)) {
            node->pic->render = true;
        }
    }

    // Load Pictures to GPU
    loadTexturesToRender(clustering);

    return true;
}
} // namespace TextureLoader
} // namespace MolSurfMapCluster
} // namespace megamol

#endif /* MOLSURFMAPCLUSTER_TEXTURELOADER_INCLUDED */
