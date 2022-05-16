#ifndef MOLSURFMAPCLUSTER_TEXTURELOADER_INCLUDED
#define MOLSURFMAPCLUSTER_TEXTURELOADER_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "HierarchicalClustering.h"
#include "PictureData.h"
#include "mmcore/utility/log/Log.h"
#include "vislib_gl/graphics/gl/OpenGLTexture2D.h"

#include <glad/gl.h>
#include <iostream>

namespace megamol {
namespace molsurfmapcluster {
namespace TextureLoader {

bool static loadTexture(HierarchicalClustering::CLUSTERNODE* node) {
    // Lade Texturen auf Grapfikkarte
    glEnable(GL_TEXTURE_2D); // OpenGL Texturen aktivieren

    if (node->pic->texture == nullptr) {
        glowl::TextureLayout layout(GL_RGB8, node->pic->width, node->pic->height, 1, GL_RGB, GL_UNSIGNED_BYTE, 1);
        node->pic->texture = std::make_unique<glowl::Texture2D>("", layout, node->pic->image->PeekDataAs<BYTE>());
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
                leaf->pic->texture.reset();
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
} // namespace molsurfmapcluster
} // namespace megamol

#endif /* MOLSURFMAPCLUSTER_TEXTURELOADER_INCLUDED */
