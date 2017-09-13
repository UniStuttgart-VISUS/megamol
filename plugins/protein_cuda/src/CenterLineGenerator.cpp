#include "stdafx.h"
#include "CenterLineGenerator.h"
#include <float.h>
#include "vislib/Array.h"
#include <time.h>
#include "vislib/sys/PerformanceCounter.h"

CenterLineGenerator::CenterLineGenerator(void) {
}


CenterLineGenerator::~CenterLineGenerator(void) {
    delete[] theNodes;
    delete[] theEdges;
}

void CenterLineGenerator::SetTriangleMesh( unsigned int meshVertexCnt, float* meshVertices, unsigned int meshEdgeCnt, unsigned int *meshEdges, unsigned int *meshFreeEdges) {
    // temp variables
    vislib::sys::PerformanceCounter perf(true);
    
    // clear previous data
    for( auto edge : edges ) {
        if( edge ) delete edge;
    }
    for( auto node : nodes ) {
        if( node ) delete node;
    }
    for( auto branch : allBranches ) {
        if( branch ) delete branch;
    }
    edges.clear();
	nodes.clear();
    allBranches.clear();
    freeEdgeRing.clear();
    freeEdgeRing.clear();    

    perf.SetMark();
    theNodes = new Node[meshVertexCnt];
    //Node **tmpNodePtr = new Node*[meshVertexCnt];
    Node *tmpNodePtr;
    Node **nodesPtr = this->nodes.get_allocator().allocate(meshVertexCnt);
    // loop over all triangles
    for( unsigned int vCnt = 0; vCnt < meshVertexCnt; vCnt++) {
        //tmpNodePtr[vCnt] = new Node();
        tmpNodePtr = theNodes + vCnt;
        // get the vertex
        //tmpNodePtr[vCnt]->p.Set( meshVertices[4 * vCnt + 0], meshVertices[4 * vCnt + 1], meshVertices[4 * vCnt + 2]);
        tmpNodePtr->p.Set( meshVertices[4 * vCnt + 0], meshVertices[4 * vCnt + 1], meshVertices[4 * vCnt + 2]);
        // insert the vertex into the nodes list
        //this->nodes.push_back( tmpNodePtr[vCnt]);
        //this->nodes.push_back( tmpNodePtr);
        nodesPtr[vCnt] = tmpNodePtr;
    }
    INT64 t2 = perf.Difference();
    //printf( "     Time to insert mesh vertices into node list:             (%.5f)\n", vislib::sys::PerformanceCounter::ToMillis(t2) / 1000.0f);
    
    perf.SetMark();
    theEdges = new Edge[meshEdgeCnt];
    Edge **edgesPtr = this->edges.get_allocator().allocate(meshEdgeCnt);
    // list for free edges
    Edges freeEdges;
    for( unsigned int eCnt = 0; eCnt < meshEdgeCnt; eCnt++ ) {
        // construct edge and insert it into the edge list
        //this->edges.push_back( new Edge());
        edgesPtr[eCnt] = theEdges + eCnt;
        //this->edges.back()->setNodes( tmpNodePtr[meshEdges[2*eCnt]], tmpNodePtr[meshEdges[2*eCnt+1]]);
        //this->edges.back()->setNodes( &testNodes[meshEdges[2*eCnt]], &testNodes[meshEdges[2*eCnt+1]]);
        edgesPtr[eCnt]->setNodes( &theNodes[meshEdges[2*eCnt]], &theNodes[meshEdges[2*eCnt+1]]);
        // add the edges to both nodes
        //tmpNodePtr[meshEdges[2*eCnt]]->edges.push_back( this->edges.back());
        //tmpNodePtr[meshEdges[2*eCnt+1]]->edges.push_back( this->edges.back());
        //testNodes[meshEdges[2*eCnt]].edges.push_back( this->edges.back());
        //testNodes[meshEdges[2*eCnt+1]].edges.push_back( this->edges.back());
        //testNodes[meshEdges[2*eCnt]].edges.push_back( edgesPtr[eCnt]);
        //testNodes[meshEdges[2*eCnt+1]].edges.push_back( edgesPtr[eCnt]);
        theNodes[meshEdges[2*eCnt]].edges[theNodes[meshEdges[2*eCnt]].edgeCnt++] = edgesPtr[eCnt];
        theNodes[meshEdges[2*eCnt+1]].edges[theNodes[meshEdges[2*eCnt+1]].edgeCnt++] = edgesPtr[eCnt];
        // add edge to free edges if it is free!
        if( meshFreeEdges[eCnt] == 1 ) {
            //freeEdges.push_back( this->edges.back());
            freeEdges.push_back( edgesPtr[eCnt]);
        }
    }
    
    // delete temporary array
    //delete[] tmpNodePtr;
    INT64 t3 = perf.Difference();
    //printf( "     Time to insert mesh edges into edge list:                (%.5f)\n", vislib::sys::PerformanceCounter::ToMillis(t3) / 1000.0f);

    perf.SetMark();
    Edges tmpFreeEdgeRing;
    // TODO make faster!!
    //this->freeEdgeRing.clear();
    while( freeEdges.size() > 0 ) {
        tmpFreeEdgeRing.clear();
        Edge *e = freeEdges.front();
        while( e != nullptr ) {
            //this->freeEdgeRing.push_back( e);
            tmpFreeEdgeRing.push_back( e);
            e = findEdgeNeighborInSet( e, freeEdges, true);
        }
        this->freeEdgeRing.push_back( tmpFreeEdgeRing);
    }

    // classify feature mesh
    if( this->freeEdgeRing.empty() ) {
        this->fType = CAVITY;
    //} else if( freeEdges.empty() ) {
    } else if( freeEdgeRing.size() == 1 ) {
        this->fType = POCKET;
    } else {
        this->fType = CHANNEL;
    }
    INT64 t4 = perf.Difference();
    //printf( "     Time to find free edge rings and classify feature:       (%.5f)\n", vislib::sys::PerformanceCounter::ToMillis(t4) / 1000.0f);
}

void CenterLineGenerator::NodesFromEdges( Edges &edges, NodeSet &nodes) {
	for(auto edge : edges) {
        nodes.insert( edge->getNode1());
        nodes.insert( edge->getNode2());
	}
}

CenterLineGenerator::CenterLineNode* CenterLineGenerator::Collapse(NodeSet &selection) {
	CenterLineGenerator::Vector v( 0.0f, 0.0f, 0.0f);

	for(auto it : selection) {
		v += it->p;
        it->visited = true;
    }
    v /= static_cast<float>(selection.size());

    float minimumDistance =  FLT_MAX;
	for(auto it : selection)
		minimumDistance = vislib::math::Min(minimumDistance, (it->p - v).Length());

    CenterLineNode *node = new CenterLineNode(v, minimumDistance);
	return node;
}

void CenterLineGenerator::NextSection(Edges &current, NodeSet &currentNodes, Section *next)
{
/*
	1. Input:
		Mesh M
		S0: Set of nodes which define the current section	= current
		E0: Set of the elements already visited				= visited
	2. Create E1, set of the elements which have nodes in S0 but do not belong to E0
	3. Create A1, set of the edges of the elements in E1
	4. Create A2, subset of A1 composed by the edges which do not have any node in S0
	5. Update E0 with the elements which have the edges that belong to A2
	6. Return A2, set of the edges of the new section.
*/

	// 2) + 3)(?)
    EdgeSet E1;
    NodeSet N1; // non-visited neigbor nodes of S0
    EdgeSet A2;
    // get all connected edges for all current nodes (one shared node)
    for( auto node : currentNodes ) {
        //for( auto edge : node->edges ) {
        for( unsigned int i = 0; i < node->edgeCnt; i++ ) {
            auto edge = node->edges[i];
            // if other node is not in current node set and edge not visited: add edge to E1
            // all nodes in E1 are now counting as visited
            if( edge->getNode1() == node && !edge->getNode2()->visited ) {
                edge->visited = true;
                E1.insert( edge );
                N1.insert( edge->getNode2());
            } else if( edge->getNode2() == node && !edge->getNode1()->visited ) {
                edge->visited = true;
                E1.insert( edge );
                N1.insert( edge->getNode1());
            } 
        }
    }
    // 4)
    for( auto node : N1 ) {
        //for( auto edge : node->edges ) {
        for( unsigned int i = 0; i < node->edgeCnt; i++ ) {
            auto edge = node->edges[i];
            // both nodes have to be in N1
            if( edge->getNode1() == node && N1.find( edge->getNode2()) != N1.end() ) {
                A2.insert( edge);
            } else if( edge->getNode2() == node && N1.find( edge->getNode1()) != N1.end() ) {
                A2.insert( edge);
            }
        }
    }
    // copy edges from set A2 to next section edge list
    next->edges.clear();
    for( auto e : A2 ) {
        next->edges.push_back( e);
    }
}

CenterLineGenerator::Edge *CenterLineGenerator::findEdgeNeighborInSet(Edge *edge, Edges &set, bool removeFromSet) {
	Edge *retEdge = nullptr;
    Node *node1 = edge->getNode1();
    Node *node2 = edge->getNode2();
	for(auto it = set.begin(); it != set.end(); ++it) {
		retEdge = *it;
        // check whether a node of the current edge 'retEdge' matches to one of the nodes of 'edge'
        if( retEdge->getNode1() == node1 || retEdge->getNode1() == node2 ||
            retEdge->getNode2() == node1 || retEdge->getNode2() == node2 )
        {
			if( removeFromSet ) {
				set.erase(it);
			}
			return retEdge;
		}
	}
    // no match found - return null pointer
    return nullptr;
}

CenterLineGenerator::Node *CenterLineGenerator::nodeSharedByEdges(Edge *e1, Edge *e2) {
    if( e1->getNode1() == e2->getNode1() )
        return e1->getNode1();
    if( e1->getNode1() == e2->getNode2() )
        return e1->getNode1();
    if( e1->getNode2() == e2->getNode1() )
        return e1->getNode2();
    if( e1->getNode2() == e2->getNode2() )
        return e1->getNode2();

	return nullptr;
}

void CenterLineGenerator::FindBranch(Section *current, std::list<Section*> &branches) {
/*
	1. Input:
		A0 : Set of edges of the current section	= current
		S0 : Set of edges which define a section (empty, initially)
		S : Set of edges found in A0 (empty, initially)
	2. a0 = First non-visited edge in A0
	3. Move a0 from A0 to S0
	4. Find a1, neighbour to a0 in A0
	5. Move a1 from A0 to S0
	6. Find anext, neighbour to a1 in A0
	7. Is anext adjacent to a0?
	No:
		Move anext from A0 to S0
		a1 = anext and go to step 5
	Yes:
		Move anext from A0 to S0
		Insert S0 in S
		Start new section (S0 = ;)
		Go to step 1
	8. Repeat steps 1 to 7 until all edges have visited.
	9. Return S, the set of sections found in the input data.
*/
	// 1)
	bool continueOnStep6 = false;

	// if current is empty and the first edge has not been reached, it is a open loop:
	// save for later
	// check every edge if it connects to one of the open loops...
    
	Section *S0;
	Edge *edge = nullptr;
	Edge *edgeNext = nullptr;
    Edge *edgeNextNext = nullptr;
    std::list<Edge*> edgeVec;
    
    while( !current->edges.empty() ) {
        S0 = new Section();
        S0->centerLineNode = nullptr;
        S0->prevCenterLineNode = current->prevCenterLineNode;
        // get first element for new ring
        Edge *e = current->edges.front();
        current->edges.pop_front();

        edgeVec.clear();
        edgeVec.push_back( e);
        while( !edgeVec.empty() ) {
            // add all neighbors of the current first edge
            while( edgeVec.back() != nullptr ) {
                edgeVec.push_back( findEdgeNeighborInSet( edgeVec.front(), current->edges, true));
            }
            // the last element is a nullptr -> remove it!
            edgeVec.pop_back();
            // insert first element of the edge list to S0 and remove it from the edge list
            S0->edges.push_back( edgeVec.front());
            edgeVec.pop_front();
        }

//#define RING_TEST
#ifdef RING_TEST
        // remove dangling edges
        NodeSet nodes;
        this->NodesFromEdges( S0->edges, nodes);
        std::vector<Node*> nodeVec;
        for( auto e : S0->edges ) {
            nodeVec.push_back( e->getNode1());
            nodeVec.push_back( e->getNode2());
        }
        unsigned int counter;
        for( auto n : nodes ) {
            counter = 0;
            for( auto n0 : nodeVec ) {
                if( n == n0 ) {
                    counter++;
                }
            }
            if( counter == 1 ) {
                for( auto e = S0->edges.begin(); e != S0->edges.end(); e++ ) {
                    if( (*e)->getNode1() == n || (*e)->getNode2() == n ) {
                        S0->edges.erase( e);
                        break;
                    }
                }
            }
        }
        // check ring
        nodeVec.clear();
        for( auto e : S0->edges ) {
            nodeVec.push_back( e->getNode1());
            nodeVec.push_back( e->getNode2());
        }
        bool isRing = true;
        for( auto n : nodes ) {
            counter = 0;
            for( auto n0 : nodeVec ) {
                if( n == n0 ) {
                    counter++;
                }
            }
            if( counter == 1 ) {
                isRing = false;
            }
        }
        // ring found -> add to branch list
        S0->isRing = isRing;
        if( isRing ) {
            branches.push_back( S0);
        } else {
            if( current->edges.empty() && !S0->edges.empty() ) {
                // HACK only add non-ring if it is the only segment
                branches.push_back( S0);
            } else {
                return;
            }
        }
#else
        branches.push_back( S0);
#endif // RING_TEST
    }

}

void CenterLineGenerator::CenterLine(Edges &selection, CenterLineEdges &centerLineEdges, CenterLineNodes &centerLineNodes, float minCenterLineNodeDist) {
	/*
	1. Input:
		Mesh M
		Initial Section S0
	2. Calculate C0, Centroid of S0
	3. S1 =NextSection(S0)
	4. vS =FindBranch(S1)
	5. Does vS have ramifications?
		No:
			S1 = vS0
			Calculate C1, Centroid of S1
			Create element E0 with nodes C0 e C1
			If S1 is the last section, end.
			Else, S0 = S1 and go to step 3.
		Yes:
			For each branch vSi, S0 = vSi and go to step 2
	*/

    std::list<Section*> branches;
    CenterLineNodes freeCLNodes;
    
    //branches.push_back( new Section());
    //for( auto e : selection ) {
    //    branches[0]->edges.push_back( e);
    //}
    //branches[0]->centerLineNode = nullptr;
    //branches[0]->prevCenterLineNode = nullptr;

    // Add all free edge rings to the branch list
    for( auto fE : this->freeEdgeRing ) {
        branches.push_back( new Section());
        for( auto e : fE ) {
            branches.back()->edges.push_back( e);
        }
        branches.back()->centerLineNode = nullptr;
        branches.back()->prevCenterLineNode = nullptr;
    }
    
	while( !branches.empty() ) {
		Section *current = branches.front();
        branches.pop_front();
        
        // DEBUG
        allBranches.push_back(current);

		// 2)
		NodeSet currentNodes;
        this->NodesFromEdges( current->edges, currentNodes);
        // set all current edges as visited
        for( auto edge : current->edges) {
            edge->visited = true;
        }
        if( currentNodes.empty() ) continue;

        float meanEdgeLength = 0;
        for( auto cle : centerLineEdges ) {
            meanEdgeLength += (cle->node1->p - cle->node2->p).Length();
        }
        meanEdgeLength /= centerLineEdges.size();

		current->centerLineNode = Collapse( currentNodes);
        current->centerLineNode->isRing = current->isRing;
        // do not follow branch if it is too narrow
        if( current->centerLineNode->minimumDistance < minCenterLineNodeDist ) {
            delete current->centerLineNode;
            continue;
        }
        centerLineNodes.push_back( current->centerLineNode);
        
        // merge center lines
        bool clMerged = false;
        // construct center line edge
        if( current->prevCenterLineNode != nullptr ) {
            CenterLineEdge *cle = new CenterLineEdge();
            cle->node1 = current->prevCenterLineNode;
            cle->node2 = current->centerLineNode;
            centerLineEdges.push_back( cle);
            // merge center lines
            for( auto it = freeCLNodes.begin(); it != freeCLNodes.end(); it++ ) {
                if( (*it) == current->prevCenterLineNode ) {
                    freeCLNodes.erase( it);
                    break;
                }
            }
            for( auto n : freeCLNodes ) {
                if( (current->centerLineNode->p - n->p).Length() < (3.0f * meanEdgeLength) ) {
                    cle = new CenterLineEdge();
                    cle->node1 = n;
                    cle->node2 = current->centerLineNode;
                    centerLineEdges.push_back( cle);
                    for( auto b = branches.begin(); b != branches.end(); ) {
                        if( (*b)->prevCenterLineNode == n ) {
                            b = branches.erase( b);
                        } else {
                            b++;
                        }
                    }
                    clMerged = true;
                }
            }
            if( !clMerged ) {
                freeCLNodes.push_back( current->centerLineNode);
            }
        } else {
            // the center line node has no previous node
            current->centerLineNode->isStartNode = true;
        }

        // do nothing if the current segment was merged with another center line segment
        if( !clMerged ) {
		    // 3)
		    Section *next = new Section();
            next->prevCenterLineNode = current->centerLineNode;
            next->centerLineNode = nullptr;
            // TODO store nodes in next for future use in next cycle???
            NextSection( current->edges, currentNodes, next);

		    // 4)
            FindBranch( next, branches);
        
            next->centerLineNode = nullptr;
            next->prevCenterLineNode = nullptr;
            delete next;
        }
	}

}
