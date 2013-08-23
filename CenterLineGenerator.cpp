#include "stdafx.h"
#include "CenterLineGenerator.h"
#include <float.h>
#include "vislib/Array.h"

CenterLineGenerator::CenterLineGenerator(void) {
}


CenterLineGenerator::~CenterLineGenerator(void) {
}

void CenterLineGenerator::SetTriangleMesh( unsigned int count, float* mesh) {
    // clear previous data
    edges.clear();
	nodes.clear();
    // Array for free edges (allocate space for three edges per triangle)
    vislib::Array<Edge*> freeEdges( count * 3);

    // loop over all triangles
    for( unsigned int tCnt = 0; tCnt < count; tCnt++) {
        Node *v1 = new Node();
        Node *v2 = new Node();
        Node *v3 = new Node();
        Edge *e1 = new Edge();
        Edge *e2 = new Edge();
        Edge *e3 = new Edge();
        std::pair<Nodes::iterator, bool> itv1, itv2, itv3;
        std::pair<Edges::iterator, bool> ite1, ite2, ite3;
        
        // get all the vertices of the current triangle from the mesh
        v1->p.Set( mesh[12 * tCnt + 0], mesh[12 * tCnt + 1], mesh[12*tCnt + 2]);
        v2->p.Set( mesh[12 * tCnt + 4], mesh[12 * tCnt + 5], mesh[12*tCnt + 6]);
        v3->p.Set( mesh[12 * tCnt + 8], mesh[12 * tCnt + 9], mesh[12*tCnt +10]);
        // (1) insert all vertices into the nodes list
        // TODO add comparison function for nodes (based on p value)
        itv1 = nodes.insert( v1);
        itv2 = nodes.insert( v2);
        itv3 = nodes.insert( v3);
        // (2) construct edges (v1, v2) (v2, v3) (v1, v3)
        e1->setNodes( *(itv1.first), *(itv2.first));
        e2->setNodes( *(itv2.first), *(itv3.first));
        e3->setNodes( *(itv1.first), *(itv3.first));
        // (3) insert all edges into the edge list
        // TODO add comparison function for edges (based on sorted nodes)
        ite1 = edges.insert( e1);
        ite2 = edges.insert( e2);
        ite3 = edges.insert( e3);
        // (4) add the two correct edges to all nodes
        (*(itv1.first))->edges.insert( *(ite1.first));
        (*(itv1.first))->edges.insert( *(ite3.first));
        (*(itv2.first))->edges.insert( *(ite1.first));
        (*(itv2.first))->edges.insert( *(ite2.first));
        (*(itv3.first))->edges.insert( *(ite2.first));
        (*(itv3.first))->edges.insert( *(ite3.first));
        // delete vertex pointer if the vertex was not newly inserted
        if( !itv1.second )
            delete v1;
        if( !itv2.second )
            delete v2;
        if( !itv3.second )
            delete v3;
        // delete edge pointer if the edge was not newly inserted
        // remove the edge pointer from the freeEdgeList if 'false', otherwise add it
        if( !ite1.second ) {
            freeEdges.Remove( *(ite1.first));
            delete e1;
        } else {
            freeEdges.Add( e1);
        }
        if( !ite2.second ) {
            freeEdges.Remove( *(ite2.first));
            delete e2;
        } else {
            freeEdges.Add( e2);
        }
        if( !ite3.second ) {
            freeEdges.Remove( *(ite3.first));
            delete e3;
        } else {
            freeEdges.Add( e3);
        }
    }
    
    // TODO make faster!!
    this->freeEdgeRing.clear();
    if( freeEdges.Count() > 0 ) {
        Edges freeEdgeSet;
        // insert all free edges to the set
        for( unsigned int i = 1; i < freeEdges.Count(); i++ ) {
            freeEdgeSet.insert( freeEdges[i]);
        }
        Edge *e = freeEdges[0];
        while( e != nullptr ) {
            this->freeEdgeRing.insert( e);
            e = findEdgeNeighborInSet( e, freeEdgeSet, true);
        }
    }
}

void CenterLineGenerator::Add(Edge &edge) {
	edges.insert(&edge);
    nodes.insert( edge.getNode1());
    nodes.insert( edge.getNode2());
}

void CenterLineGenerator::Add(Edges::iterator start, Edges::iterator end) {
	edges.insert(start, end);
	for(Edges::iterator it = start; it != end; ++it) {
        nodes.insert( (*it)->getNode1());
        nodes.insert( (*it)->getNode2());
    }
    // TODO: the nodes need to know to which edges they are connected!
}

void CenterLineGenerator::NodesFromEdges(Edges &edges, Nodes &nodes) {
	for(auto edge : edges) {
        nodes.insert( edge->getNode1());
        nodes.insert( edge->getNode2());
	}
}

CenterLineGenerator::CenterLineNode CenterLineGenerator::Collapse(Nodes &selection) {
	CenterLineGenerator::Vector v( 0.0f, 0.0f, 0.0f);

	for(auto it : selection) {
		v += it->p;
        it->visited = true;
    }
    v /= static_cast<float>(selection.size());

    float minimumDistance =  FLT_MAX;
	for(auto it : selection)
		minimumDistance = vislib::math::Min(minimumDistance, (it->p - v).Length());

	CenterLineNode node(v, minimumDistance);
	return node;
}

void CenterLineGenerator::NextSection(Edges &current, Nodes &currentNodes, Section *next)
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
    Edges E1;
    Nodes N1; // non-visited neigbor nodes of S0
    Edges A2;
    // get all connected edges for all current nodes (one shared node)
    for( auto node : currentNodes ) {
        for( auto edge : node->edges ) {
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
        for( auto edge : node->edges ) {
            // both nodes have to be in N1
            if( edge->getNode1() == node && N1.find( edge->getNode2()) != N1.end() ) {
                next->edges.insert( edge);
            } else if( edge->getNode2() == node && N1.find( edge->getNode1()) != N1.end() ) {
                next->edges.insert( edge);
            }
        }
    }


    /*
	Edges E1;
	for(auto edge : edges) {
		//for(auto node : edge->nodes) {
		//	if( visited.find( node ) == visited.end() )	{
		//		bool found = false;
		//		for(auto curNode : currentNodes) {
		//			if( found = (node == curNode)) {
		//				E1.insert( edge );
		//				break;
		//			}
		//		}
		//	}
		//}
        auto node = edge->getNode1();
		if( visited.find( node ) == visited.end() )	{
			bool found = false;
			for(auto curNode : currentNodes) {
				if( found = (node == curNode)) {
					E1.insert( edge );
					break;
				}
			}
		}
        node = edge->getNode2();
		if( visited.find( node ) == visited.end() )	{
			bool found = false;
			for(auto curNode : currentNodes) {
				if( found = (node == curNode)) {
					E1.insert( edge );
					break;
				}
			}
		}
	}
    */

	// 3), 4)
    /*
	for(auto edge : E1) {
		//for(auto node : edge->nodes) {
		//	bool found = false;
		//	for(auto curNode : currentNodes) {
		//		found = (node == curNode);
		//	}

		//	if( !found ) {
		//		next.edges.insert( edge );

		//		// 5)
		//		visited.insert(edge->nodes.begin(), edge->nodes.end());
		//	}
		//}
        
        auto node = edge->getNode1();
        bool found = false;
		for(auto curNode : currentNodes) {
			found = (node == curNode);
		}
		if( !found ) {
			next.edges.insert( edge );
			// 5)
			//visited.insert(edge->nodes.begin(), edge->nodes.end());
            visited.insert(edge->getNode1());
            visited.insert(edge->getNode2());
		}
        node = edge->getNode2();
        found = false;
		for(auto curNode : currentNodes) {
			found = (node == curNode);
		}
		if( !found ) {
			next.edges.insert( edge );
			// 5)
			//visited.insert(edge->nodes.begin(), edge->nodes.end());
            visited.insert(edge->getNode1());
            visited.insert(edge->getNode2());
		}
	}
    */
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
				set.erase( it );
			}
			return retEdge;
		}
	}
    // no match found - return null pointer
    return nullptr;
}

CenterLineGenerator::Node *CenterLineGenerator::nodeSharedByEdges(Edge *e1, Edge *e2) {
	//for(auto n1 : e1->nodes) {
	//	for(auto n2 : e2->nodes) {
	//		if( n1 == n2 )
	//			return n1;
	//	}
	//}
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

void CenterLineGenerator::FindBranch(Section *current, std::vector<Section*> &branches) {
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
        auto eit = current->edges.begin();
        Edge *e = *eit;
        current->edges.erase( eit);
        /*
        while( e != nullptr ) {
            S0->edges.insert( e);
            e = findEdgeNeighborInSet( e, current->edges, true);
        }
        */
        // DEBUG
        edgeVec.clear();
        edgeVec.push_back( e);
        while( !edgeVec.empty() ) {
            while( edgeVec.back() != nullptr ) {
                edgeVec.push_back( findEdgeNeighborInSet( edgeVec.front(), current->edges, true));
            }
            edgeVec.pop_back();
            S0->edges.insert( edgeVec.front());
            edgeVec.pop_front();
        }
        // ring found -> add to branch list
        branches.push_back( S0);
    }


    /*
    while( !current->edges.empty() ) {

		if( !continueOnStep6 ) {
			// 2)
            for(auto it = current->edges.begin(); it != current->edges.end(); ++it) {
				edge = *it;
				if( edge->visited )
					continue;

				// 3)
				S0->edges.insert( edge);
                it = current->edges.erase( it );
				break;
			}

			assert(edge);

			// 4), 5)
            edgeNext = findEdgeNeighborInSet(edge, current->edges, true);
            if( edgeNext == nullptr ) {
			    branches.push_back(S0);
                if( !current->edges.empty() ) {
			        continueOnStep6 = false;
                    S0 = new Section();
                    S0->prevCenterLineNode = current->centerLineNode;
                }
                continue;
            }
		    S0->edges.insert( edgeNext );
		}
		// 6)
        edgeNextNext = findEdgeNeighborInSet(edgeNext, current->edges, true);
        if( edgeNextNext == nullptr ) {
			branches.push_back(S0);
            if( !current->edges.empty() ) {
			    continueOnStep6 = false;
                S0 = new Section();
                S0->prevCenterLineNode = current->centerLineNode;
            }
            continue;
        }
		S0->edges.insert( edgeNextNext );

		// 7), 8)
		if( this->nodeSharedByEdges(edge, edgeNextNext) == nullptr ) {
			edgeNext = edgeNextNext;
			continueOnStep6 = true;
		} else {
			branches.push_back(S0);
            if( !current->edges.empty() ) {
			    continueOnStep6 = false;
                S0 = new Section();
                S0->prevCenterLineNode = current->centerLineNode;
            }
		}
	}
    */
}

void CenterLineGenerator::CenterLine(Edges &selection, CenterLineEdges &centerLineEdges, CenterLineNodes &centerLineNodes) {
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

    std::vector<Section*> branches;
    
    branches.push_back( new Section());
    branches[0]->edges.insert( selection.begin(), selection.end());
    branches[0]->centerLineNode = nullptr;
    branches[0]->prevCenterLineNode = nullptr;
    	
	while( !branches.empty() ) {
		Section *current = branches.front();
		branches.erase( branches.begin() );
        
        // DEBUG
        allBranches.push_back(current);

		// 2)
		Nodes currentNodes;
        this->NodesFromEdges( current->edges, currentNodes);
        // set all current edges as visited
        for( auto edge : current->edges) {
            edge->visited = true;
        }

        current->centerLineNode = new CenterLineNode();
		*current->centerLineNode = Collapse( currentNodes);
        centerLineNodes.push_back( current->centerLineNode);
        
        // construct center line edge
        if( current->prevCenterLineNode != nullptr ) {
            CenterLineEdge *cle = new CenterLineEdge();
            cle->node1 = current->prevCenterLineNode;
            cle->node2 = current->centerLineNode;
            centerLineEdges.push_back( cle);
        }

		// 3)
		Section *next = new Section();
        next->prevCenterLineNode = current->centerLineNode;
        next->centerLineNode = nullptr;
        // TODO store nodes in next for future use in next cycle???
        NextSection( current->edges, currentNodes, next);

		// 4)
#if 1
        FindBranch( next, branches);
        
        next->centerLineNode = nullptr;
        next->prevCenterLineNode = nullptr;
        delete next;
#else
        // DEBUG
        if( !next->edges.empty() )
            branches.push_back( next);
#endif
	}

}
