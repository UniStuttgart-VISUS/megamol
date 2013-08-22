#include "stdafx.h"
#include "CenterLineGenerator.h"

CenterLineGenerator::CenterLineGenerator(void) {
}


CenterLineGenerator::~CenterLineGenerator(void) {
}

void CenterLineGenerator::SetTriangleMesh( unsigned int count, float* mesh) {
    // clear previous data
    edges.clear();
	nodes.clear();

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
        e1->setNodes( *(itv2.first), *(itv3.first));
        e1->setNodes( *(itv1.first), *(itv3.first));
        // (3) insert all edges into the edge list
        // TODO add comparison function for edges (based on sorted nodes)
        ite1 = edges.insert( e1);
        ite2 = edges.insert( e2);
        ite3 = edges.insert( e3);
        // (4) add all edges to all vertices
        (*(itv1.first))->edges.insert( *(ite1.first));
        (*(itv1.first))->edges.insert( *(ite2.first));
        (*(itv1.first))->edges.insert( *(ite3.first));
        (*(itv2.first))->edges.insert( *(ite1.first));
        (*(itv2.first))->edges.insert( *(ite2.first));
        (*(itv2.first))->edges.insert( *(ite3.first));
        (*(itv3.first))->edges.insert( *(ite1.first));
        (*(itv3.first))->edges.insert( *(ite2.first));
        (*(itv3.first))->edges.insert( *(ite3.first));
        // delete vertex pointer if the vertex was not newly inserted
        if( !itv1.second ) delete v1;
        if( !itv2.second ) delete v2;
        if( !itv3.second ) delete v3;
        // delete edge pointer if the edge was not newly inserted
        if( !ite1.second ) delete e1;
        if( !ite2.second ) delete e2;
        if( !ite3.second ) delete e3;
    }
}

void CenterLineGenerator::Add(Edge &edge) {
	edges.insert(&edge);
	//nodes.insert(edge.nodes.begin(), edge.nodes.end());
    nodes.insert( edge.getNode1());
    nodes.insert( edge.getNode2());
}

void CenterLineGenerator::Add(Edges::iterator start, Edges::iterator end) {
	edges.insert(start, end);
	for(Edges::iterator it = start; it != end; ++it) {
		//nodes.insert((*it)->nodes.begin(), (*it)->nodes.end());
        nodes.insert( (*it)->getNode1());
        nodes.insert( (*it)->getNode2());
    }
    // TODO: the nodes need to know to which edges they are connected!
}

void CenterLineGenerator::NodesFromEdges(Edges &edges, Nodes &nodes) {
	for(auto edge : edges) {
		//nodes.insert( edge->nodes.begin(), edge->nodes.end() );
        nodes.insert( edge->getNode1());
        nodes.insert( edge->getNode2());
	}
}

CenterLineGenerator::CenterLineNode CenterLineGenerator::Collapse(Nodes &selection)
{
	CenterLineGenerator::Vector v;

	for(auto it : selection)
		v += it->p;

    // TODO Visual Studio compiler says: warning C4056: overflow in floating-point constant arithmetic
	float minimumDistance = 1e100;
	for(auto it : selection)
		minimumDistance = vislib::math::Min(minimumDistance, (it->p - v).Length());

	CenterLineNode node(v / selection.size(), minimumDistance);
	return node;
}

void CenterLineGenerator::NextSection(Edges &current, Section &next, Nodes &visited)
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
	Nodes currentNodes;
	this->NodesFromEdges(current, currentNodes);

	// 2)
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

	// 3), 4)
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
}

CenterLineGenerator::Edge *CenterLineGenerator::findEdgeNeighborInSet(Edge *edge, Edges &set, bool removeFromSet) {
    // TODO this seems to be wrong - why just search for first node in the edges of 'set'?
    // TODO should this return just the first neighbor?
    // TODO what happens if there is no neighbor?
	Edge *retEdge = nullptr;
	//Node *node = *edge->nodes.begin();
	//for(auto it = set.begin(); it != set.end(); ++it) {
	//	retEdge = *it;
	//	if( retEdge->nodes.find( node ) != retEdge->nodes.end() ) {
	//		if( removeFromSet ) {
	//			set.erase( it );
	//		}
	//		break;
	//	}
	//}

	return retEdge;
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

void CenterLineGenerator::FindBranch(Edges &current, std::vector<Edges*> &branches)
{
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

	while( !current.empty() ) {
		Edges *S0 = new Edges();
		Edge *edge = nullptr;
		Edge *edgeNext = nullptr;
		Edge *edgeNextNext = nullptr;

		if( !continueOnStep6 ) {
			// 2)
			for(auto it = current.begin(); it != current.end(); ++it) {
				edge = *it;
				if( edge->visited )
					continue;

				// 3)
				S0->insert( edge );
				it = current.erase( it );
				break;
			}

			assert(edge);

			// 4), 5)
			edgeNext = findEdgeNeighborInSet(edge, current, true);
	
		}
		// 6)
		edgeNextNext = findEdgeNeighborInSet(edgeNext, current, true);

		S0->insert( edgeNextNext );
		current.erase( edgeNextNext );

		// 7), 8)
		if( this->nodeSharedByEdges(edge, edgeNextNext) != nullptr ) {
			edgeNext = edgeNextNext;
			continueOnStep6 = true;
		} else {
			continueOnStep6 = false;
			branches.push_back(S0);
			S0 = nullptr;
		}
	}
}

void CenterLineGenerator::CenterLine(Edges &selection, Edges &centerLineEdges, Nodes &centerLineNodes)
{
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

	std::vector<Edges*> branches;

	Edges initial( selection );
	branches.push_back( &initial );

	Edges visited( initial );
	Nodes visitedNodes;
	this->NodesFromEdges( initial, visitedNodes );

	bool firstRun = true;
	CenterLineNode centerLineNode;
	
	while( !branches.empty() ) {
		Edges *current = branches.front();
		branches.erase( branches.begin() );

		// 2)
		Nodes currentNodes;
		this->NodesFromEdges( *current, currentNodes );
		CenterLineNode newCenterLineNode = Collapse( currentNodes );

		if( firstRun ) {
			firstRun = false;
		} else {
		}
		
		centerLineNode = newCenterLineNode;

		// 3)
		Section next;
		next.centerLineNode = newCenterLineNode;
		NextSection( *current, next, visitedNodes);

		// 4)
		FindBranch( *current, branches );
	}

}

void CenterLineGenerator::CenterLine_Panagiotidis(Edges &selection, Edges &centerLineEdges, Nodes &centerLineNodes)
{
	/*

	selection:
		mark edges and nodes visited

	for each node in selection:
		mark all edge visited


	for each edge in selection
		node1,node2 = edge

		find unmarked edges for node1 and node2


	*/
}