#pragma once

#include "vislib/Point.h"
#include <unordered_set>
#include <vector>

class CenterLineGenerator
{
public:

	typedef vislib::math::Vector<float, 3> Vector;

	struct Node;
	class Edge;

    typedef std::unordered_set<Node*> Nodes;
	typedef std::unordered_set<Edge*> Edges;

	struct Node
	{
		Node()
			: p()
			, edges()
			, visited(false)
		{}

		Vector p;
		Edges edges;
		bool visited;

		void getNodes(Nodes nodes) {
			for(auto it : edges) {
				//nodes.insert(it->nodes.begin(), it->nodes.end());
                nodes.insert( it->getNode1());
                nodes.insert( it->getNode2());
			}
		}
	};

	class Edge {
    public:
        Edge() : node1(0), node2(0), visited(false) {}
		//	: nodes()
		//	, visited(false)
		//{}

		bool visited;

		void getEdges(Edges edges) {
			//for(auto it : nodes) {
			//	edges.insert(it->edges.begin(), it->edges.end());
			//}
            edges.insert(node1->edges.begin(), node1->edges.end());
            edges.insert(node2->edges.begin(), node2->edges.end());
		}
        
        void setNodes( Node *n1, Node *n2) {
            // TODO check for null pointers
            // sort nodes by p
            if( n1->p.X() > n2->p.X() ) {
                node1 = n2;
                node2 = n1;
            } else if( n2->p.X() > n1->p.X() ) {
                node1 = n1;
                node2 = n2;
            } else {
                if( n1->p.Y() > n2->p.Y() ) {
                    node1 = n2;
                    node2 = n1;
                } else if( n2->p.Y() > n1->p.Y() ) {
                    node1 = n1;
                    node2 = n2;
                } else {
                    if( n1->p.Z() > n2->p.Z() ) {
                        node1 = n2;
                        node2 = n1;
                    } else {
                        node1 = n1;
                        node2 = n2;
                    }
                }
            }
        }

        inline Node* getNode1() { return node1; }
        inline Node* getNode2() { return node2; }
    protected:
        // TODO why use a node set here? an edge has alwas exactly two nodes!
		//Nodes nodes;
        Node *node1;
        Node *node2;
	};
	
	struct CenterLineNode;
	struct CenterLineEdge;
	typedef std::unordered_set<CenterLineEdge*> CenterLineEdges;
	typedef std::unordered_set<CenterLineNode*> CenterLineNodes;

	struct CenterLineNode
	{
		CenterLineNode()
			: p()
			, minimumDistance(0.0f)
		{}

		CenterLineNode(Vector position, float minDistance)
			: p(position)
			, minimumDistance(minDistance)
		{}

		Vector p;
		float minimumDistance;

		CenterLineEdges edges;
	};

	struct CenterLineEdge
	{
		CenterLineEdge()
		{}

		CenterLineNodes nodes;
	};

	
	struct Section
	{
		Edges edges;
		CenterLineNode centerLineNode;
	};


	CenterLineGenerator(void);
	virtual ~CenterLineGenerator(void);

    /**
     * Use the given triangle mesh for center line extraction.
     * The tringles have to be in the format [(xyzw)(xyzw)(xyzw)][...]..., 
     * however, the w coordinate is ommitted.
     * The function clears all previously set data.
     *
     * @param count The number of triangles.
     * @param mesh The set of unordered triangles.
     */
    void SetTriangleMesh( unsigned int count, float* mesh);

	void Add(Edge &edge);
	void Add(Edges::iterator start, Edges::iterator end);

	void CenterLine(Edges &selection, Edges &centerLineEdges, Nodes &centerLineNodes);

    /**
     * TODO
     */
	void CenterLine_Panagiotidis(Edges &selection, Edges &centerLineEdges, Nodes &centerLineNodes);

private:

	CenterLineNode Collapse(Nodes &selection);

	Edge *findEdgeNeighborInSet(Edge *edge, Edges &set, bool removeFromSet);

	Node *nodeSharedByEdges(Edge *e1, Edge *e2);

	void NextSection(Edges &current, Section &next, Nodes &visited);
	void FindBranch(Edges &current, std::vector<Edges*> &branches);

	void NodesFromEdges(Edges &edges, Nodes &nodes);

	Edges edges;
	Nodes nodes;

	//std::vector<Edges*> &openLoops;
};

