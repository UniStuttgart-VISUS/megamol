#pragma once

#include "vislib/Point.h"
#include <unordered_set>
#include <vector>
#include <functional>   // std::equal_to
#include <algorithm>    // std::mismatch

class CenterLineGenerator
{
public:

	typedef vislib::math::Vector<float, 3> Vector;

	struct Node;
	class Edge;

    struct equal_to_nodeptr {
        bool operator() (const Node* const &x, const Node* const &y) const {
            return ( vislib::math::IsEqual( x->p.X(), y->p.X(), 2.0f * vislib::math::FLOAT_EPSILON) &&
                     vislib::math::IsEqual( x->p.Y(), y->p.Y(), 2.0f * vislib::math::FLOAT_EPSILON) &&
                     vislib::math::IsEqual( x->p.Z(), y->p.Z(), 2.0f * vislib::math::FLOAT_EPSILON));
        }
    };

    struct equal_to_edgeptr {
        bool operator() (const Edge* const &x, const Edge* const &y) const {
            return ((x->Node1() == y->Node1()) && (x->Node2() == y->Node2()));
        }
    };

    struct hash_node {
	    size_t operator()( const Node* const &val) const {
            return static_cast<size_t>(ceilf(val->p.X() + val->p.Y() + val->p.Z()));
		}
	};
    
    struct hash_edge {
	    size_t operator()( const Edge* const &val) const {
            Vector v = (val->Node1()->p - val->Node2()->p) / 2.0f + val->Node1()->p;
            return static_cast<size_t>( ceilf(v.Length()));
		}
	};

    typedef std::unordered_set<Node*, hash_node, equal_to_nodeptr> Nodes;
	typedef std::unordered_set<Edge*, hash_edge, equal_to_edgeptr> Edges;

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

        ~Edge() {}

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
        inline const Node* Node1() const { return node1; }
        inline const Node* Node2() const { return node2; }        
    protected:
		//Nodes nodes;
        Node *node1;
        Node *node2;
	};
	
	struct CenterLineNode;
	struct CenterLineEdge;
	typedef std::vector<CenterLineEdge*> CenterLineEdges;
	typedef std::vector<CenterLineNode*> CenterLineNodes;

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
        
		CenterLineNode *node1;
		CenterLineNode *node2;
	};

	
	struct Section {
		Edges edges;
		CenterLineNode *centerLineNode;
		CenterLineNode *prevCenterLineNode;
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

	void CenterLine(Edges &selection, CenterLineEdges &centerLineEdges, CenterLineNodes &centerLineNodes);
    
    // TODO fix this!!!
    Edges freeEdgeRing;

    // TODO make this private again!!!
	Edges edges;
	Nodes nodes;

    std::vector<Section*> allBranches;
private:

	CenterLineNode Collapse(Nodes &selection);

	Edge *findEdgeNeighborInSet(Edge *edge, Edges &set, bool removeFromSet);

	Node *nodeSharedByEdges(Edge *e1, Edge *e2);

	void NextSection(Edges &current, Nodes &currentNodes, Section *next);
    void FindBranch(Section *current, std::vector<Section*> &branches);

	void NodesFromEdges(Edges &edges, Nodes &nodes);

};

