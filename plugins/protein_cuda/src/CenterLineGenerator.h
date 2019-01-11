#pragma once

#include "vislib/math/Point.h"
#include <unordered_set>
#include <vector>
#include <functional>   // std::equal_to
#include <algorithm>    // std::mismatch
#include <list>

class CenterLineGenerator
{
public:

    /** The names of the coloring modes */
    enum FeatureType {
        CAVITY  = 0,
        CHANNEL = 1,
        POCKET  = 2
    };

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

    typedef std::unordered_set<Node*, hash_node, equal_to_nodeptr> NodeSet;
	typedef std::unordered_set<Edge*, hash_edge, equal_to_edgeptr> EdgeSet;

    typedef std::list<Node*> Nodes;
	typedef std::list<Edge*> Edges;

	struct Node
	{
		Node()
			: p()
			//, edges()
            , edgeCnt(0)
			, visited(false)
		{}

		Vector p;
		//Edges edges;
        // TODO is 16 really a good value for marching tetrahedra meshes?
        Edge *edges[16];
        unsigned int edgeCnt;
		bool visited;
	};

	class Edge {
    public:
        Edge() : node1(0), node2(0), visited(false) {}
		//	: nodes()
		//	, visited(false)
		//{}

        ~Edge() {}

		bool visited;

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
	typedef std::list<CenterLineEdge*> CenterLineEdges;
	typedef std::list<CenterLineNode*> CenterLineNodes;

	struct CenterLineNode
	{
		CenterLineNode()
			: p()
			, minimumDistance(0.0f)
            , isRing(true)
            , isStartNode(false)
		{}

		CenterLineNode(Vector position, float minDistance)
			: p(position)
			, minimumDistance(minDistance)
            , isRing(true)
            , isStartNode(false)
		{}

		Vector p;
		float minimumDistance;
        bool isRing;
        bool isStartNode;

		CenterLineEdges edges;
	};

	struct CenterLineEdge
	{
		CenterLineEdge()
            : node1(nullptr)
            , node2(nullptr)
            , visited(false)
		{}
        
		CenterLineNode *node1;
		CenterLineNode *node2;
        bool visited;
	};

	
	struct Section {
		Edges edges;
        bool isRing;
		CenterLineNode *centerLineNode;
		CenterLineNode *prevCenterLineNode;
	};


	CenterLineGenerator(void);
	virtual ~CenterLineGenerator(void);

    /**
     * TODO
     *
     * @param TODO
     */
    void SetTriangleMesh( unsigned int vertexCnt, float* vertices, unsigned int edgeCnt, unsigned int *edges, unsigned int *freeEdge);

	void CenterLine(Edges &selection, CenterLineEdges &centerLineEdges, CenterLineNodes &centerLineNodes, float minCenterLineNodeDist = 0.0f);
    
    std::vector<Edges> freeEdgeRing;
    FeatureType fType;

    // TODO make this private again!!!
	Edges edges;
	Nodes nodes;

    std::list<Section*> allBranches;
    
    Node *theNodes;

    Edge *theEdges;

private:

	CenterLineNode* Collapse(NodeSet &selection);

	Edge *findEdgeNeighborInSet(Edge *edge, Edges &set, bool removeFromSet);

	Node *nodeSharedByEdges(Edge *e1, Edge *e2);

	void NextSection(Edges &current, NodeSet &currentNodes, Section *next);
    void FindBranch(Section *current, std::list<Section*> &branches);

	void NodesFromEdges(Edges &edges, NodeSet &nodes);

};

