#pragma once

#include <vislib\Point.h>
#include <set>
#include <vector>

class CenterLineGenerator
{
public:

	typedef vislib::math::Vector<float, 3> Vector;

	struct Node;
	struct Edge;

	typedef std::set<Node*> Nodes;
	typedef std::set<Edge*> Edges;

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
				nodes.insert(it->nodes.begin(), it->nodes.end());
			}
		}
	};

	struct Edge
	{
		Edge()
			: nodes()
			, visited(false)
		{}

		Nodes nodes;
		bool visited;

		void getEdges(Nodes edges) {
			for(auto it : nodes) {
				edges.insert(it->edges.begin(), it->edges.end());
			}
		}
	};

	CenterLineGenerator(void);
	virtual ~CenterLineGenerator(void);

	void Add(Edge &edge);
	void Add(Edges::iterator start, Edges::iterator end);

	void CenterLine(Edges &selection, Edges &centerLineEdges, Nodes &centerLineNodes);


	void CenterLine_Panagiotidis(Edges &selection, Edges &centerLineEdges, Nodes &centerLineNodes);

private:

	Vector Collapse(Nodes &selection);

	Edge *findEdgeNeighborInSet(Edge *edge, Edges &set, bool removeFromSet);

	Node *nodeSharedByEdges(Edge *e1, Edge *e2);

	void NextSection(Edges &current, Edges &next, Nodes &visited);
	void FindBranch(Edges &current, std::vector<Edges*> &branches);

	void NodesFromEdges(Edges &edges, Nodes &nodes);

	Edges edges;
	Nodes nodes;
};

