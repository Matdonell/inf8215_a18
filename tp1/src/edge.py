class Edge:
    def __init__(self, from_v, to_v, weight):
        """
        Represents an Edge in the graph
        from_v : source of the edge
        to_v: end of the edge
        weight: related weight of the edge
        """
        self.from_v = from_v
        self.to_v = to_v
        self.weight = weight

    def __lt__(self, other):
        return self.weight < other.weight
