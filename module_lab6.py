# imports
import numpy as np
import math


def interpolate_linear(ti, yi, tj, default=None):
    """
    Perform linear interpolation of sampled data to a desired set of measurement points.

    Parameters:
        ti (1D array): Measurement points of the sampled data.
        yi (1D array): Measurement values of the sampled data.
        tj (1D array): Measurement points of the desired linearly interpolated data.
        default (None or other): Default value for the measurement value when the
                            corresponding measurement point is outside the sampled data.
                            Default is None.

    Returns:
        yj (1D array): Measurement values for the linearly interpolated data.
    """
    # Initialize yj with the same shape as tj
    yj = np.empty_like(tj)

    for j in range(len(tj)):
        if tj[j] < ti[0] or tj[j] > ti[-1]:
            yj[j] = default
        else:
            # Iterate through ti until you find the right interval
            for i in range(len(ti) - 1):
                if ti[i] <= tj[j] < ti[i + 1]:
                    # Perform the linear interpolation
                    yj[j] = yi[i] + (yi[i + 1] - yi[i]) * (tj[j] - ti[i]) / (ti[i + 1] - ti[i])
                    break

    return yj


def integrate_composite_trapezoid(tj, yj):
    """
    Perform numerical integration of the provided function using the composite trapezoidal rule.

    Parameters:
        tj (1D array): Measurement points of the integrand.
                       Assumes a closed interval i.e. first and last points correspond to integral limits.
        yj (1D array): Measurement values of the integrand.

    Returns:
        integral (float): Numerical approximation of integral.
    """
    # Ensure that tj and yj are numpy arrays
    tj = np.array(tj)
    yj = np.array(yj)

    # Calculate the differences between consecutive tj values
    dt = np.diff(tj)

    # Calculate the average of consecutive yj values
    y_avg = 0.5 * (yj[:-1] + yj[1:])

    # Calculate the integral as the sum of dt*y_avg
    integral = np.sum(dt * y_avg)

    return integral


def spath_initialise(network, source_name):
    """
    Initialise a graph for Dijkstra's shortest path algorithm.

    Parameters:
        network (Network object): The graph where each node has 'name' and 'value' attributes.
        source_name (str): The name of the source node.

    Returns:
        unvisited (set): A set of unvisited nodes.
    """
    unvisited = set()
    for node in network.nodes:
        if node.name == source_name:
            node.value = [0, None]  # Distance from source to source is 0, predecessor is None
        else:
            node.value = [math.inf, None]  # Set initial distance as infinity for all other nodes
        unvisited.add(node.name)
    return unvisited


def spath_iteration(network, unvisited):
    """
    Perform an iteration of Dijkstra's shortest path algorithm.

    Parameters:
        network (Network object): The graph where each node has 'name' and 'value' attributes.
        unvisited (set): A set of names of unvisited nodes.

    Returns:
        solved_name (str or None): The name of the solved node, or None if no node could be solved.
    """
    unvisited_nodes = [(node.value[0], node.name) for node in network.nodes if node.name in unvisited]
    if not unvisited_nodes:
        return None  # If there are no unvisited nodes, return None

    # Select the node with the smallest distance value
    _, solved_name = min(unvisited_nodes, key=lambda x: x[0])
    unvisited.remove(solved_name)

    # Update distances for each neighbor of the solved node
    for arc in network.get_node(solved_name).arcs_out:
        alt_distance = network.get_node(solved_name).value[0] + arc.weight
        if alt_distance < arc.to_node.value[0]:
            arc.to_node.value = [alt_distance, solved_name]

    return solved_name


def spath_extract_path(network, destination_name):
    """
    Extract the shortest path by backtracking from the destination node.

    Parameters:
        network (Network object): The graph where each node has 'name' and 'value' attributes.
        destination_name (str): The name of the destination node.

    Returns:
        path (list): The shortest path as a list of node names.
    """
    path = []
    current_name = destination_name
    while current_name is not None:
        path.append(current_name)
        current_name = network.get_node(current_name).value[1]
    return path[::-1]  # Return reversed path


def spath_algorithm(network, source_name, destination_name):
    """
       Run Dijkstra's shortest path algorithm on a graph.

       Parameters:
           network (Network object): The graph where each node has 'name' and 'value' attributes.
           source_name (str): The name of the source node.
           destination_name (str): The name of the destination node.

       Returns:
           length and path (tuple): The length of the shortest path and the path itself as a list of node names.
                                    Returns (None, None) if no path can be found.
       """
    unvisited = spath_initialise(network, source_name)
    while unvisited:
        solved_name = spath_iteration(network, unvisited)
        if solved_name == destination_name:  # If we've solved the destination node, we can stop
            break
        if solved_name is None:  # If no node could be solved (e.g., unvisited set is empty), there's no viable path
            return None, None

    # If the destination node value is still infinity, there's no path
    if network.get_node(destination_name).value[0] == math.inf:
        return None, None

    path = spath_extract_path(network, destination_name)
    return network.get_node(destination_name).value[0], path


class Node(object):
    """
    Object representing network node.

    Attributes:
    -----------
    name : str, int
        unique identifier for the node.
    value : float, int, bool, str, list, etc...
        information associated with the node.
    arcs_in : list
        Arc objects that end at this node.
    arcs_out : list
        Arc objects that begin at this node.
    """
    def __init__(self, name=None, value=None, arcs_in=None, arcs_out=None):

        self.name = name
        self.value = value
        if arcs_in is None:
            self.arcs_in = []
        if arcs_out is None:
            self.arcs_out = []

    def __repr__(self):
        return f"node:{self.name}"


class Arc(object):
    """
    Object representing network arc.

    Attributes:
    -----------
    weight : int, float
        information associated with the arc.
    to_node : Node
        Node object (defined above) at which arc ends.
    from_node : Node
        Node object at which arc begins.
    """
    def __init__(self, weight=None, from_node=None, to_node=None):
        self.weight = weight
        self.from_node = from_node
        self.to_node = to_node

    def __repr__(self):
        return f"arc:({self.from_node.name})--{self.weight}-->({self.to_node.name})"


class Network(object):
    """
    Basic Implementation of a network of nodes and arcs.

    Attributes
    ----------
    nodes : list
        A list of all Node (defined above) objects in the network.
    arcs : list
        A list of all Arc (defined above) objects in the network.
    """
    def __init__(self, nodes=None, arcs=None):
        if nodes is None:
            self.nodes = []
        if arcs is None:
            self.arcs = []

    def __repr__(self):
        node_names = '\n'.join(node.__repr__() for node in self.nodes)
        arc_info = '\n'.join(arc.__repr__() for arc in self.arcs)
        return f'{node_names}\n{arc_info}'

    def get_node(self, name):
        """
        Return network node with name.

        Parameters:
        -----------
        name : str
            Name of node to return.

        Returns:
        --------
        node : Node, or None
            Node object (as defined above) with corresponding name, or None if not found.
        """
        # loop through list of nodes until node found
        for node in self.nodes:
            if node.name == name:
                return node

        # if node not found, return None
        return None

    def add_node(self, name, value=None):
        """
        Adds a node to the Network.

        Parameters
        ----------
        name : str
            Name of the node to be added.
        value : float, int, str, etc...
            Optional value to set for node.
        """
        # create node and add it to the network
        new_node = Node(name, value)
        self.nodes.append(new_node)

    def add_arc(self, node_from, node_to, weight):
        """
        Adds an arc between two nodes with a desired weight to the Network.

        Parameters
        ----------
        node_from : Node
            Node from which the arc departs.
        node_to : Node
            Node to which the arc arrives.
        weight : float
            Desired arc weight.
        """
        # create the arc and add it to the network
        new_arc = Arc(weight, node_from, node_to)
        self.arcs.append(new_arc)

        # update the connected nodes to include arc information
        node_from.arcs_out.append(new_arc)
        node_to.arcs_in.append(new_arc)

    def read_network(self, filename):
        """
        Reads a file to construct a network of nodes and arcs.

        Parameters
        ----------
        filename : str
            The name of the file (inclusive of extension) from which to read the network data.
        """
        with open(filename, 'r') as file:

            # get first line in file
            line = file.readline()

            # check for end of file, terminate if found
            while line != '':
                items = line.strip().split(',')

                # create source node if it doesn't already exist
                if self.get_node(items[0]) is None:
                    self.add_node(items[0])

                # get starting node for this line
                source_node = self.get_node(items[0])

                for item in items:

                    # initial item ignored as it has no arc
                    if item == source_node.name:
                        continue

                    # separate out to destination node name and arc weight
                    data = item.split(';')
                    destination_node = data[0]
                    arc_weight = data[1]

                    # Create destination node if not already in network, then obtain the node itself
                    if self.get_node(destination_node) is None:
                        self.add_node(destination_node)
                    destination_node = self.get_node(destination_node)

                    # Add arc from source to destination node, with associated weight
                    self.add_arc(source_node, destination_node, float(arc_weight))

                # get next line in file
                line = file.readline()

