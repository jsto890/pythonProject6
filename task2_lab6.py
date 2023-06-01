from module_lab6 import *

network = Network()
network.read_network('network_waka_voyage.txt')

# 2.1 Shortest path from Taiwan to Hokianga

# We call the spath_algorithm function and provide the network,
# the starting point 'Taiwan' and the destination 'Hokianga'
time, path = spath_algorithm(network, 'Taiwan', 'Hokianga')

# The function returns the shortest time and the path taken,
# which we print out to the console.
print("The shortest path from Taiwan to Hokianga takes", time, "time units. The path is:")
print(path)

# 2.2 Largest possible shortest path between any pair of islands

# We'll need to keep track of the longest shortest path that we've found so far.
# We'll start by assuming that it's zero.
longest_path_length = 0

# We'll also keep track of the start and end points of this path.
# We'll start by assuming there are none.
longest_path_start = None
longest_path_end = None

# Now we'll go through every pair of start and end points in the network.
for start_node in network.nodes:
    for end_node in network.nodes:
        # We don't want to calculate the path from a node to itself, so we skip those cases.
        if start_node != end_node:
            # Calculate the shortest path from start_node to end_node.
            time, path = spath_algorithm(network, start_node.name, end_node.name)

            # If this path is longer than our current longest path, we save it.
            if time is not None and time > longest_path_length:
                longest_path_length = time
                longest_path_start = start_node.name
                longest_path_end = end_node.name

# Finally, we print out the start and end points of the longest path,
# as well as the time it takes.
print("The longest shortest path is from", longest_path_start, "to", longest_path_end,
      "and takes", longest_path_length, "time units.")