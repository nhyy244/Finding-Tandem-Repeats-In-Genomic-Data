import copy
import time
import timeit

import numpy
from matplotlib import pyplot as plt


class Node():
    def __init__(self, lab, leafnum=None):
        self.lab = lab  #label (index tuple) on path leading to this node
        self.text = None #Text that the index represents
        self.out = {}  # outgoing edges; maps characters to nodes
        self.parent = None
        self.leafnum = leafnum #If the node is a leaf, this leafnum corresponds to the index of the suffix. If the node is not a leaf, this is None
        self.dfsnum = None
        self.slink = None

    def __str__(self):
        return self.lab

    def addChild(self, n):
        self.out.update({n.lab: n})
        n.parent = self
        return self

    def set_label(self, lab):
        if self.lab in self.parent.out:
            self.parent.out.pop(self.lab)
            self.lab = lab
            self.parent.out.update({self.lab: self})
        return self

#def suftree_naive(s):
#    #Naive implementation that stores strings on the edges, inefficient, but easier to understand. Deprecated now. Doesnt work
#    #Naive suffix tree construction algorithm
#    #The algorithm is as described in the "Algorithms on Strings, Trees and Sequences" by Dan Gusfield, chapter 5.4
#    #The algorithm starts with the suffix tree t1
#
#    s += "$"
#    root = Node("")
#    for i in range(0, len(s)):
#        suf = s[i:]
#        current = root
#        matching_prefix = node_offset = 0
#        while True:
#            next_node = None
#            node_offset = 0
#            for child in current.out:
#                j = 0
#                while j < len(suf) and j < len(child) and suf[matching_prefix + j] == child[j]:
#                    j += 1
#                    if j > node_offset:
#                        node_offset = j
#                        next_node = current.out.get(child)
#
#            matching_prefix += node_offset
#
#            if next_node is not None:
#                current = next_node
#                if node_offset < len(next_node.lab):
#                    break
#            else:
#                break
#
#        if node_offset == 0:
#            current.addChild(Node(suf[matching_prefix:], leafnum = i))
#        else:
#            current_children = current.out
#            new_node = Node(current.lab[node_offset:], current.leafnum)
#            new_node.out = current_children
#            for child in current_children:
#                current_children.get(child).parent = new_node
#            current.out = {}
#            current.addChild(new_node)
#            current.addChild(Node(suf[matching_prefix:], leafnum = i))
#            current.leafnum = None
#            current.set_label(current.lab[0:node_offset])
#
#    return root


















#The index representation follows the conventions of representing substrings by indexing, meaning that we index from 0 and exclude the last character, i.e "abc"[0:2] = "ab", ""abc"[1:1] = ""
def suftree_naive(s):
    s += "$"
    root = Node((0, 0))
    for i in range(0, len(s)):
        suf = s[i:]
        current = root
        matching_prefix = offset = 0
        #Scanning
        while True:
            next_node = None
            offset = 0
            for child in current.out:
                j = 0
                while (j < len(suf) and j < (child[1] - child[0])
                       and suf[matching_prefix + j] == s[child[0] + j]):
                    j += 1
                if j > offset:
                    offset = j
                    next_node = current.out.get(child)
                    break

            matching_prefix += offset

            if next_node is not None:
                current = next_node
                if offset < next_node.lab[1] - next_node.lab[0]:
                    break
            else:
                break

        #Splitting
        if offset == 0:
            current.addChild(Node((i + matching_prefix, len(s)), leafnum = i))
        else:
            parent = current.parent
            new_node = Node((current.lab[0], current.lab[0] + offset), 0)

            tail = Node((i + matching_prefix, len(s)), leafnum=i)
            new_node.addChild(tail)
            new_node.addChild(current)
            parent.out.pop(current.lab)
            current.set_label((current.lab[0] + offset, current.lab[1]))
            parent.addChild(new_node)

    return root


def mccreight(s):

    def split_or_add(node, offset, matching_prefix):
        if offset == 0:
            tail = Node((i + matching_prefix, len(s)), leafnum=i)
            node.addChild(tail)
        else:
            parent = node.parent
            new_node = Node((node.lab[0], node.lab[0] + offset), 0)
            tail = Node((i + matching_prefix, len(s)), leafnum=i)
            new_node.addChild(tail)
            new_node.addChild(node)
            parent.out.pop(node.lab)
            node.set_label((node.lab[0] + offset, node.lab[1]))
            parent.addChild(new_node)
        return tail

    def slowscan(node_from, string_to):
        string_to = s[string_to[0]:string_to[1]]
        current = node_from
        matching_prefix = node_offset = 0
        while True:
            next_node = None
            offset = 0
            for child in current.out:
                j = 0
                while j < len(string_to) and j < (child[1] - child[0]) and string_to[matching_prefix + j] == s[child[0] + j]:
                    j += 1
                if j > offset:
                    offset = j
                    next_node = current.out.get(child)
                    break

            matching_prefix += offset

            if next_node is not None:
                current = next_node
                if offset < next_node.lab[1] - next_node.lab[0]:
                    return current, offset, matching_prefix
            else:
                return current, offset, matching_prefix

    def fastscan(node_from, string_to):
        matching_prefix = string_to[1] - string_to[0] + node_from.lab[1] - node_from.lab[0]
        current = node_from
        while True:
            if string_to[1] - string_to[0] == 0:
                return current, 0, matching_prefix

            next_node = None
            offset = 0
            for child in current.out:
                if s[child[0]] == s[string_to[0]]:
                    offset = string_to[1] - string_to[0]
                    next_node = current.out.get(child)
                    break

            if next_node is not None:
                current = next_node
                string_to = (string_to[0] + (current.lab[1] - current.lab[0]), string_to[1])
                if offset < next_node.lab[1] - next_node.lab[0]:
                    return current, offset, matching_prefix #edge
            else:
                return current, offset, matching_prefix #node, offset = 0

    s += "$"
    root = Node((0, 0))
    head = root
    head.slink = root
    tail = root.addChild(Node((0, len(s)), leafnum=0))

    for i in range(1, len(s)):
        new_head = None
        suf = s[i:]
        if head == root:
            node, offset, matching_prefix = slowscan(root, (i, len(s)))
            tail = split_or_add(node, offset, matching_prefix)
            head = tail.parent
            continue
        u = head.parent
        v = head.lab
        if not u == root:
            w, w_offset, w_prefix = fastscan(u.slink, v)
        else:
            w, w_offset, w_prefix = fastscan(root, (v[0]+1, v[1]))
        if w_offset != 0:
            tail = split_or_add(w, w_offset, w_prefix)
            w = tail.parent
            new_head = w
        else: #w is a node
            node, offset, matching_prefix = slowscan(w, tail.lab)
            tail = split_or_add(node, offset, w_prefix + matching_prefix)
            new_head = tail.parent

        head.slink = w
        head = new_head

    return root


#Helper for computing LL(v)
def leafcollector(node):
    #Return a list of all leaves in the subtree rooted at node, represented by the leafnum
    if len(node.out) == 0:
        return [node]
    else:
        leaves = []
        for child in node.out:
            leaves += leafcollector(node.out.get(child))
        return leaves

def path_label_length(node):
    #Return the length of the label of the path from the root to node
    if type(node.lab) is str:
        if node.parent is None:
            return len(node.lab)
        else:
            return path_label_length(node.parent) + len(node.lab)
    else: #index representation
        if node.parent is None:
            return 0
        else:
            return path_label_length(node.parent) + (1 if node.lab[0] == node.lab[1] else node.lab[1] - node.lab[0])

def internal_nodes(node):
    #Return a list of all internal nodes in the subtree rooted at node, traverse by dfs
    if len(node.out) == 0:
        return []
    else:
        nodes = []
        nodes.append(node)
        for child in node.out:
            nodes += internal_nodes(node.out.get(child))
        return nodes

def dfs(node, acc = 0):
    # Perform a dfs traversal of the tree, and assign a dfsnum to each node.
    if len(node.out) == 0:
        node.dfsnum = (acc, acc)
        acc += 1
    else:
        node.dfsnum = acc
        for child in node.out:
            acc = dfs(node.out.get(child), acc)
        node.dfsnum = (node.dfsnum, acc - 1)
    return acc

def remove_vprime(node):
    # takes a node, and removes the v' child of v, where v' is the child of v whose leaf list is largest over all children of v
    maxdfs = max(list(map(lambda child: child.dfsnum[1] - child.dfsnum[0] + 1, node.out.values()))) #linear time
    if maxdfs == 1:
        return node
    else:
        for child in node.out.values(): #linear time
            if child.dfsnum[1] - child.dfsnum[0] + 1 == maxdfs:
                node.out.pop(child.lab)
                return node

def find_left_rotations(s, btr):
    # Takes a string s, and list of indexes of branching tandem repeats in s, and performs left-rotations to find the non-branching tandem repeats.
    # Returns all the tandem repeats in a list of tuples (including the branching ones)

    s += "$"
    tr = []
    for (i, j) in btr:
        tr.append((i, j))
        inc = 1
        while s[i - inc] == s[j - inc]:
            tr.append((i - inc, j - inc))
            inc += 1
    return tr

def stoye_gusfield_basic(node, s):
    #The first iteration of the "basic" algorithm, for finding all branching tandem repeats (k=2) in a string, as described in
    #chapter 3.1 of "Simple and Flexible Detection of Contiguous Repeats Using a Suffix Tree" by Stoye and Gusfield.
    #Runs in n^3 time

    s+= "$"
    repeats = set()
    #iterate over all internal nodes
    for child in internal_nodes(node):
        leaves = leafcollector(child)
        for leaf in leaves:
            if any(leaf.leafnum + path_label_length(child) == l.leafnum for l in leaves):
                if not s[leaf.leafnum] == s[(leaf.leafnum) + 2 * path_label_length(child)]:
                    repeats.add((leaf.leafnum, leaf.leafnum + 2 * (path_label_length(child))))
    return repeats

def stoye_gusfield_dfs(node, s):
    #The second iteration of the "basic" algorithm, with the dfs optimization
    s += "$"
    dfs(node)

    #theres definently a better way to do this
    leaves = leafcollector(node)
    dfsnums = [l.dfsnum for l in leaves]
    leafnums = [l.leafnum for l in leaves]

    dfs_by_leaf = [dfsnums[leafnums.index(i)] for i in range(len(s))]
    repeats = set()
    #iterate over all internal nodes
    for child in internal_nodes(node):
        leaves = leafcollector(child)
        for leaf in leaves:
            if child.dfsnum[0] <= dfs_by_leaf[leaf.leafnum + path_label_length(child)][0] <= child.dfsnum[1]:
                if not s[leaf.leafnum] == s[(leaf.leafnum) + 2 * path_label_length(child)]:
                    repeats.add((leaf.leafnum, leaf.leafnum + 2 * (path_label_length(child))))
    return repeats

def stoye_gusfield_optimized(node, s):
    #The third iteration of the "basic" algorithm, with both the dfs optimization and the v' optimization.
    #runs in time O(n log n), allegedly

    s += "$"
    dfs(node)

    #theres definently a better way to do this
    leaves = leafcollector(node)
    dfsnums = [l.dfsnum for l in leaves]
    leafnums = [l.leafnum for l in leaves]
    dfs_by_leaf = [dfsnums[leafnums.index(i)] for i in range(len(s))]

    repeats = set()

    #iterate over all internal nodes, except the one with the leaf list
    for child in internal_nodes(node):

        Dv = path_label_length(child)
        for leaf in leafcollector(remove_vprime(child)):

            #step 2b
            if child.dfsnum[0] <= dfs_by_leaf[leaf.leafnum + Dv][0] <= child.dfsnum[1]:
                if not s[leaf.leafnum] == s[(leaf.leafnum) + 2 * Dv]:
                    repeats.add((leaf.leafnum, leaf.leafnum + 2 * (Dv)))
            #step 2c
            if child.dfsnum[0] <= dfs_by_leaf[leaf.leafnum - Dv][0] <= child.dfsnum[1]:
                if not s[leaf.leafnum] == s[(leaf.leafnum - Dv) + 2 * Dv]:
                    repeats.add((leaf.leafnum - Dv, leaf.leafnum - Dv + 2 * (Dv)))

    return repeats

def tandem_repeats_brute_force(string):
    repeats = []
    n = len(string)
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % 2 == 0:
                mid = (i + j) // 2
                if string[i:mid] == string[mid:j]:
                    if j < n and string[mid] != string[j]:
                        repeats.append((i, j))
    return repeats

#String generators for testing, helpers and more
def fibstring(n):
    if n == 0:
        return "A"
    elif n == 1:
        return "B"
    else:
        return fibstring(n - 1) + fibstring(n - 2)


def generate_alphabet(size):
    alphabet = []

    # Adding capital letters (A-Z)
    for i in range(65, 91):
        alphabet.append(chr(i))
        if len(alphabet) == size:
            return alphabet

    # Adding lowercase letters (a-z)
    for i in range(97, 123):
        alphabet.append(chr(i))
        if len(alphabet) == size:
            return alphabet

    # Adding numbers (0-9)
    for i in range(48, 58):
        alphabet.append(chr(i))
        if len(alphabet) == size:
            return alphabet

    # Adding special characters
    alphabet.append("!")
    alphabet.append("@")
    alphabet.append("#")
    alphabet.append("%")
    alphabet.append("^")
    alphabet.append("&")





    return alphabet

def random_string(n, sigma = 4):
    alphabet = generate_alphabet(sigma)
    import random
    genome = ""
    for i in range(0, n):
        genome += random.choice(alphabet)
    return genome

def print_tree(root, string = None, level=0):
    s = "{lab:<20}       ln:{ln:<10}    dfsnum:{dfsnum:<10}       slink:{slink:<10}".format(
        lab = "━━" * (level - 1) + str(root.lab) if string is None else "━━" * (level - 1) + str(root.lab) + "  " + string[root.lab[0]:root.lab[1]],
        ln = str(root.leafnum),
        dfsnum = str(root.dfsnum),
        slink = str(root.slink.lab if root.slink else str(root.slink))
    )

    print(s)
    for child in root.out:
        print_tree(root.out.get(child), string, level + 1)
    return root

def print_just_the_tree(root, level=0):
    s = "{lab:<20} ".format(
        lab = "━━" * (level - 1) + str(root.lab),
    )

    print(s)
    for child in root.out:
        print_just_the_tree(root.out.get(child), level + 1)
    return root

def print_index_collection(indexes, s):
    for i in indexes:
        print("{index:<10}    {repeat}".format(index = str(i), repeat = s[i[0]:i[1]]))

def test_naive_vs_mccreight(s):
    start_nv = time.time()
    nv_stree = suftree_naive(s)
    end_nv = time.time()
    time_nv = end_nv - start_nv

    start_mc = time.time()
    mc_stree = mccreight(s)
    end_mc = time.time()
    time_mc = end_mc - start_mc

    print("    naive:  " + str(time_nv))
    print("mccreight:  " + str(time_mc))

    return time_nv, time_mc

def test_runningtimes(s, print_repeats = False):
    print(s)

    start_bf = time.time()
    repeats_bf = tandem_repeats_brute_force(s)
    end_bf = time.time()
    time_bf = end_bf - start_bf

    start_suftree = time.time()
    stree = suftree_naive(s)
    end_suftree = time.time()
    time_suftree = end_suftree - start_suftree

    start_sgb = time.time()
    repeats_sgb = stoye_gusfield_basic(stree, s)
    end_sgb = time.time()
    time_sgb = end_sgb - start_sgb

    start_sgdfs = time.time()
    repeats_sgdfs = stoye_gusfield_dfs(stree, s)
    end_sgdfs = time.time()
    time_sgdfs = end_sgdfs - start_sgdfs

    start_sgo = time.time()
    repeats_sgo = stoye_gusfield_optimized(stree, s)
    end_sgo = time.time()
    time_sgo = end_sgo - start_sgo


    print("{construction_alg} construction took {time:f} seconds".format(construction_alg="Suffix tree naive", time=time_suftree))

    print("Algorithm:                Number of repeats:        Time:")
    form = "{repeat_alg:<25} {num_repeats:<25} {time:f}"

    print(form.format(repeat_alg="Stoye-Gusfield Basic", num_repeats=len(repeats_sgb), time=time_sgb))
    if(print_repeats):
        print_index_collection(repeats_sgb, s)

    print(form.format(repeat_alg="Stoye-Gusfield DFS", num_repeats=len(repeats_sgdfs), time=time_sgdfs))
    if (print_repeats):
        print_index_collection(repeats_sgdfs, s)

    print(form.format(repeat_alg="Stoye-Gusfield Optimized", num_repeats=len(repeats_sgo), time=time_sgo))
    if (print_repeats):
        print_index_collection(repeats_sgo, s)

    print(form.format(repeat_alg="Brute force", num_repeats=len(repeats_bf), time=time_bf))
    if (print_repeats):
        print_index_collection(repeats_bf, s)

def plot_runtimes(s):
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    import time

    times_sgb = []
    times_sgdfs\
        = []
    times_bf = []

    for i in range(1, len(s)):

        #start_sgb = time.time()
        #repeats_sgb = stoye_gusfield_basic(suftree_naive_indexed(s[:i]), s[:i])
        #end_sgb = time.time()
        #time_sgb = end_sgb - start_sgb
        #times_sgb.append(time_sgb)

        start_sgdfs\
            = time.time()
        repeats_sgdfs\
            = stoye_gusfield_dfs(suftree_naive(s[:i]), s[:i])
        end_sgdfs\
            = time.time()
        time_sgdfs\
            = end_sgdfs\
              - start_sgdfs

        times_sgdfs\
            .append(time_sgdfs
                    )

        start_bf = time.time()
        repeats_bf = tandem_repeats_brute_force(s[:i])
        end_bf = time.time()
        time_bf = end_bf - start_bf
        times_bf.append(time_bf)

    #plt.plot(range(1, len(s)), times_sgb, label="Stoye-Gusfield Basic")
    plt.plot(range(1, len(s)), times_sgdfs
             , label="Stoye-Gusfield Optimized")
    plt.plot(range(1, len(s)), times_bf, label="Brute force")
    plt.legend()
    plt.show()

def test_dfsnums():
    s = "mississippi"
    stree = suftree_naive(s)
    dfs(stree)
    leaves = leafcollector(stree)
    leafnums = [l.leafnum for l in leaves]
    dfsnums = [l.dfsnum for l in leaves]

    # the dfsnums indexed by the leafnums
    dfs_by_leaf = [dfsnums[[l.leafnum for l in leafcollector(stree)].index(i)] for i in range(len(s) + 1)]
    print("leafnums:    "  +  str(leafnums))
    print("dfsnums:     "  +  str(dfsnums))
    print("dfs_by_leaf: "  +  str(dfs_by_leaf))

    repeats = stoye_gusfield_dfs(stree, s)

    print_index_collection(repeats, s)

def test_sgdfs(s):
    stree = suftree_naive(s)
    dfs(stree)

    print_tree(stree)

    repeats = stoye_gusfield_dfs(stree, s)
    print_index_collection(repeats, s)

def test_sgo():
    s = "mississippi"
    stree = suftree_naive(s)
    print_tree(stree)
    repeats = stoye_gusfield_optimized(stree, s)
    print_index_collection(repeats, s)

def test_sgo_vs_sgdfs():
    s = random_string(1000)
    print(s)
    stree = suftree_naive(s)
    dfs(stree)
    print_tree(stree)
    print("SGDFS")
    repeats_sgdfs = stoye_gusfield_dfs(stree, s)
    print_index_collection(repeats_sgdfs, s)
    print("SGO")
    repeats_sgo = stoye_gusfield_optimized(stree, s)
    print_index_collection(repeats_sgo, s)

    if(repeats_sgo != repeats_sgdfs):
        print("ERROR: results differ")

def write_string_to_file(s):
    with open("test.txt", "w") as f:
        f.write(s)

def file_to_string(filename):
    with open(filename, "r") as f:
        return f.read()


def compare_strings(limit):
    strings = {}
    strings.update({"period1": "a"*10000})
    strings.update({"period2": "ab" * 5000})
    strings.update({"period4": "abcd" * 2500})
    strings.update({"period8": "abcdefgh" * 1250})
    strings.update({"period16": "abcdefghijklmnop" * 1250})
    strings.update({"fib": fibstring(20)[:10000]})
    strings.update({"rand2": random_string(10000, 2)})
    strings.update({"rand4": random_string(10000, 4)})
    strings.update({"rand8": random_string(10000, 8)})
    strings.update({"rand16": random_string(10000, 16)})


    form = "{input:<10}      {mc:<25}       {nv:<25}       {diff:<25}"
    print(form.format(input = "input", mc = "McCreight", nv = "Naive", diff = "Difference"))
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    res = []
    for test in strings:
        s = strings.get(test)[:limit]

        start_mc = time.time()
        stree = mccreight(s)
        end_mc = time.time()
        time_mc = end_mc - start_mc

        start_nv = time.time()
        stree = suftree_naive(s)
        end_nv = time.time()
        time_nv = end_nv - start_nv

        res.append((test, time_mc, time_nv))

        print(form.format(input = test, mc = time_mc, nv = time_nv, diff = time_mc - time_nv))


    #Plot the results in a bar chart with two columns per string
    import matplotlib.pyplot as plt
    import numpy as np

    names = [x[0] for x in res]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, [x[1] for x in res], width, label='McCreight')
    rects2 = ax.bar(x + width/2, [x[2] for x in res], width, label='Naive')

    ax.set_ylabel('Time')
    ax.set_title('Time to construct suffix tree')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    fig.tight_layout()

    plt.show()

def compare_rand_alphabetsizes(stringlimit, samples_per_size):

    res = []
    for i in range(2, 40):
        group = []
        for j in range(1, samples_per_size):
            print(i)
            s = random_string(stringlimit, i)
            time_mc = timeit.timeit("mccreight(\"{string}\")".format(string=s), setup="from __main__ import mccreight",number=1)
            time_nv = timeit.timeit("suftree_naive(\"{string}\")".format(string=s), setup="from __main__ import suftree_naive", number=1)
            group.append((time_mc, time_nv))
            #append the mean of both mc and nv as a final tuple in the group
            group.append((sum([x[0] for x in group])/len(group), sum([x[1] for x in group])/len(group)))

        res.append((i, group))


    #With alphabet size along the x-axis, and time to construct suffix tree along the y-axis
    #plot the time to construct a suffix tree for a random string of length 10000 for McCreight and Naive respectively
    #For each alphabet size, there are "samples_per_size" samples of both mc and nv. These should be plotted as points on two vertical line, one for mc and one for nv, with a slight distance between them
    #The mean of the samples should be plotted as a horizontal line, with a slight distance between the two lines

    #plot the results in a bar chart with two columns per string
    import matplotlib.pyplot as plt
    import numpy as np

    names = [x[0] for x in res]
    means_mc = [x[1][-1][0] for x in res]
    means_nv = [x[1][-1][1] for x in res]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots()

    #Plot the means in a line chart
    ax.plot(x, means_mc, label='McCreight')
    ax.plot(x+0.2, means_nv, label='Naive')

    #Plot the samples in a scatter plot
    for i in range(len(res)):
        for j in range(len(res[i][1])-1):
            ax.scatter(i, res[i][1][j][0], color="blue", alpha = 0.1, marker=".")
            ax.scatter(i+0.2, res[i][1][j][1], color="orange", alpha = 0.1, marker=".")

    ax.set_ylabel('Time')
    ax.set_xlabel('Alphabet size')
    ax.set_title('Alphabet size vs time to construct suffix tree')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    fig.tight_layout()

    plt.show()

def plot_mccreight_vs_naive(limit, interval, samples_per_point):

    res = []
    for i in range(1, limit):
        print(i*interval)
        s = random_string(i*interval, 20)
        try:
            time_mc = timeit.timeit("mccreight(\"{string}\")".format(string=s), setup="from __main__ import mccreight",number=samples_per_point)
            time_nv = timeit.timeit("suftree_naive(\"{string}\")".format(string=s), setup="from __main__ import suftree_naive", number=samples_per_point)
            res.append((i*interval, time_mc, time_nv))
        except:
            continue

    #Plot the results in a line chart

    import matplotlib.pyplot as plt
    import numpy as np

    names = [x[0] for x in res]
    x = np.arange(len(names))

    fig, ax = plt.subplots()
    # show the value markings on the x axis in increments of 200
    ax.plot(x, [x[1] for x in res], label='McCreight')
    ax.plot(x, [x[2] for x in res], label='Naive')


    ax.set_ylabel('Time')
    ax.set_xlabel('String length')
    ax.set_title('String length vs time to construct suffix tree')
    ax.legend()

    fig.tight_layout()

    plt.show()

def plot_stoye_gusfield(limit, interval, samples_per_point):


    res = []
    for i in range(1, limit):
        print(i*interval)
        s = random_string(i*interval, 1)
        stree = suftree_naive(s)
        repeats = len(stoye_gusfield_optimized(copy.deepcopy(stree), s))

        time_sgb = 0
        time_sgdfs = timeit.timeit(stmt=lambda: stoye_gusfield_dfs(stree, s), number=samples_per_point)
        time_sgo = timeit.timeit(stmt = lambda: stoye_gusfield_optimized(stree, s), number=samples_per_point)

        res.append((i*interval, repeats, time_sgb, time_sgdfs, time_sgo))


    #Plot the results in a line chart

    import matplotlib.pyplot as plt
    import numpy as np


    stringlength = [x[0] for x in res]
    repeats = [x[1] for x in res]
    executionTime_sgb = [x[2] for x in res]
    executionTime_sgdfs = [x[3] for x in res]
    executionTime_sgo = [x[4] for x in res]
    x = np.arange(len(stringlength))

    fig, ax = plt.subplots()
    # show the value markings on the x axis in increments of 200
    # sgb = ax.plot(stringlength, executionTime_sgb, label='SGB')
    sgdfs = ax.plot(stringlength, executionTime_sgdfs, color = "orange", label='SGDFS')
    sgo = ax.plot(stringlength, executionTime_sgo, color = "green", label='SGO')
    # ax.set_yscale("log")

    ax2 = ax.twinx()
    repeats = ax2.plot(stringlength, repeats, color="black", alpha=0.4, label="# of Repeats")

    lns = sgdfs + sgo + repeats
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)


    fig.tight_layout()

    plt.show()





if __name__=='__main__':

    plot_stoye_gusfield(150, 1, 1)




#(0, 0)                     ln:None          dfsnum:None             slink:(0, 0)
#(1, 2)                     ln:0             dfsnum:None             slink:(0, 0)
#━━(9, 18)                  ln:8             dfsnum:None             slink:None
#━━(2, 3)                   ln:0             dfsnum:None             slink:(1, 2)
#━━━━(9, 18)                ln:7             dfsnum:None             slink:None
#━━━━(3, 4)                 ln:0             dfsnum:None             slink:(2, 3)
#━━━━━━(9, 18)              ln:6             dfsnum:None             slink:None
#━━━━━━(4, 5)               ln:0             dfsnum:None             slink:(3, 4)
#━━━━━━━━(9, 18)            ln:5             dfsnum:None             slink:None
#━━━━━━━━(5, 6)             ln:0             dfsnum:None             slink:(4, 5)
#━━━━━━━━━━(9, 18)          ln:4             dfsnum:None             slink:None
#━━━━━━━━━━(6, 7)           ln:0             dfsnum:None             slink:(5, 6)
#━━━━━━━━━━━━(9, 18)        ln:3             dfsnum:None             slink:None
#━━━━━━━━━━━━(7, 8)         ln:0             dfsnum:None             slink:(6, 7)
#━━━━━━━━━━━━━━(9, 18)       ln:2             dfsnum:None             slink:None
#━━━━━━━━━━━━━━(17, 18)       ln:10            dfsnum:None             slink:None
#━━━━━━━━━━━━━━(8, 9)       ln:0             dfsnum:None             slink:None
#━━━━━━━━━━━━━━━━(17, 18)       ln:12            dfsnum:None             slink:None
#━━━━━━━━━━━━━━━━(9, 18)       ln:1             dfsnum:None             slink:None
#━━━━━━━━━━━━(13, 18)       ln:11            dfsnum:None             slink:None
#(0, 8)                     ln:0             dfsnum:None             slink:(7, 8)
#━━(17, 18)                 ln:9             dfsnum:None             slink:None
#━━(8, 18)                  ln:0             dfsnum:None             slink:None

#(0, 0)                     ln:None          dfsnum:None             slink:None
#(1, 2)                     ln:0             dfsnum:None             slink:None
#━━(9, 18)                  ln:8             dfsnum:None             slink:None
#━━(2, 3)                   ln:0             dfsnum:None             slink:None
#━━━━(9, 18)                ln:7             dfsnum:None             slink:None
#━━━━(3, 4)                 ln:0             dfsnum:None             slink:None
#━━━━━━(9, 18)              ln:6             dfsnum:None             slink:None
#━━━━━━(4, 5)               ln:0             dfsnum:None             slink:None
#━━━━━━━━(9, 18)            ln:5             dfsnum:None             slink:None
#━━━━━━━━(5, 6)             ln:0             dfsnum:None             slink:None
#━━━━━━━━━━(9, 18)          ln:4             dfsnum:None             slink:None
#━━━━━━━━━━(6, 7)           ln:0             dfsnum:None             slink:None
#━━━━━━━━━━━━(9, 18)        ln:3             dfsnum:None             slink:None
#━━━━━━━━━━━━(7, 8)         ln:0             dfsnum:None             slink:None
#━━━━━━━━━━━━━━(9, 18)       ln:2             dfsnum:None             slink:None
#━━━━━━━━━━━━━━(8, 18)       ln:1             dfsnum:None             slink:None
#━━━━━━━━━━━━━━(17, 18)       ln:10            dfsnum:None             slink:None
#━━━━━━━━━━━━(17, 18)       ln:11            dfsnum:None             slink:None
#━━━━━━━━━━(17, 18)         ln:12            dfsnum:None             slink:None
#(0, 8)                     ln:0             dfsnum:None             slink:None
#━━(17, 18)                 ln:9             dfsnum:None             slink:None
#━━(8, 18)                  ln:0             dfsnum:None             slink:None

