import pandas as pd
import numpy as np


class RuleBased:
    """
    form filling based on association rule mining
    """
    @staticmethod
    def rule_generation(train, min_support, min_confidence):
        """generate association rules
        param train: training set (dataframe)
        param config: config file
        return rules:list of rules
        """
        transactions1 = train.values.tolist()

        for i in range(len(transactions1)):
            transactions1[i] = [str(j) for j in transactions1[i]]
        columns_l = train.columns.tolist()
        transactions = []
        for i in range(len(transactions1)):
            transactions.append([i + '=' + j for i, j in zip(columns_l, transactions1[i])])
        #print(transactions)
        result = []
        res_for_rul = {}
        for itemset, support in Rules.find_frequent_itemsets(transactions, min_support, True):
            result.append((itemset, support))
            res_for_rul[tuple(itemset)] = support

        result = sorted(result, key=lambda i: i[0])
        patterns = pd.DataFrame(result, columns=["pattern", "support"])
        rules = Rules.assoc_rule(res_for_rul, min_confidence)
        frame_rules = pd.DataFrame(rules,columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Conviction"])

        return rules, patterns, frame_rules

    @staticmethod
    def rules_filtering(rules, unknown_ids):
        """
        filter the association rule with more than one consequent
        param rules: list of rules
        return fnl_rules: list of the filtered rules
        """
        def entries_to_remove(entries, the_dict):  # delete keys of a dictionnary
            for key in entries:
                if key in the_dict:
                    del the_dict[key]

        fnl_rules = []
        for rule in rules:  # select rules with only one consequent
            consequent = rule['Consequent']
            if len(rule['Consequent']) == 1:
                # array = consequent[0].split("=")
                # if unknown_ids[array[0]] != array[1]:   # remove rule that the consequent is unknown
                rule['Antecedent'] = set(rule['Antecedent'])
                rule['Consequent'] = rule['Consequent'][0]
                fnl_rules.append(rule)

        # entries= ['Confidence','Conviction','Lift','Support']
        entries = ['Conviction', 'Lift']
        for i in range(len(fnl_rules)):  # create rules that only contain the antecedent and the consequent
            entries_to_remove(entries, fnl_rules[i])

        return fnl_rules

    @staticmethod
    def prepare_test_set(test, unknown_ids, target):
        """
        transform test instances to a dictionary
        of the form Antecedent: antecedent, Consequent: consequent
        this form will help to measure the context matching score of each new instance
        param test_set: test
        param unknown_ids: the id for unknown
        param target: the current target
        return test_for_prediction: test set for prediction (in the format {Antecedent, Consequent)
        return true_values: list of true values
        """
        true_values = test[target].values  # get the true value from the target
        test_copy = test.copy(deep=True)
        test_copy.drop("target", axis=1, inplace=True)  # eliminate the "target" column
        n_instances = test_copy.values.tolist()
        for i in range(len(n_instances)):
            n_instances[i] = [str(j) for j in n_instances[i]]
        columns_t = test_copy.columns.tolist()

        # input instances 'key=value' (string)
        input_instances = []
        for i in range(len(n_instances)):
            antecedent = set()
            for i, j in zip(columns_t, n_instances[i]):
                if j != unknown_ids[i] and i != target:
                    antecedent.add(i + '=' + j)
            input_instances.append(antecedent)
        # true_value 'key=value' (string)
        consequence = []
        for i in range(len(true_values)):
            consequence.append(target + '=' + str(true_values[i]))

        test_for_prediction = []
        for i in range(
                len(n_instances)):  # dictionary of the new instance antecedent "context" , consequence "true value"
            test_for_prediction.append({'Antecedent': input_instances[i], 'Consequent': consequence[i]})
        return test_for_prediction, consequence

    @staticmethod
    def rules_matching(fnl_rules,target):
        """
        Step 1 Matching rules : Select rules with the consequent that equal value of the target
        param fnl_rules: final rules
        param target: current target
        returns matched_rules: matched rules
        """
        matched_rules = []
        for item in fnl_rules:  # rules only have one consequent [0]
            if target in item['Consequent']:
                matched_rules.append(item)
        return matched_rules

    @staticmethod
    def calc_matching_score(matched_rules, test_instance, matched_rules_df):
        """
        Compute the context matching score between the test_instance and the different rules
        param matched_rules: list of matched rules
        param test_instance:  dict of the test instance
        param matched_rules_df: dataframe of the matched rules
        return matched_rules_df: add a new column called 'Score'
        """
        context_score = []
        for i in range(len(matched_rules)):
            intersection_size = len(matched_rules[i]['Antecedent'].intersection(test_instance['Antecedent']))
            union_size = len(matched_rules[i]['Antecedent'].union(test_instance['Antecedent']))
            context_matching_score = 1.0 * intersection_size / union_size
            context_score.append(context_matching_score)
        confidence_score = matched_rules_df['Confidence']
        score = np.array(context_score) * np.array(confidence_score)
        matched_rules_df['Score'] = score  # Step 2.2 Calculate recommendation score
        return matched_rules_df

    @staticmethod
    def rule_ranking(target, test_instance, unknown_id, matched_rules_df):
        """
        rank different matched rules according to score
        param target: the current target field
        param test_instance: the testing input instance
        param matched_rules_df: dataframe of the matched rules
        return ranked_value: a list of suggested values ranked by score
        """
        # ranked consequents based on 'score'. if the score is the same, rank according to the frequency
        unknown_id = str(target+"="+unknown_id)

        matched_rules_group = matched_rules_df.groupby('Score')
        ranked_consequent = []
        not_zero = False

        for score, matched_rules in matched_rules_group:
            # method 1
            if score > 0.0:
                not_zero = True
            sorted_consequents = matched_rules.sort_values(['Support'], ascending=False)
            unique_consequent_set = set()
            unique_consequent_list = []
            for consequent in sorted_consequents['Consequent']:
                if consequent not in unique_consequent_set and consequent != unknown_id:
                    unique_consequent_list.append(consequent)
                    unique_consequent_set.add(consequent)
            ranked_consequent.append({'Score': score, 'Consequents': unique_consequent_list})

            # method 2
            # unique_consequent_list = matched_rules['Consequent'].value_counts().index.tolist()
            # ranked_consequent.append({'Score':score, 'Consequents':unique_consequent_list})
        ranked_consequent = pd.DataFrame(ranked_consequent)
        ranked_consequent = ranked_consequent.sort_values(['Score'], ascending=False)

        # take the ranked consequents as the suggestions
        ranked_value = []
        exist_values = set()
        for index, row in ranked_consequent.iterrows():
            consequents = row['Consequents']
            for consequent in consequents:
                if consequent not in exist_values:
                    ranked_value.append(consequent)
                    exist_values.add(consequent)

        return ranked_value, not_zero


class Rules:
    """
    algorithm for association rule mining (from internet)
    nothing changes
    """
    @staticmethod
    def find_frequent_itemsets(transactions, minimum_support, include_support=False):

        """
        Find frequent itemsets in the given transactions using FP-growth.

            Parameters:
                transactions: can be any iterable of iterables of items.

                minimum_support: Integer
                specifies the minimum number of occurrences of an itemset to be accepted.

                Include support: Boolean
                If `include_support` is true, yield (itemset, support) pairs instead of
                just the itemsets.

            Returns:
                This function returns a generator of items.

        """

        from collections import defaultdict, namedtuple

        items = defaultdict(lambda: 0)  # mapping from items to their supports

        # if using support rate instead of support count
        if 0 < minimum_support <= 1:
            minimum_support = minimum_support * len(transactions)

        # Load the passed-in transactions and count the support that individual
        # items have.
        for transaction in transactions:
            for item in transaction:
                items[item] += 1

        # Remove infrequent items from the item support dictionary.
        items = dict(
            (item, support) for item, support in items.items() if support >= minimum_support
        )

        # Build our FP-tree. Before any transactions can be added to the tree, they
        # must be stripped of infrequent items and their surviving items must be
        # sorted in decreasing order of frequency.
        def clean_transaction(transaction):
            transaction = filter(lambda v: v in items, transaction)
            transaction = sorted(transaction, key=lambda v: items[v], reverse=True)
            return transaction

        master = FPTree()
        for transaction in list(map(clean_transaction, transactions)):
            master.add(transaction)

        def find_with_suffix(tree, suffix):
            for item, nodes in tree.items():
                support = sum(n.count for n in nodes)
                if support >= minimum_support and item not in suffix:
                    # New winner!
                    found_set = [item] + suffix
                    yield (found_set, support) if include_support else found_set

                    # Build a conditional tree and recursively search for frequent
                    # itemsets within it.
                    cond_tree = Rules.conditional_tree_from_paths(tree.prefix_paths(item))
                    for s in find_with_suffix(cond_tree, found_set):
                        yield s  # pass along the good news to our caller

        # Search for frequent itemsets, and yield the results we find.
        for itemset in find_with_suffix(master, []):
            yield itemset

    @staticmethod
    def conditional_tree_from_paths(paths):
        """
        Build a conditional FP-tree from the given prefix paths.

        Parameters:
            paths:
                List of transactions
        Returns:
            FP-tree
        """
        tree = FPTree()
        condition_item = None
        items = set()

        # Import the nodes in the paths into the new tree. Only the counts of the
        # leaf notes matter; the remaining counts will be reconstructed from the
        # leaf counts.
        for path in paths:
            if condition_item is None:
                condition_item = path[-1].item

            point = tree.root
            for node in path:
                next_point = point.search(node.item)
                if not next_point:
                    # Add a new node to the tree.
                    items.add(node.item)
                    count = node.count if node.item == condition_item else 0
                    next_point = FPNode(tree, node.item, count)
                    point.add(next_point)
                    tree._update_route(next_point)
                point = next_point

        assert condition_item is not None

        # Calculate the counts of the non-leaf nodes.
        for path in tree.prefix_paths(condition_item):
            count = path[-1].count
            for node in reversed(path[:-1]):
                node._count += count

        return tree

    @staticmethod
    def subs(l):
        """
        Used for assoc_rule
        """
        assert type(l) is list
        if len(l) == 1:
            return [l]
        x = Rules.subs(l[1:])
        return x + [[l[0]] + y for y in x]

    @staticmethod
    # Association rules
    def assoc_rule(freq, min_conf=0.6):
        """
        This assoc_rule must input a dict for itemset -> support rate
        And also can customize your minimum confidence

        Parameters:
            freq: Dictionnary

            min_conf: double
            Specifies the minimum confidence of the associaion rules
        Returns:
            list:
            List of kept association rules

        """
        assert type(freq) is dict
        result = []
        for item, sup in freq.items():
            for subitem in Rules.subs(list(item)):
                sb = [x for x in item if x not in subitem]
                if sb == [] or subitem == []:
                    continue
                if len(subitem) == 1 and (subitem[0][0] == "in" or subitem[0][0] == "out"):
                    continue
                conf = sup / freq[tuple(subitem)]
                lift = (conf / sup)
                conviction = (1 - sup) / ((1 - conf) + 0.1)
                if conf >= min_conf:
                    result.append(
                        {"Antecedent": subitem, "Consequent": sb, "Support": sup, "Confidence": conf, "Lift": lift,
                         "Conviction": conviction})
        return result

# basic tree structure to mine rules in the memory
class FPNode(object):
    """A node in an FP tree."""

    def __init__(self, tree, item, count=1):
        self._tree = tree
        self._item = item
        self._count = count
        self._parent = None
        self._children = {}
        self._neighbor = None

    def add(self, child):
        """Add the given FPNode `child` as a child of this node."""

        if not isinstance(child, FPNode):
            raise TypeError("Can only add other FPNodes as children")

        if child.item not in self._children:
            self._children[child.item] = child
            child.parent = self

    def search(self, item):
        """
        Check whether this node contains a child node for the given item.
        If so, that node is returned; otherwise, `None` is returned.
        """
        try:
            return self._children[item]
        except KeyError:
            return None

    def __contains__(self, item):
        return item in self._children

    @property
    def tree(self):
        """The tree in which this node appears."""
        return self._tree

    @property
    def item(self):
        """The item contained in this node."""
        return self._item

    @property
    def count(self):
        """The count associated with this node's item."""
        return self._count

    def increment(self):
        """Increment the count associated with this node's item."""
        if self._count is None:
            raise ValueError("Root nodes have no associated count.")
        self._count += 1

    @property
    def root(self):
        """True if this node is the root of a tree; false if otherwise."""
        return self._item is None and self._count is None

    @property
    def leaf(self):
        """True if this node is a leaf in the tree; false if otherwise."""
        return len(self._children) == 0

    @property
    def parent(self):
        """The node's parent"""
        return self._parent

    @parent.setter
    def parent(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a parent.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a parent from another tree.")
        self._parent = value

    @property
    def neighbor(self):
        """
        The node's neighbor; the one with the same value that is "to the right"
        of it in the tree.
        """
        return self._neighbor

    @neighbor.setter
    def neighbor(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a neighbor.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a neighbor from another tree.")
        self._neighbor = value

    @property
    def children(self):
        """The nodes that are children of this node."""
        return tuple(self._children.values())

    def inspect(self, depth=0):
        print(("  " * depth) + repr(self))
        for child in self.children:
            child.inspect(depth + 1)

    def __repr__(self):
        if self.root:
            return "<%s (root)>" % type(self).__name__
        return "<%s %r (%r)>" % (type(self).__name__, self.item, self.count)


class FPTree(object):
    """
     An FP tree.
     This object may only store transaction items that are hashable
     (i.e., all items must be valid as dictionary keys or set members).
    """
    from collections import defaultdict, namedtuple

    Route = namedtuple("Route", "head tail")

    def __init__(self):
        # The root node of the tree.
        self._root = FPNode(self, None, None)

        # A dictionary mapping items to the head and tail of a path of
        # "neighbors" that will hit every node containing that item.
        self._routes = {}

    @property
    def root(self):
        """The root node of the tree."""
        return self._root

    def add(self, transaction):
        """Add a transaction to the tree."""
        point = self._root

        for item in transaction:
            next_point = point.search(item)
            if next_point:
                # There is already a node in this tree for the current
                # transaction item; reuse it.
                next_point.increment()
            else:
                # Create a new point and add it as a child of the point we're
                # currently looking at.
                next_point = FPNode(self, item)
                point.add(next_point)

                # Update the route of nodes that contain this item to include
                # our new node.
                self._update_route(next_point)

            point = next_point

    def _update_route(self, point):
        """Add the given node to the route through all nodes for its item."""
        assert self is point.tree

        try:
            route = self._routes[point.item]
            route[1].neighbor = point  # route[1] is the tail
            self._routes[point.item] = self.Route(route[0], point)
        except KeyError:
            # First node for this item; start a new route.
            self._routes[point.item] = self.Route(point, point)

    def items(self):
        """
        Generate one 2-tuples for each item represented in the tree. The first
        element of the tuple is the item itself, and the second element is a
        generator that will yield the nodes in the tree that belong to the item.
        """
        for item in self._routes.keys():
            yield (item, self.nodes(item))

    def nodes(self, item):
        """
        Generate the sequence of nodes that contain the given item.
        """

        try:
            node = self._routes[item][0]
        except KeyError:
            return

        while node:
            yield node
            node = node.neighbor

    def prefix_paths(self, item):
        """Generate the prefix paths that end with the given item."""

        def collect_path(node):
            path = []
            while node and not node.root:
                path.append(node)
                node = node.parent
            path.reverse()
            return path

        return (collect_path(node) for node in self.nodes(item))

    def inspect(self):
        print("Tree:")
        self.root.inspect(1)

        print()
        print("Routes:")
        for item, nodes in self.items():
            print("  %r" % item)
            for node in nodes:
                print("    %r" % node)
