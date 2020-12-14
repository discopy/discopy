

"""
Implementing linear parser "LinPP" for pregroups.
The algorithm parses lists of Word objects (i.e. Words with codomain
given by rigid.Ty types, Ty), outputting an irreducible form.
The parser is correct, i.e. it reduces a string to the sentence type if and only if the string is a grammatical sentence, for lists of words satisfying the following restrictions:

1 - winding number for any simple type must be -1 <= z <= 1 (complexity 2).
2 - any critical fan is 'guarded', i.e. critical fans do not nest.

The proof of correctness can be found at:
"""

import logging
from discopy import *
from discopy.grammar import *
from discopy.grammar.pregroup import draw

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(message)s')


def check_reduction(ty_1, ty_2):
    """
    Checks whether simple types ty_1 ,ty_2 reduce to the empty type.


    Parameters
    ----------
    ty_1 : Ty class from discopy.rigid
        left type
    ty_2 : Ty class from discopy.rigid
        right type


    Example
    -------
    >>> check_reduction(Ty('n').l, Ty('n'))
    True
    >>> check_reduction(Ty('n'), Ty('n').r)
    True
    >>> check_reduction(Ty('n').r , Ty('n'))
    False
    """
    return (ty_1 == ty_2.l) or (ty_2 == ty_1.r)


def critical(ty):
    """
    Checks if type is critical (assuming types in dictionary satisfy z<=1).


    Paramenters
    -----------
    ty = Ty class from discopy.rigid


    Example
    -------
    >>> critical(Ty('n').r)
    True
    >>> critical(Ty('n').l)
    False
    >>> critical(Ty('n'))
    False
    """
    return ty.z > 0


class Stack:
    """
    A class to represent the stack of reduced types
    for a string being parsed.

    Attributes
    ----------
    types : list
        the stack containing the reduced pregroup types
    indices : list
        the stack containing the indices corresponding to the reduced
        pregroup types in the parsed string
    """

    def __init__(self):
        self.types = []
        self.indices = []

    def push_types(self, string, word_index, ty_index_start=None,
                   ty_index_end=None, backward=False):
        """
        Adding segments of words from the string to the stack (both types and
        corresponding indices).


        Parameters
        ----------
        string : list
            list of discopy Words (i.e. a sentence)
        word_index : int
            index of word in string that we want to push
        ty_index_start : None or int
            index of first simple type in word we want to push.
            If None, we push from the beginning of the word
        ty_index_end : None or int
           index of first simple type in word we do NOT want to push.
           If None, we push until the end of the word
        backward : Bool
            if True, will push types at the bottom of the stack.


        Example
        -------
        >>> stack = Stack()
        >>> Alice = Word('Alice', Ty('n'))
        >>> loves = Word('loves', Ty('n').r @ Ty('s') @ Ty('n').l)
        >>> string = [Alice, loves]
        >>> stack.push_types(string, 1, ty_index_start=1)
        >>> stack.types
        [Ty('s'), Ty(Ob('n', z=-1))]

        >>> stack.push_types(string,0, backward=True)
        >>> stack.types
        [Ty('n'), Ty('s'), Ty(Ob('n', z=-1))]
        """
        if ty_index_start is None:
            ty_index = 0
        else:
            ty_index = ty_index_start

        if ty_index_end is None:
            temp = len(string[word_index].cod)
        else:
            temp = ty_index_end

        while ty_index < temp:
            if backward is False:
                self.types.append(Ty(string[word_index].cod[ty_index]))
                self.indices.append((word_index, ty_index))
            else:
                self.types.insert(0, Ty(string[word_index].cod[ty_index]))
                self.indices.insert(0, (word_index, ty_index))
            ty_index += 1

    def pop(self, i=-1):
        """
        Pops a type out of stack in position i.


        Parameters
        -----------
        i = int
            indicates the index of the type we want to pop out
            (default i=-1, i.e. the top of the stack)


        Returns
        -------
        pair of integers (word_index, ty_index).
            coordinates of the type in the string
        """
        self.types.pop(i)
        coordinates = self.indices.pop(i)
        return coordinates

    def add(self, string, word_index, ty_index, backward=False):
        """
        Adds one type to the stack from the string


        Parameters
        ----------
        string : list
            list of discopy pregroupwords (sentence being parsed)
        word_index : int
            word index of element that needs to be added
        ty_index : int
            simple type index in word, of element that needs to be added
        backward : Bool
            if False it adds type to bottom of the stack rather than top
        """
        if not backward:
            self.types.append(Ty(string[word_index].cod[ty_index]))
            self.indices.append((word_index, ty_index))
        else:
            self.types.insert(0, Ty(string[word_index].cod[ty_index]))
            self.indices.insert(0, (word_index, ty_index))


class Fan:
    """
    Class to represents a critical segment, i.e. the right endpoints
    of a critical fan.

    Attributes
    ----------
    start_from : pair of int
        word and type indices (coordinates) of the first type after the
        fan. Forward parsing will be resumed from this type onward.
    s : Stack
        initialises a new stack (so-called back stack), where the fan is
        pushed.
    """
    def __init__(self):
        self.start_from = None
        self._stack = Stack()

    def fan2stack(self, string, w_critical, t_critical):
        """
        Reads string after a given critical type and finds its critical fan.
        Then pushes it to an empty Stack. It also updates the attribute
        start_from (the first type after the fan).

        Parameters
        ----------
        string : list
            list of :class:`Word` (sentence)
        w_critical : int
            word index of critical type
        t_index : int
            type index of critical type

        Example
        -------
        >>> fan = Fan()
        >>> A =  Word('A', Ty('n'))
        >>> B = Word('B', Ty('b')@ Ty('n').r @ Ty('s').r)
        >>> C = Word('C', Ty('b').r @ Ty('n'))
        >>> string = [A, B, C]
        >>> fan.fan2stack(string, 1, 1)
        >>> fan._stack.types
        [Ty(Ob('n', z=1)), Ty(Ob('s', z=1)), Ty(Ob('b', z=1))]
        """
        assert len(self._stack.types) == 0, 'stack needs to be empty'

        # function that reads word (or word segment) and
        # checks if its types form a critical fan
        def check_fan(types, ty_index):
            while ty_index < len(types):
                ty = Ty(types[ty_index])
                if critical(ty):
                    ty_index += 1
                else:
                    break
            if ty_index == len(types):
                return 'continue'  # fan potentially continues in the next word

            return ty_index    # fan ends in this word at ty ty_index

        word_index = w_critical
        ty_index = t_critical

        # we apply check_fan to each word until we find end of fan
        # we push found critical segments to an empty stack
        while word_index < len(string):
            word = string[word_index].cod
            if check_fan(word, ty_index) == 'continue':
                self._stack.push_types(string, word_index,
                                       ty_index_start=ty_index)
                word_index += 1
                ty_index = 0
            elif check_fan(word, ty_index) == 0:
                self.start_from = (word_index, check_fan(word, ty_index))
                break
            else:
                self.start_from = (word_index, check_fan(word, ty_index))
                self._stack.push_types(string, word_index,
                                       ty_index_start=ty_index,
                                       ty_index_end=check_fan(word, ty_index))
                break


class BackParser():
    """
    Represents the backparsing part of the pregroup parser algorithm.

    Attributes
    ----------
    f : class Fan
        initialises a new fan object
    s : attribute of Fan
        the new empty stack initialised by the new fan.
        This is the so called backward stack
    bound : int
        the bound of the backparser
    middle_criticals : list
        stores the coordinates of the middle-criticals
        (left left_adjoints of the critical types)
    reductions : dict
        stores the reduction links of critical types - found by backparsing
    start_from : None or pair of int
        we store here the start_from coordinates from Fan()
    """

    def __init__(self, bound):
        # private attributes
        self._bound = bound
        self._fan = Fan()
        self._stack = self._fan._stack
        # outputs
        self.middle_criticals = []
        self.reductions = {}
        self.start_from = None

    def push_fan(self, string, w_critical, t_critical):
        """
        Uses fan2stack to push fan to stack and update start_from cooridinates.

        Parameters
        ----------
        string : list
            list of discopy pregroupwords (sentence)
        w_critical : int
            word index of critical type
        t_index : int
            type index of critical type

        Returns
        -------
        pair of lists
            these are the backward stack, which now contains the fan.
        """

        self._fan.fan2stack(string, w_critical, t_critical)  # fan pushed to Stack
        self.start_from = self._fan.start_from
        logger.info(str(('backparse stack was initialised with critical fan',
                        self._stack.types)))
        logger.info(str(('forward parsing will resume from (word_index, ty_index):',
                        self.start_from)))
        return self._stack.types, self._stack.indices

    def back_parse(self, string, w_critical, top_coordinates):
        """
        Backparser function.
        It will first read forward from a given critical type, to find its
        critical fan.
        Then the fan is pushed onto the backward fan and
        the string is processed backward from the fan.
        The backparsing will stop once the stack is found empty.
        Indeed, this implies that the critical types have been reduced.
        It will also stop if it reaches the coordinates of
        the top of the forward stack
        (those reductions are already checked by forward parser).
        It will also stop, and rise error if it reaches a given bound
        (maximum number of types the backparser is allowed to process).
        The critical reductions are stored in middle_criticals
        and reductions attributes.

        Parameters
        ----------
        string : list
            list of discopy pregroupwords (sentence)
        w_critical : int
            word index of newly found critical type
            (critical type triggers initialisation of BackParser in Parser)
        top_coordinates : pair of int
            coordinates (word_index, ty_index) of top of forward stack

        Returns
        -------
        bool
            True if fan was reduced. False otherwise.
        """

        logger.info('backparsing started...')
        w_top, t_top = top_coordinates
        word_index = w_critical - 1
        counter = 0
        # making sure the parsing is bounded
        # parsing stops when the stack is  found empty
        # (this means we have reduced the critical fan)
        while counter <= self._bound and len(self._stack.types) > 0 and word_index >= w_top:
            word = string[word_index].cod
            ty_index = len(word) - 1
            while ty_index >= 0:
                # making sure we don't exceed top of forward stack
                if word_index == w_top and ty_index == t_top:
                    break

                ty = Ty(word[ty_index])

                if check_reduction(ty, self._stack.types[0]):
                    w_temp, t_temp = self._stack.pop(i=0)

                    # storing new reduction links and
                    # types that need to be added to forward stack
                    if critical(Ty(string[w_temp].cod[t_temp])):
                        middle_critical = (word_index, ty_index)
                        self.middle_criticals.append(middle_critical)
                        self.reductions[(w_temp, t_temp)] = middle_critical

                    if len(self._stack.types) == 0:
                        break

                    ty_index = ty_index - 1
                    counter += 1


                else:
                    self._stack.push_types(string, word_index,
                                      ty_index_end=ty_index, backward=True)
                    word_index = word_index - 1
                    counter += ty_index
                    break

        if len(self._stack.types) > 0:
            logger.info('Underlink bound reached without reducing critical fan. No sentence parsing')
            return False

        logger.info('critical fan successfully reduced')
        return True


class LinPP():
    """
    A class to represent the LinPP parsing algorithm.

    The parsing at each stage updates the stack containing the reduced string,
    and a set of reductions, containings links of reduction pairs.
    The links are identified by pairs of coordinates (word index, type index).
    The parsing proceeds as in Lazy Parsing until a critical type that does
    NOT reduce with the top of the stack in encountered.
    At this point the bounded BackParser is called.
    This will find a critical fan and process the string backward
    until the fan reduces.
    Lazy parsing is resumed from after the critical fan.
    This parser computes a reduction of the string in linear time
    (linearly proportional to the number of simple types in the sentence).
    The coefficient of proportinality given by the bound of the backparser.
    If we the bound number approaches infinity,
    then the algorithm computes in quadratic time.

    Parameters
    ----------
    bound : int, optional
        bound for backward parsing.

    Attributes
    ----------
    _bound : int
        bound for BackParser

    Parameters
    ----------
    bound : int
        bound for backparser. Default is 7.
    """
    def __init__(self, bound=7):
        # private attribute
        self._bound = bound

    def parse(self, string):
        """
        Computes pregroup reduction of the string.
        Given the appropriate restrictions on the dictionary (see theory),
        if it is a sentence it will compute a parsing.
        If string is not a sentence it will reduce to types
        other than the sentence type.


        Parameters
        ----------
        string : list
            list of discopy pregroupwords (sentence)

        Returns
        -------
        reductions : list
            set of reduction links
        stack : list
            reduced sentence
        """

        logger.info('at each stage we will print the updated stack, and the stage coordinates: w_index, t_index')
        logger.info('parsing algorithm started...')

        # initiating stack and reduction set
        reductions = {}
        stack = Stack()

        # pushing types of first word onto stack
        stack.push_types(string, 0)
        logger.info(str(('adding first word to stack:', stack.types)))
        word_index = 1                     # word index
        ty_index = 0                       # simple type index in word
        while word_index < len(string):    # iterating over remaining sentence
            word = string[word_index]
            # iterating over the simple types of each word
            while ty_index <= len(word.cod):
                # if we reach end of the word we are going to the next word
                if ty_index == len(word.cod):
                    word_index += 1
                    ty_index = 0
                    break

                ty = Ty(word.cod[ty_index])
                length = len(stack.types)
                # if stack is empty or type does not reduce with top of stack
                if (length == 0) or (length > 0 and not check_reduction(stack.types[-1], ty)):
                    # if type critical we backparse
                    if critical(ty):
                        logger.info(str(('critical type found at:',
                                         [word_index, ty_index])))
                        back = BackParser(self._bound)
                        fan_types, fan_indices = back.push_fan(string,
                                                               word_index,
                                                               ty_index)

                        if length == 0:
                            # substitute for top of the stack coordinates,
                            # in case of empty stack
                            backparser = back.back_parse(string, word_index,
                                                         (0, -1))
                        elif length > 0:
                            backparser = back.back_parse(string, word_index,
                                                         stack.indices[-1])
                        # if critical fan was reduced by backparser,
                        # we update reduction set and stack according
                        # to the corrected reductions
                        if backparser:
                            logger.info('correcting forward stack and set of reductions')
                            # correcting links in reduction set and
                            # adding removed (from reductions ) types to stack
                            left_criticals = []   # storing removed types
                            for middle_critical in back.middle_criticals:
                                # removing stale links
                                left_criticals.append(reductions.pop(middle_critical))
                            # adding critical links to reduction set
                            reductions.update(back.reductions)
                            # adding removed types to stack
                            # and making sure types appear in right order
                            for left_adjoint in sorted(left_criticals):
                                word_index, ty_index = left_adjoint
                                # pushing stale types onto stack
                                stack.add(string, word_index, ty_index)
                            logger.info(str(('updated stack:',
                                             stack.types)))

                        # if backparser did not reduce the whole critical fan
                        # we add it back on the stack
                        else:
                            logger.info('adding critical fan to forward stack')
                            stack.types += fan_types
                            stack.indices += fan_indices
                            logger.info(str(('updated stack:',
                                             stack.types)))
                        logger.info(str(('resuming forward parsing from stage:',
                                         back.start_from)))
                        # resuming forward parser from after the fan
                        word_index, ty_index = back.start_from
                        break

                    # if linear type we push word to stack
                    else:
                        stack.push_types(string, word_index,
                                         ty_index_start=ty_index)
                        logger.info(str(('stack updated and stage:',
                                         stack.types,
                                         [word_index, ty_index])))
                        word_index += 1
                        ty_index = 0
                        break

                # type reducing with top of the stack
                if length > 0 and check_reduction(stack.types[-1], ty):
                    # we pop stack and update reductions
                    type_pop = stack.pop()
                    logger.info(str(('stack updated and stage:',
                                     stack.types,
                                     [word_index, ty_index])))
                    reductions[(word_index, ty_index)] = type_pop
                    ty_index += 1

        logger.info('parsing completed.')

        # transforming reduction dict into list of pairs (left, right)
        reduction_list = []
        for key, value in reductions.items():
            reduction_list.append([value, key])

        return reduction_list, stack.types


def is_sentence(sentence, bound=7, target=Ty('s')):
    """
    Runs parsing and checks if string reduces to target type.


    Parameters
    ----------
    sentence : list
        list of discopy pregroup words

    target : class Ty from discopy.rigid
        discopy grammatical type. Default is sentence type 's'.

    bound : int
        bound for parser


    Returns
    -------
    Bool
        True is reduced to target type.

        """
    stack = LinPP(bound=bound).parse(sentence)[1]
    return stack == [target]


def coordinates_to_index(sentence, reductions):
    """
    Parser returns reductions links as
    pairs of coordinates [(word_index, type_index), (word_index, type_index)].
    This function transforms the set of reductions into single indices pairs
    (left_index, right_index).

    Parameters
    ----------
    sentence : list
        list of discopy words
    reductions : list
        list of reductions as produced by the parser

    Returns
    -------
    list of single index reduction pairs


    Example
    -------
    >>> n, s = Ty('n'), Ty('s')
    >>> Alice, Bob = Word('Alice', n), Word('Bob', n)
    >>> loves = Word('loves', n.r @ s @ n.l)
    >>> sentence = [Alice, loves, Bob]
    >>> reductions = [[(0, 0), (1, 0)], [(1, 2), (2, 0)]]
    >>> coordinates_to_index(sentence, reductions)
    [[0, 1], [3, 4]]
    """
    red = []
    for link in reductions:
        new_link = []
        for coor in link:
            word_index, type_index = coor
            if word_index == 0:
                new_link.append(type_index)
            else:
                index = len(sentence[0].cod) - 1
                for word in sentence[1:word_index]:
                    index = index + len(word.cod)
                new_link.append(index + type_index + 1)
        red.append(new_link)

    return red

def reduction_layers(reductions):
    """
    This function identifies layers of nested reduction cups.
    It returns a list of the different layers in order of appearance.
    Input needs to be already converted into indices.

    Parameters
    ----------
    reductions : list
        list of single index reduction pairs


    Returns
    -------
    list of layers. Layers are dictionaries.


    Example
    -------
    reductions = [[0, 1], [5, 6], [8, 9], [7, 10], [4, 11], [3, 12], [2, 13], [15, 16]]
    >>> reduction_layers(reductions)
    [{0: 1, 5: 6, 8: 9, 15: 16}, {7: 10}, {4: 11}, {3: 12}, {2: 13}]
    """
    layers = []
    while True:
        layer_1 = {reductions[0][0]: reductions[0][1]}
        layer_2 = []
        counter = 1
        while counter < len(reductions):
            if reductions[counter][0] < reductions[counter -1][0]:
                layer_2.append(reductions[counter])
            if reductions[counter][0] > reductions[counter -1][0]:
                layer_1[reductions[counter][0]] = reductions[counter][1]
            counter += 1
        layers.append(layer_1)
        if layer_2 == []:
            break
        reductions = layer_2
    return layers


def sentence_diagram(sentence, bound_parser=7, reductions=None):
    """
    This function transforms a list of words (unparsed sentence) into
    its sentence diagram. If the sentence was already parsed, we can
    specify a list of reductions as a paramenter.

    """
    # if not parsed yet, we parse the sentence
    if reductions is None:
        parser = LinPP(bound=bound_parser)
        reductions = parser.parse(sentence)[0]

    # converting coordinates into indices and separate layers of cups.
    # this step is working okay
    reductions = coordinates_to_index(sentence, reductions)
    layers = reduction_layers(reductions)

    # constructing diagram of tensored words
    # this step is working okay
    diagram = sentence[0]
    if len(sentence) == 1:
        return diagram
    for word in sentence[1:]:
        diagram = diagram.tensor(word)

    string = diagram[:].cod # tensored sentence we loop over. static
    layer_set = {}
    # constructing diagram of layers of cups
    # TODO: check this loop: issues of mismatching of different layers
    for layer in layers:
        counter = 0
        diagram_temp = Id(Ty()) # initial empty diagram
        while counter < len(string):
            if counter in layer: # index in reduction layer
                # cup of reduction
                cup = Cup(Ty(string[counter]), Ty(string[layer[counter]]))
                diagram_temp = diagram_temp @ cup # add cup
                counter = layer[counter] + 1
            else:
                if counter in layer_set: # if index in previous layers
                    counter += layer_set[counter] + 1
                else:
                    id_diag = Id(Ty(string[counter]))  # add id diagram of type
                    diagram_temp = diagram_temp @ id_diag
                    counter += 1
        diagram = diagram >> diagram_temp
        layer_set.update(layer)
    return diagram
