# -*- coding: utf-8 -*-

"""
Implements distributional compositional models.

>>> from discopy.tensor import TensorFunctor
>>> from discopy.rigid import Ty
>>> s, n = Ty('s'), Ty('n')
>>> Alice, Bob = Word('Alice', n), Word('Bob', n)
>>> loves = Word('loves', n.r @ s @ n.l)
>>> grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
>>> sentence = grammar << Alice @ loves @ Bob
>>> ob = {s: 1, n: 2}
>>> ar = {Alice: [1, 0], loves: [0, 1, 1, 0], Bob: [0, 1]}
>>> F = TensorFunctor(ob, ar)
>>> assert F(sentence) == True

>>> from discopy.quantum import qubit, Ket, CX, H, X, sqrt, CircuitFunctor
>>> from discopy.rigid import Ty
>>> s, n = Ty('s'), Ty('n')
>>> Alice = Word('Alice', n)
>>> loves = Word('loves', n.r @ s @ n.l)
>>> Bob = Word('Bob', n)
>>> grammar = Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
>>> sentence = grammar << Alice @ loves @ Bob
>>> ob = {s: Ty(), n: qubit}
>>> ar = {Alice: Ket(0),
...       loves: CX << sqrt(2) @ H @ X << Ket(0, 0),
...       Bob: Ket(1)}
>>> F = CircuitFunctor(ob, ar)
>>> assert abs(F(sentence).eval().array) ** 2
"""

import random
import re

from discopy import messages, drawing, biclosed, rigid
from discopy.cat import AxiomError
from discopy.monoidal import Ty, Box, Diagram, Id
from discopy.rigid import Cup, Ob


class Word(Box):
    """
    Implements words as boxes with a :class:`moncat.Ty` as codomain.

    >>> from discopy.rigid import Ty
    >>> Alice = Word('Alice', Ty('n'))
    >>> loves = Word('loves',
    ...     Ty('n').r @ Ty('s') @ Ty('n').l)
    >>> Alice
    Word('Alice', Ty('n'))
    >>> loves
    Word('loves', Ty(Ob('n', z=1), 's', Ob('n', z=-1)))
    """
    def __init__(self, name, cod, dom=Ty(), data=None, _dagger=False):
        if not isinstance(name, str):
            raise TypeError(messages.type_err(str, name))
        if not isinstance(dom, Ty):
            raise TypeError(messages.type_err(Ty, dom))
        if not isinstance(cod, Ty):
            raise TypeError(messages.type_err(Ty, cod))
        super().__init__(name, dom, cod, data, _dagger)

    def __repr__(self):
        return "Word({}, {}{})".format(
            repr(self.name), repr(self.cod),
            ", dom={}".format(repr(self.dom)) if self.dom else "")


class CCGWord(Word, biclosed.Box):
    """ Word with a :class:`biclosed.Ty` as codomain. """


class PregroupWord(Word, rigid.Box):
    """ Word with a :class:`rigid.Ty` as codomain. """


class CFG:
    """
    Implements context-free grammars.

    >>> s, n, v, vp = Ty('S'), Ty('N'), Ty('V'), Ty('VP')
    >>> R0, R1 = Box('R0', vp @ n, s), Box('R1', n @ v , vp)
    >>> Jane, loves = Word('Jane', n), Word('loves', v)
    >>> cfg = CFG(R0, R1, Jane, loves)
    >>> gen = cfg.generate(start=s, max_sentences=2, max_depth=6)
    >>> for sentence in gen: print(sentence)
    Jane >> loves @ Id(N) >> Jane @ Id(V @ N) >> R1 @ Id(N) >> R0
    Jane >> loves @ Id(N) >> Jane @ Id(V @ N) >> R1 @ Id(N) >> R0
    >>> gen = cfg.generate(
    ...     start=s, max_sentences=2, max_depth=6,
    ...     remove_duplicates=True, max_iter=10)
    >>> for sentence in gen: print(sentence)
    Jane >> loves @ Id(N) >> Jane @ Id(V @ N) >> R1 @ Id(N) >> R0
    """
    def __init__(self, *productions):
        self._productions = productions

    @property
    def productions(self):
        """
        Production rules, i.e. boxes with :class:`moncat.Ty` as dom and cod.
        """
        return self._productions

    def __repr__(self):
        return "CFG{}".format(repr(self._productions))

    def generate(self, start, max_sentences, max_depth, max_iter=100,
                 remove_duplicates=False, not_twice=[], seed=None):
        """
        Generate sentences from a context-free grammar.
        Assumes the only terminal symbol is :code:`Ty()`.

        Parameters
        ----------

        start : type
            root of the generated trees.
        max_sentences : int
            maximum number of sentences to generate.
        max_depth : int
            maximum depth of the trees.
        max_iter : int
            maximum number of iterations, set to 100 by default.
        remove_duplicates : bool
            if set to True only distinct syntax trees will be generated.
        not_twice : list
            list of productions that you don't want appearing twice
            in a sentence, set to the empty list by default
        """
        if seed is not None:
            random.seed(seed)
        prods, cache = list(self.productions), set()
        n, i = 0, 0
        while (not max_sentences or n < max_sentences) and i < max_iter:
            i += 1
            sentence = Id(start)
            depth = 0
            while depth < max_depth:
                recall = depth
                if sentence.dom == Ty():
                    if remove_duplicates and sentence in cache:
                        break
                    yield sentence
                    if remove_duplicates:
                        cache.add(sentence)
                    n += 1
                    break
                tag = sentence.dom[0]
                random.shuffle(prods)
                for prod in prods:
                    if prod in not_twice and prod in sentence.boxes:
                        continue
                    if Ty(tag) == prod.cod:
                        sentence = sentence << prod @ Id(sentence.dom[1:])
                        depth += 1
                        break
                if recall == depth:  # in this case, no production was found
                    break


def eager_parse(*words, target=Ty('s')):
    """
    Tries to parse a given list of words in an eager fashion.
    """
    result = Id(rigid.Ty()).tensor(*words)
    scan = result.cod
    while True:
        fail = True
        for i in range(len(scan) - 1):
            if scan[i: i + 1].r != scan[i + 1: i + 2]:
                continue
            cup = Cup(scan[i: i + 1], scan[i + 1: i + 2])
            result = result >> Id(scan[: i]) @ cup @ Id(scan[i + 2:])
            scan, fail = result.cod, False
            break
        if result.cod == target:
            return result
        if fail:
            raise NotImplementedError


def brute_force(*vocab, target=Ty('s')):
    """
    Given a vocabulary, search for grammatical sentences.
    """
    test = [()]
    for words in test:
        for word in vocab:
            try:
                yield eager_parse(*(words + (word, )), target=target)
            except NotImplementedError:
                pass
            test.append(words + (word, ))


def draw(diagram, **params):
    """
    Draws a pregroup diagram, i.e. of shape :code:`word @ ... @ word >> cups`.

    Parameters
    ----------
    width : float, optional
        Width of the word triangles, default is :code:`2.0`.
    space : float, optional
        Space between word triangles, default is :code:`0.5`.
    textpad : pair of floats, optional
        Padding between text and wires, default is :code:`(0.1, 0.2)`.
    draw_types : bool, optional
        Whether to draw type labels, default is :code:`True`.
    aspect : string, optional
        Aspect ratio, one of :code:`['equal', 'auto']`.
    margins : tuple, optional
        Margins, default is :code:`(0.05, 0.05)`.
    fontsize : int, optional
        Font size for the words, default is :code:`12`.
    fontsize_types : int, optional
        Font size for the types, default is :code:`12`.
    figsize : tuple, optional
        Figure size.
    path : str, optional
        Where to save the image, if `None` we call :code:`plt.show()`.

    Raises
    ------
    ValueError
        Whenever the input is not a pregroup diagram.
    """
    if not isinstance(diagram, Diagram):
        raise TypeError(messages.type_err(Diagram, diagram))
    words, is_pregroup = rigid.Id(rigid.Ty()), True
    for left, box, right in diagram.layers:
        if isinstance(box, Word):
            if right:  # word boxes should be tensored left to right.
                is_pregroup = False
                break
            words = words @ box
        else:
            break
    cups = diagram[len(words):].foliation().boxes\
        if len(words) < len(diagram) else []
    is_pregroup = is_pregroup and words and all(
        isinstance(box, Cup) for s in cups for box in s.boxes)
    if not is_pregroup:
        raise ValueError(messages.expected_pregroup())
    drawing.pregroup_draw(words, cups, **params)


def cat2ty(string):
    """ Takes the string repr of a CCG category, returns a biclosed.Ty """
    def unbracket(string):
        return string[1:-1] if string[0] == '(' else string

    def remove_modifier(string):
        return re.sub(r'\[[^]]*\]', '', string)

    def split(string):
        par_count = 0
        for i, char in enumerate(string):
            if char == "(":
                par_count += 1
            elif char == ")":
                par_count -= 1
            elif char in ["\\", "/"] and par_count == 0:
                return unbracket(string[:i]), char, unbracket(string[i + 1:])
        return remove_modifier(string), None, None

    left, slash, right = split(string)
    if slash == '\\':
        return biclosed.Under(cat2ty(right), cat2ty(left))
    if slash == '/':
        return biclosed.Over(cat2ty(left), cat2ty(right))
    return biclosed.Ty(left)


def tree2diagram(tree, dom=biclosed.Ty()):
    """ Takes a depccg.Tree in JSON format, returns a biclosed.Diagram """
    if 'word' in tree:
        return CCGWord(tree['word'], cat2ty(tree['cat']), dom=dom)
    children = list(map(tree2diagram, tree['children']))
    dom = biclosed.Ty().tensor(*[child.cod for child in children])
    cod = cat2ty(tree['cat'])
    if tree['type'] == 'ba':
        box = biclosed.BA(dom[:1], dom[1:])
    elif tree['type'] == 'fa':
        box = biclosed.FA(dom[:1], dom[1:])
    else:
        box = biclosed.Box(tree['type'], dom, cod)
    return biclosed.Id(biclosed.Ty()).tensor(*children) >> box


####################################################### LINEAR PARSER ############################################################


##################################### necessary functions #######################################
def check_reduction(t1,t2):
    """
    Checks whether simple types t1,t2 reduce to the empty type.


    Parameters
    ----------
    t1 : Ty class from discopy
        left type
    t2 : Ty class from discopy
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

    if (t1 == t2.l) or (t2 == t1.r):
        return True
    else:
        return False




def critical(t):
    """
    Checks if type is critical (assuming types in dictionary satisfy z<=1).


    Paramenters
    -----------
    t = Ty class from discopy


    Example
    -------
    >>> critical(Ty('n').r)
    True
    >>> critical(Ty('n').l)
    False
    >>> critical(Ty('n'))
    False
    """


    if t[0].z > 0:
        return True
    else:
        return False

################################################## necessary classes : Stack, Fan, BackParser ###################

class Stack():

    def __init__(self):
        """
        A class to represent the stack of reduced types for a string being parsed.

        ...

        Attributes
        ----------
        types : list
            the stack containing the reduced types
        indices : list
            the stack containing the indices corresponding to the reduced types in the parsed string


        Methods
        -------
        push_types(string, word_index, t1=None, t2=None, backward=False)
            adds word segments from the string to the stack

        add(string, w, t, backward=False)
            adds one type to from the string to the stack

        pop(i=-1)
            pops type from stack
        """



        self.types = []
        self.indices = []

    def push_types(self, string, word_index, t1=None, t2=None, backward=False):
        """
        Adding segments of words from the string to the stack (both types and corresponding indices).


        Parameters
        ----------
        string : list
            list of discopy words (i.e. a sentence)
        word_index : int
            index of word in string that we want to push
        t1 : None or int
            index of first simple type in word we want to push. If None, we push from the beginning of the word
        t2 : None or int
           index of first simple type in word we do NOT want to push. If None, we push until the end of the word
        backward : Bool
            if True, will push types at the bottom of the stack rather than at the top.


        Example
        -------
        >>> stack = Stack()
        >>> Alice = Word('Alice', Ty('n'))
        >>> loves = Word('loves', Ty('n').r @ Ty('s') @ Ty('n').l)
        >>> string = [Alice, loves]
        >>> stack.push_types(string, 1, t1=1)
        >>> stack.types
        [Ty('s'), Ty(Ob('n', z=-1))]

        >>> stack.push_types(string,0, backward=True)
        >>> stack.types
        [Ty('n'), Ty('s'), Ty(Ob('n', z=-1))]
        """



        if t1== None:
            t = 0
        else:
            t = t1

        if t2 == None:
            tmin = len(string[word_index].cod)
        else:
            tmin = t2

        while t < tmin:
            if backward==False:
                   self.types.append(Ty(string[word_index].cod[t]))
                   self.indices.append((word_index,t))
            else:
                   self.types.insert(0, Ty(string[word_index].cod[t]))
                   self.indices.insert(0, (word_index, t))
            t+=1

    def pop(self,i=-1):
        """
        Pops a type out of stack in position i.


        Parameters
        -----------
        i = int
            indicates the index of the type we want to pop out (default i=-1, i.e. the top of the stack)


        Returns
        -------
        pair of integers (w,t). w is the word index in the string, and t is the type index  in the string of the popped element.
        """
        self.types.pop(i)
        coordinates= self.indices.pop(i)
        return coordinates

    def add(self, string, w, t, backward=False):
        """
        Adds one type to the stack from the string


        Parameters
        ----------
        string : list
            list of discopy words (sentence being parsed)
        w : int
            word index of element that needs to be added
        t : int
            simple type index in word, of element that needs to be added
        backward : Bool
            if False it adds type to bottom of the stack rather than top
        """

        if not backward:
            self.types.append(Ty(string[w].cod[t]))
            self.indices.append((w,t))
        else:
            self.types.insert(0, Ty(string[word_index].cod[t]))
            self.indices.insert(0, (word_index, t))







class Fan():

    def __init__(self):
        """
        Class to represents a critical segment, i.e. the right endpoints of a critical fan.


        Attributes
        ----------
        start_from : pair of int
            word and type indices (coordinates) of the first type after the fan. Forward
            parsing will be resumed from this type onward.
        s = class Stack
            initialises a new stack (so-called back stack), where the fan is pushed.


        Methods
        -------
        fan2stack(string, w_critical, t_critical)
            finds the fan, reading from the newly found critical type onwards. Then pushes it to the back Stack
        """

        self.start_from = None
        self.s = Stack()




    def fan2stack(self, string, w_critical, t_critical):
        """
        Reads string after a given critical type and finds its critical fan. Then pushes it to an empty Stack.
        It also updates the attribute start_from (the first type after the fan).

        Parameters
        ----------
        string : list
            list of discopy words (sentence)
        w_critical : int
            word index of critical type
        t_index : int
            type index of critical type


        Example
        -------
        >>> fan = Fan()
        >>> A, B, C = Word('A', Ty('n')), Word('B', Ty('b')@Ty('n').r @ Ty('s').r), Word('c', Ty('b').r @ Ty('n'))
        >>> string = [A,B,C]
        >>> fan.fan2stack(string, 1,1)
        >>> fan.s.types
        [Ty(Ob('n', z=1)), Ty(Ob('s', z=1)), Ty(Ob('b', z=1))]
        """

        assert len(self.s.types)==0, 'stack needs to be empty'

        #function that reads word (or word segment) and checks if its types form a critical fan
        def check_fan(Types,t):
            while t < len(Types):
                type = Ty(Types[t])
                if critical(type):
                    t+=1
                else:
                    break
            if t == len(Types):
                return 'continue' #the fan potentially continues in the next word
            else:
                return t          #the fan ends in this word at type t



        w = w_critical
        t = t_critical

        #we apply check_fan to each word until we find end of fan
        #we push found critical segments to an empty stack
        while w < len(string):

                word = string[w].cod
                c = check_fan(word,t)
                if c == 'continue':
                    self.s.push_types(string, w, t1=t)
                    w+=1
                    t=0
                elif c == 0:
                    self.start_from = (w,c)
                    break
                else:
                    self.start_from = (w,c)
                    self.s.push_types(string, w, t1=t, t2=c)
                    break







class BackParser():

    def __init__(self, K):
        """
        A class to represent the backparsing part of the pregroup parser algorithm.

        Attributes
        ----------
        f : class Fan
            initialises a new fan object
        s : attribute of Fan
            the new empty stack initialised by the new fan. This is the so called backward stack
        K : int
            the bound of the backparser
        middle_criticals : list
            stores the coordinates of the middle-criticals (left left_adjoints of the critical types)
        reductions : dict
            stores the reduction links of the critical types, found by backparsing
        start_from : None or pair of int
            we store here the start_from coordinates from Fan()


        Methods
        -------
        push_fan(string, w_critical, t_critical)
            uses fan2stack to push fan to stack and update start_from cooridinates

        back_parse(string, w_critical, t_critical, top_coordinates)
            backward parsing, from fan to the coordinates of the top of the forward stack (or the bound if it is reached before)
        """




        self.f = Fan()
        self.s= self.f.s
        self.K = K

        #outputs
        self.middle_criticals = []
        self.reductions = {}
        self.start_from = None

    def push_fan(self,string,w_critical, t_critical):
        """
        Uses fan2stack to push fan to stack and update start_from cooridinates.

        Parameters
        ----------
        string : list
            list of discopy words (sentence)
        w_critical : int
            word index of critical type
        t_index : int
            type index of critical type

        Returns
        -------
        pair of lists
            these are the backward stack, which now contains the fan.
        """

        self.f.fan2stack(string,w_critical, t_critical) #fan pushed to Stack
        self.start_from = self.f.start_from
        print('backparse stack was initialised with critical fan', self.s.types)
        print('forward parsing will resume from (w,t):', self.start_from)
        return self.s.types , self.s.indices


    def back_parse(self, string, w_critical, t_critical, top_coordinates):
        """
        Backparser function. It will first read forward from a given critical type, to find its
        critical fan. Then the fan is pushed onto the backward fan and the string is processed backward from the fan.
        The backparsing will stop once the stack is found empty. Indeed, this implies that the critical types have been reduced.
        It will also stop if it reaches the coordinates of the top of the forward stack (those reductions are already checked by forward parser).
        It will also stop, and rise error if it reaches a given bound K (maximum number of types the backparser is allowed to process, excluding the fan).
        The critical reductions are stored in middle_criticals and reductions attributes.

        Parameters
        ----------
        string : list
            list of discopy words (sentence)
        w_critical : int
            word index of newly found critical type (critical type triggers initialisation of BackParser in Parser)
        t_critical : int
            type index of critical type
        top_coordinates : pair of int
            coordinates (w,t) of top of forward stack

        Returns
        -------
        bool
            True if fan was reduced. False otherwise.
        """



        print('backparsing started...')

        w_top, t_top = top_coordinates
        w= w_critical -1
        counter = 0
        while counter <= self.K and len(self.s.types)> 0 and w >= w_top: #making sure the parsing is bound by K
                                                                         #parsing stops when the stack is  found empty
                                                                         #(this means we have reduced the critical fan)

            word = string[w].cod
            t = len(word)-1
            while t >= 0:
                if w == w_top and t == t_top: #making sure we don't exceed top of forward stack (not included)
                        break
                else:
                        type = Ty(word[t])
                        if check_reduction(type, self.s.types[0]):
                            W,T = self.s.pop(i=0)

                            if critical(Ty(string[W].cod[T])):
                                middle_critical = (w,t)
                                self.middle_criticals.append(middle_critical)
                                self.reductions[(W,T)]=middle_critical
                                t = t-1
                                counter += 1
                        else:
                                self.s.push_types(string, w, t2=t, backward=True)
                                w = w-1
                                counter += t
                                break

        if len(self.s.types) > 0:
            print('Underlink bound reached without reducing critical fan. No sentence parsing')
            return False
        else:
            print('critical fan successfully reduced')
            return True



################################################# main class: Parser LinPP_W ########################################################



class LinPP_W():

    def __init__(self, K = 7):          #K = bound for backward parsing
        """
        A class to represent the LinPP_W parsing algorithm.

        The parsing at each stage updates the stack containing the reduced string,
        and a set of reductions, containings links of reduction pairs.
        The links are identified by pairs of coordinates (word index and type index).
        The parsing proceeds as in Lazy Parsing until a critical type that does
        NOT reduce with the top of the stack in encountered.
        At this point the bounded BackParser is called. This will find a critical fan and process the string backward until the fan reduces.
        Lazy parsing is resumed from after the critical fan.
        This parser computes a reduction of the string in linear time (linearly proportional to the number of simple types in the sentence).
        The coefficient of proportinality given by the bound of the backparser.
        If we the bound number approaches infinity, then the algorithm computes in quadratic time.

        Attributes
        ----------
        stack : class Stack
            forward stack
        back : class BackParser
            backparsing algorithm
        R : dict
            set of reductions. The keys are the right endpoints of the links, the values are the left endpoints.


        Methods
        -------
        parse(string)
            computes pregroup reduction of the string. Given the appropriate restrictions on the dictionary (see theory),
            if it is a sentence it will compute a parsing.
            If string is not a sentence it will reduce to types other than the sentence type.
        is_sentence()
            checks if string was parsed to sentence type or not.


        Parameters
        ----------
        K : int
            bound for backparser. Default is 7.
        """




        self.stack = Stack()        #initialise stack
        self.back = BackParser(K)   #initialise bounded backparser
        self.R = {}                 #initialise set of reductions as dictionary


    def parse(self, string):
        """
        Computes pregroup reduction of the string. Given the appropriate restrictions on the dictionary (see theory),
        if it is a sentence it will compute a parsing.
        If string is not a sentence it will reduce to types other than the sentence type.


        Parameters
        ----------
        string : list
            list of discopy words (sentence)

        Returns
        -------
        R : dict
            set of reduction links
        """

        print('at each stage we will print the updated stack, and the stage coordinates: w_index, t_index')
        print('parsing algorithm started...')

        self.stack.push_types(string, 0) #pushing types of first word onto stack
        print('adding first word to stack:', self.stack.types)
        w = 1                          #word index
        t = 0                          #simple type index in word
        while w < len(string):         #iterating over remaining sentence
            word = string[w]


            while t <=len(word.cod):    #iterating over the simple types of each word
                if t == len(word.cod):
                    w+=1
                    t=0                 #going to next word
                    break
                else:
                    type = Ty(word.cod[t])
                    if len(self.stack.types) == 0:                          #if stack is empty, we push type to stack
                        self.stack.push_types(string, w, t1=t)
                        print('stack updated:', self.stack.types, 'stage:', w,t)
                        w+=1
                        t=0
                        break
                    else:                                                   #else we check reductions

                        if check_reduction(self.stack.types[-1], type):     #if type reduces with top(stack)
                            type_pop= self.stack.pop()                      #popping stack
                            print('stack updated:', self.stack.types, 'stage:', w, t)
                            self.R[(w,t)] = type_pop                        #adding link to reduction set
                            t+=1

                        elif not critical(type):                         #if no reduction, but type is linear
                            self.stack.push_types(string, w, t1=t)       #add the rest of the word to the stack
                            print('stack updated:', self.stack.types, 'stage:', w,t)
                            w +=1
                            t=0
                            break

                        else:
                            print('critical type found at:', w,t, 'backparser called')
                            fan_types, fan_indices = self.back.push_fan(string,w,t)
                            backparser = self.back.back_parse(string, w, t, self.stack.indices[-1]) #backparsing


                            #if critical fan was reduced by backparser, we update reduction set and stack according to the corrected reductions
                            if backparser:
                                print('correcting forward stack and set of reductions')
                                #correcting links in reduction set and adding removed (from self.R) types to stack
                                left_criticals = []                                    #storing removed types
                                for middle_critical in self.back.middle_criticals:
                                    left_criticals.append(self.R.pop(middle_critical)) #removing stale links

                                self.R.update(self.back.reductions)                    #adding critical links (found by backparser) to reduction set


                                #adding removed types to stack
                                for left_adjoint in sorted(left_criticals): #making sure types appear in right order
                                    w,t = left_adjoint
                                    self.stack.add(string, w, t)                #pushing stale types onto stack
                                print('updated stack:', self.stack.types, 'stage:', fan_indices[-1])

                            #if backparser did not reduce the whole critical fan, we add it back on the stack
                            else:
                                print('adding critical fan to forward stack')
                                self.stack.types += fan_types
                                self.stack.indices += fan_indices
                                print('updated stack:', self.stack.types, 'stage:', fan_indices[-1])


                            print('resuming forward parsing from stage:', self.back.start_from)
                            w, t =  self.back.start_from              #resuming forward parser from after the fan
                            break




        print('parsing completed.')
        return self.R



    def is_sentence(self, target=Ty('s')):
        """
        Checks if string reduces to target type. Must be ran after running function parse,
        otherwise it will rise error.

        Parameters
        ----------
        target : class Ty
            discopy grammatical type. Default is sentence type Ty('s')

        Returns
        -------
        Bool
            True is reduced to target type.

        None
            if string not parsed yet

        """




        if self.stack.types == [target]:
            print('YES, this string was parsed to sentence type')
            return True

        elif self.stack.types == []:
            print('string not parsed yet. Please call parse function')
            return None
        else:
            print('NOT a sentence, string parsed to type:', self.stack.types)
            return False



########################################## callable function to parse string ##########################

def get_parsing(sentence, k = 7, target= Ty('s')):
    """
    Main function that runs the parsing algorithm and checks if parsed to sentence type.

    Parameters
    ---------
    sentence : list
        list of discopy words
    k : int
        bound of backparser, that makes algortithm linear. Default is 7.

    target : class Ty
        target type for parsing. Default is sentence type Ty('s').

    Return
    ------
    Bool
        True is parsed to sentence (target) type. False otherwise


    Example
    -------

    """

    bounded_parser = LinPP_W(K=k)
    print('parsing', sentence, 'with bound', k)
    reductions = bounded_parser.parse(sentence)

    print('checking if string was reduced to type', target, '...')
    answer = bounded_parser.is_sentence(target)
    print('returning: sentence status, reduction set')
    return answer, reductions
