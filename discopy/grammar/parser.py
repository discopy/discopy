"""
Implementing linear parser LinPP for pregroups.
The algorithm parses lists of Word objects (i.e. Words with codomain
given by rigid.Ty types, Ty)
The parser is correct for lists of words satosfying the following restrictions:

1 - winding number for any simple type must be z <= 1. (right-lateral
complexity 2)
2 - any critical fan is 'guarded', i.e. critical fans do not nest.

"""

import logging

from discopy import *
from discopy.grammar import *


logger = logging.getLogger()

def check_reduction(t1, t2):
    """
    Checks whether simple types t1,t2 reduce to the empty type.


    Parameters
    ----------
    t1 : Ty class from discopy.rigid
        left type
    t2 : Ty class from discopy.rigid
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
    return (t1 == t2.l) or (t2 == t1.r)


def critical(t):
    """
    Checks if type is critical (assuming types in dictionary satisfy z<=1).


    Paramenters
    -----------
    t = Ty class from discopy.rigid


    Example
    -------
    >>> critical(Ty('n').r)
    True
    >>> critical(Ty('n').l)
    False
    >>> critical(Ty('n'))
    False
    """
    return t.z > 0


class Stack:
    def __init__(self):
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
        Adding segments of words from the string to the stack (both types and
        corresponding indices).


        Parameters
        ----------
        string : list
            list of discopy Words (i.e. a sentence)
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
        coordinates = self.indices.pop(i)
        return coordinates

    def add(self, string, w, t, backward=False):
        """
        Adds one type to the stack from the string


        Parameters
        ----------
        string : list
            list of discopy pregroupwords (sentence being parsed)
        w : int
            word index of element that needs to be added
        t : int
            simple type index in word, of element that needs to be added
        backward : Bool
            if False it adds type to bottom of the stack rather than top
        """

        if not backward:
            self.types.append(Ty(string[w].cod[t]))
            self.indices.append((w, t))
        else:
            self.types.insert(0, Ty(string[word_index].cod[t]))
            self.indices.insert(0, (word_index, t))


class Fan:
    def __init__(self):
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

        self.start_from = None
        self.s = Stack()

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
        >>> A, B, C = Word('A', Ty('n')), Word('B', Ty('b')@ Ty('n').r @ Ty('s').r), Word('c', Ty('b').r @ Ty('n'))
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

        self.f.fan2stack(string,w_critical, t_critical) #fan pushed to Stack
        self.start_from = self.f.start_from
        logger.info(str('backparse stack was initialised with critical fan', self.s.types))
        logger.info(str('forward parsing will resume from (w,t):', self.start_from))
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
            list of discopy pregroupwords (sentence)
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



        logger.info('backparsing started...')

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
            logger.info('Underlink bound reached without reducing critical fan. No sentence parsing')
            return False
        else:
            logger.info('critical fan successfully reduced')
            return True


class LinPP():
    """
    A class to represent the LinPP parsing algorithm.

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

    Parameters
    ----------
    K : int, optional
        bound for backward parsing.

    Attributes
    ----------
    stack : class Stack
        forward stack
    back : class BackParser
        backparsing algorithm
    R : dict
        set of reductions. The keys are the right endpoints of the links, the values are the left endpoints.

    Parameters
    ----------
    K : int
        bound for backparser. Default is 7.
    """
    def __init__(self, K=7):
        self.stack = Stack()
        self.back = BackParser(K)
        self.R = {}

    def parse(self, string):
        """
        Computes pregroup reduction of the string. Given the appropriate restrictions on the dictionary (see theory),
        if it is a sentence it will compute a parsing.
        If string is not a sentence it will reduce to types other than the sentence type.


        Parameters
        ----------
        string : list
            list of discopy pregroupwords (sentence)

        Returns
        -------
        R : dict
            set of reduction links
        """

        logger.info('at each stage we will print the updated stack, and the stage coordinates: w_index, t_index')
        logger.info('parsing algorithm started...')

        self.stack.push_types(string, 0) #pushing types of first word onto stack
        logger.info(str(('adding first word to stack:', self.stack.types)))
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
                        logger.info(str(('stack updated:', self.stack.types, 'stage:', w,t)))
                        w+=1
                        t=0
                        break
                    else:                                                   #else we check reductions

                        if check_reduction(self.stack.types[-1], type):     #if type reduces with top(stack)
                            type_pop= self.stack.pop()                      #popping stack
                            logger.info(str(('stack updated:', self.stack.types, 'stage:', w, t)))
                            self.R[(w,t)] = type_pop                        #adding link to reduction set
                            t+=1

                        elif not critical(type):                         #if no reduction, but type is linear
                            self.stack.push_types(string, w, t1=t)       #add the rest of the word to the stack
                            logger.info(str('stack updated:', self.stack.types, 'stage:', w,t))
                            w +=1
                            t=0
                            break

                        else:
                            logger.info(str('critical type found at:', w,t, 'backparser called'))
                            fan_types, fan_indices = self.back.push_fan(string,w,t)
                            backparser = self.back.back_parse(string, w, t, self.stack.indices[-1]) #backparsing


                            #if critical fan was reduced by backparser, we update reduction set and stack according to the corrected reductions
                            if backparser:
                                logger.info('correcting forward stack and set of reductions')
                                #correcting links in reduction set and adding removed (from self.R) types to stack
                                left_criticals = []                                    #storing removed types
                                for middle_critical in self.back.middle_criticals:
                                    left_criticals.append(self.R.pop(middle_critical)) #removing stale links

                                self.R.update(self.back.reductions)                    #adding critical links (found by backparser) to reduction set


                                #adding removed types to stack
                                for left_adjoint in sorted(left_criticals): #making sure types appear in right order
                                    w,t = left_adjoint
                                    self.stack.add(string, w, t)                #pushing stale types onto stack
                                logger.info(str(('updated stack:', self.stack.types, 'stage:', fan_indices[-1])))

                            #if backparser did not reduce the whole critical fan, we add it back on the stack
                            else:
                                logger.info('adding critical fan to forward stack')
                                self.stack.types += fan_types
                                self.stack.indices += fan_indices
                                logger.info(str(('updated stack:', self.stack.types, 'stage:', fan_indices[-1])))


                            logger.info(str(('resuming forward parsing from stage:', self.back.start_from)))
                            w, t =  self.back.start_from              #resuming forward parser from after the fan
                            break




        logger.info('parsing completed.')
        return self.R



    def is_sentence(self, target=Ty('s')):
        """
        Checks if string reduces to target type. Must be ran after running function parse,
        otherwise it will rise error.

        Parameters
        ----------
        target : class Ty from discopy.rigid
            discopy grammatical type. Default is sentence type 's'.

        Returns
        -------
        Bool
            True is reduced to target type.

        None
            if string not parsed yet

        """




        if self.stack.types == [target]:
            logger.info('YES, this string was parsed to sentence type')
            return True

        elif self.stack.types == []:
            logger.info('string not parsed yet. Please call parse function')
            return None
        else:
            logger.info(str('NOT a sentence, string parsed to type:', self.stack.types))
            return False





def get_parsing(sentence, k = 7, target= Ty('s')):
    """
    Main function that runs the parsing algorithm and checks if parsed to sentence type.

    Parameters
    ---------
    sentence : list
        list of discopy pregroupwords
    k : int
        bound of backparser, that makes algortithm linear. Default is 7.

    target : class Ty from discopy.rigid
        target type for parsing. Default is sentence type 's'.

    Return
    ------
    Bool
        True is parsed to sentence (target) type. False otherwise


    Examples
    --------
    >>> s, n = Ty('s'), Ty('n')
    >>> Alice, Bob = Word("Alice", n), Word("Bob", n)
    >>> loves = Word("loves", n.r @ s @ n.l)
    >>> get_parsing([Alice, loves, Bob])
    (True, {(1, 0): (0, 0), (2, 0): (1, 2)})
    """
    bounded_parser = LinPP(K=k)
    logger.info(str(('parsing', sentence, 'with bound', k)))
    reductions = bounded_parser.parse(sentence)

    logger.info('checking if string was reduced to type', target, '...')
    answer = bounded_parser.is_sentence(target)
    logger.info('returning: sentence status, reduction set')
    return answer, reductions
