"""Scratch module used to dry-run the style reviewer, reverted right after."""


class Snake:
    """A snake that will be yanked straight by the reviewer."""

    def __init__(self, name):
        # store the name on a private attribute
        self._name = name  # TODO: rename this attribute

    def get_name(self):
        # return the private name attribute
        return self._name

    def normalise(self, xss):
        # loop over the lists of lists and keep the truthy entries
        result = []
        for xs in xss:
            for x in xs:
                if x is not None:
                    if x:
                        result.append(x)
        return result

    def normalize(self, xss):
        # same as normalise but spelled with a z
        result = []
        for xs in xss:
            for x in xs:
                if x is not None:
                    if x:
                        result.append(x)
        return result
