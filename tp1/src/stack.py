# ----------------------------------------------------------------
# Authors: Mathieu Kabore, Florence Gaborit and Reph D. Mombrun
# Date: 11/09/2018
# Last update: 12/09/2018
# INF8215 TP1
# ----------------------------------------------------------------

# TODO: May be we should remove this reference, I am not sure it's necessary since it's basic
# Mini implementation of a stack data structure in python
# https://docs.python.org/3.1/tutorial/datastructures.html
# https://stackoverflow.com/questions/4688859/stack-data-structure-in-python
# http://openbookproject.net/thinkcs/python/english3e/stacks.html


class Stack:
    """
    Represents a custom stack build with a list.
    """

    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def is_empty(self):
        return self.items == []
