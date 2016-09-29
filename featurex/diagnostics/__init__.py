__all__ = ['collinearity', 'misc', 'plots']

class Diagnostic(object):

    ''' Based class for diagnostics '''
    def __init__(self, flag=None):
        self.flag = flag

    def flag(self, point):
        return self.flag(point)

    def apply(self, data):
        return self._apply(data)