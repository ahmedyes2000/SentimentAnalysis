class Document(object):
    '''
    This is an abstract document.
    '''
    def getContent(self):
        raise NotImplementedError("Abstract Documents needs to be implemented")