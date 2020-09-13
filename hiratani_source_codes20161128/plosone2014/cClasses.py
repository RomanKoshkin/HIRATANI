from ctypes import *
from numpy.ctypeslib import ndpointer

lib = cdll.LoadLibrary('./bmm.dylib')

class Params(Structure):
    _fields_ = [("alpha", c_double),
                ("usd", c_double),
                ("JEI", c_double),
                ("ita", c_int),
                ("itb", c_int),
                ("T", c_double),
                ("h", c_double),
                ("NE", c_int),
                ("NI", c_int),
                ("cEE", c_double),
                ("cIE", c_double),
                ("cEI", c_double),
                ("cII", c_double),
                ("JEE", c_double),
                ("JEEinit", c_double),
                ("JIE", c_double),
                ("JII", c_double),
                ("JEEh", c_double),
                ("sigJ", c_double),
                ("Jtmax", c_double),
                ("Jtmin", c_double),
                ("hE", c_double),
                ("hI", c_double),
                ("IEex", c_double),
                ("IIex", c_double),
                ("mex", c_double),
                ("sigex", c_double),
                ("tmE", c_double),
                ("tmI", c_double),
                ("trec", c_double),
                ("Jepsilon", c_double),
                ("tpp", c_double),
                ("tpd", c_double),
                ("twnd", c_double),
                ("g", c_double),
                ("itauh", c_int),
                ("hsd", c_double),
                ("hh", c_double),
                ("Ip", c_double),
                ("a", c_double),
                ("xEinit", c_double),
                ("xIinit", c_double),
                ("tinit", c_double),
                ("tdur", c_double)]

class cClassOne(object):
       
    # we have to specify the types of arguments and outputs of each function in the c++ class imported
    # the C types must match.

    def __init__(self, N):

        self.params_c_obj = Params()

        lib.createModel.argtypes = None # if the function gets no arguments
        lib.createModel.restype = c_void_p # returns a pointer of type void

        lib.sim.argtypes = [c_void_p, c_int] # takes no args
        lib.sim.restype = c_void_p    # returns a void pointer

        lib.setParams.argtypes = [c_void_p, Structure] # takes no args
        lib.setParams.restype = c_void_p    # returns a void pointer

        lib.getState.argtypes = [c_void_p] # takes no args
        lib.getState.restype = Params    # returns a void pointer

        lib.getWeights.argtypes = [c_void_p] # takes no args
        lib.getWeights.restype = ndpointer(dtype=c_double, shape=(N,N))

        # we call the constructor from the imported libpkg.so module
        self.obj = lib.createModel() # look in teh cpp code. CreateNet returns a pointer

        
  
    # in the Python wrapper, you can name these methods anything you want. Just make sure
    # you call the right C methods (that in turn call the right C++ methods)
    def createModel(self):
        lib.createModel(self.obj)

    def setParams(self, params):
        for key, typ in zip(params.keys(), self.params_c_obj._fields_):
            # if the current field must be c_int
            if typ[1].__name__ == 'c_int':
                setattr(self.params_c_obj, key, c_int(params[key]))
            # if the current field must be c_double
            if typ[1].__name__ == 'c_double':
                setattr(self.params_c_obj, key, c_double(params[key]))
        lib.setParams(self.obj, self.params_c_obj)

    def getState(self):
        resp = lib.getState(self.obj)
        return resp
    
    def getWeights(self):
        resp = lib.getWeights(self.obj)
        return resp
    
    def sim(self, interval):
        lib.sim(self.obj, interval)