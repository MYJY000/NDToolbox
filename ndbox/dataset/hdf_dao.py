import h5py
import numpy as np

class HierarchicalFileLoader:
    def __init__(self, path: str) -> None:
        """
        path: path of your data file(*.mat, *.nwb)
        >>> file = HierarchicalFileLoader("D:/datasets/my_data.nwb")
        """
        self.path = path
        self.file = h5py.File(path)
        self.variable_list = _load_vars(self.file)
    
    def keys(self)->list[str]:
        """
        Get all the valid variable names
        >>> file.keys()
        >>> ['var1', 'var2/var21', 'var2/var22', ...]
        """
        return self.variable_list
    
    def load(self, variable):
        """
        Load the data values of given variable(must be in the variable_list)
        >>> file.load('var2/var21')
        >>> [[0.1, 0.2 .....]]
        """
        values = self.file[variable]
        if isinstance(values, h5py.Dataset):
            val_arr = self.file[variable][:]
            val_item0 = val_arr.item(0)
            if isinstance(val_item0, h5py.Reference):
                for i, x in np.ndenumerate(val_arr):
                    val_arr[i] = self.load(x)
                return val_arr
            elif isinstance(val_item0, bytes):
                return val_arr.astype(str)
            else:
                return val_arr
        if isinstance(values, h5py.Group):
            var_list = _load_vars(values)
            data = []
            for var in var_list:
                data.append(self.load(var))
            return np.array(data)


def _load_vars(file):
    var_list = []
    def valid_keys(name):
        if isinstance(file[name], h5py.Dataset):
            if file[name].shape != ():
                var_list.append(name)
    file.visit(valid_keys)
    return var_list


