import os
import numpy as np
from pandas import read_csv
from shutil import copyfile


class batcher:
        
    def __init__(self, csv, load_dir, save_dir):
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.img_list = read_csv(csv)
        self.sets = self.img_list['set'].unique()
        
    def getLoadPath (self, row):
        """
        Determine the loading path of a file (load_dir/volcano/orbit/file)
        """
        volcano = row['volcano'].lower()
        orbit = str(int(row['orbit']))
        file = row['image'] + '.tif'
        return (os.path.join(self.load_dir, volcano, orbit), file)
    
    def getSetData (self, set):
        """
        Get information on a set in form of pd.DataFrame
        """
        return self.img_list.where(self.img_list.set == set).dropna()

    def group_images(self):
        """
        Copy images into directories corresponding with the set to which they belong
        """
        bad_sets = [0]
        for s in self.sets:
            if s in bad_sets:
                print("Skipping set %i" % s)
                continue
            rows = self.getSetData(s)
            _set = self.get_serial_string(s)
            save_path = os.path.join(self.save_dir, _set)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            for _row in rows.iterrows():
                row = _row[1]
                load_path, file = self.getLoadPath(row)
                file_in = os.path.join(load_path, file)
                file_out = os.path.join(save_path, file)
                if os.path.isfile(file_in):
                    copyfile(file_in, file_out)
                else:
                    print('NOT FOUND %s' % file_in)
    
    def get_serial_string(self,serial):
        """
        Generate a fixed-length serial number string for a set
        """
        serial_string = str(serial)
        if (serial > 0):
            while len(serial_string) < 3: serial_string = '0' + serial_string
        return serial_string
