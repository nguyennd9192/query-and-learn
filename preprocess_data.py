# encoding: utf-8
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
from params import *
import re, os, copy, ast
import pickle, io
from utils.utils import load_pickle
import numpy as np
from utils.general_lib import *

def get_job_get_id(structure_dir):

    slashes = [m.start() for m in re.finditer('/', structure_dir)]
    structure_name = structure_dir[slashes[-1]+1:]

    begin_job_id = structure_dir.find("/origin_struct/")+len("/origin_struct/")
    end_job_id = slashes[-1]
    structure_job = structure_dir[begin_job_id: end_job_id]

    # # convert to new format of name
    if structure_job == "check_Mo_2-22-2":
        structure_job = "Sm2-Fe22-M2"
    structure_name = structure_name.replace("*", "__")

    return structure_job, structure_name


def convert(data):
    if isinstance(data, dict):
        return {convert(key): convert(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert(element) for element in input]
    elif isinstance(data, str):
        return data.encode('utf-8')
    else:
        return data

def convert_2(data):
    # if isinstance(data, dict):   return dict(map(convert, data.items()))
    # if isinstance(data, tuple):  return map(convert, data)
    new_data = copy.copy(data)
    for key in data.keys():
        if isinstance(key, bytes):  
            new_key = key.decode('ascii')
        new_data[new_key] = data[key]
    del data
    return new_data

def convert_p3(old_pkl, new_pkl):
    """
    Convert a Python 2 pickle to Python 3
    """
    # Make a name for the new pickle

    # Convert Python 2 "ObjectType" to Python 3 object
    dill._dill._reverse_typemap["ObjectType"] = object

    # Open the pickle using latin1 encoding
    with open(old_pkl, "rb") as f:
        loaded = pickle.load(f, encoding="latin1")

    # Re-save as Python 3 pickle
    with open(new_pkl, "wb") as outfile:
        pickle.dump(loaded, outfile)

def read_deformation(qr_indexes):
    result_dir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data/SmFe12/structure_trajectory"
    jobs = ["Sm-Fe10-M2_wyckoff", 
                # "Sm-Fe11-M_wyckoff_2",
                # "Sm2-Fe22-M2", "Sm2-Fe23-M", "Sm2-Fe21-M3",
                # "Sm-Fe11-Ga1", "Sm-Fe10-Ga2",
                # "Sm2-Fe23-Ga1", "Sm2-Fe22-Ga2", "Sm2-Fe21-Ga3",
                # "CuAlZnTi_Sm2-Fe21-M3"
                ]

    # # results 
    deform_results = dict({})
    for job in jobs:
        jobfile = result_dir+"/"+job+".npy"
        # result = load_pickle(jobfile)

        # with open(jobfile, "rb") as f:
        #     result = np.load(f, encoding="bytes", allow_pickle=True) # latin-1, latin, bytes
        #     # result_cvt = convert(data=result)
        #     print (type(result))
        #     deform_results[job] = result
        
        result = np.load(jobfile,encoding="bytes",allow_pickle=True) # latin-1, latin, bytes
        result_dict = result.item()

        result_cvt = convert(data=result_dict)


        closer = result_cvt[b"coarse_relax/Sm-Fe10-M2_wyckoff/Zn_3__j8_4__i8"]
        print (closer.keys())
        print (closer[b"sum_vasp"][b"positions_ofm"])
        print (type(closer[b"sum_vasp"][b"positions_ofm"]))
        byte_str = closer[b"sum_vasp"][b"positions_ofm"]
        dict_str = byte_str.decode("UTF-8")
        mydata = ast.literal_eval(dict_str)

    for index in qr_indexes:

        structure_job, structure_name = get_job_get_id(structure_dir=index)
        # print (structure_job, structure_name)

        this_result = deform_results[structure_job]
        qr_id = "coarse_relax/"+structure_job+"/"+structure_name.replace(".poscar", "")
        if qr_id in this_result.keys():
            # print(qr_id)

            # mydata = ast.literal_eval(dict_str)
            # # this_results.keys: ['positions', 'sum_vasp', 'opt_lattice_vectors']
            # # "sum_vasp": (['volume', 'natoms', 'errors', 'ene_f', 'latt_pos_spec_charg', 
            # # 'energy', 'energy_pa', 'magmom_pa', 'positions_ofm', 'ncores', 'unit_comp', 'converged', 'runtime', 'nsteps']
            print(this_result[qr_id]["sum_vasp"])
            print(type(this_result[qr_id]["sum_vasp"]))


            # ofm = dict(this_result[qr_id]["sum_vasp"]["positions_ofm"])
            # print (ofm)
            # print (type(ofm))

            print ("======")
            break


    #   x = result["struct_infom"]["sum_vasp"]["positions_ofm"]
    #   print(x)
        # break


def query_db():
    std_dir = ALdir + "/data/standard"
    coarse_db_dir = ALdir + "/data/coarse_relax"
    fine_db_dir = ALdir + "/data/fine_relax"

    std_file = std_dir+"/summary.csv"
    coarse_file = coarse_db_dir+"/summary.csv"
    fine_file = fine_db_dir+"/summary.csv"

    # # # data base storage
    database_jobs = [
      "init/Sm-Fe11-M_wyckoff_2.csv", "init/Sm-Fe10-M2_wyckoff.csv", 
      # "Sm-Fe11-Ga1", 
      "init/Sm-Fe10-Ga2.csv",
      "init/Sm2-Fe23-M.csv", "init/check_Mo_2-22-2.csv", 
      "init/Sm2-Fe21-M3.csv",  "init/CuAlZnTi_Sm2-Fe22-M2.csv",
      "init/Sm2-Fe23-Ga1.csv", "init/Sm2-Fe22-Ga2.csv", 
      "init/Sm2-Fe21-Ga3.csv", "init/CuAlZnTi_Sm2-Fe21-M3.csv",


      "mix/query_1.csv",  "mix/supp_2.csv", "mix/supp_3.csv", "mix/supp_4.csv",  
      "mix/supp_5.csv", "mix/supp_6.csv", "mix/supp_7.csv", "mix/supp_8.csv",
      "mix/supp_9.csv", "mix/supp_10.csv",


              # "mix_2-24/query_1.csv"
              ]
    database_results = [std_dir+"/"+k for k in database_jobs]
    fine_db_rst = [fine_db_dir+"/"+k for k in database_jobs]
    coarse_db_rst = [coarse_db_dir+"/"+k for k in database_jobs]

    if os.path.isfile(std_file) and os.path.isfile(coarse_file) and os.path.isfile(fine_file):
      std_results = pd.read_csv(std_file, index_col="index_reduce")
      coarse_results = pd.read_csv(coarse_file, index_col="index_reduce")
      fine_results = pd.read_csv(fine_file, index_col="index_reduce")
      print ("Done reading database.")
    else:
      # # standard result
      frames = [pd.read_csv(k, index_col=0) for k in database_results]
      std_results = pd.concat(frames)
      index_reduce = [norm_id(k) for k in std_results.index]
      std_results["index_reduce"] = index_reduce
      std_results.set_index('index_reduce', inplace=True)

      # /Volumes/Nguyen_6TB/work/SmFe12_screening/result/standard/CoTiNH_ijk/ErFe12_I4mm_cnvg

      # # coarse, fine db
      crs_frames = [pd.read_csv(k, index_col=0) for k in coarse_db_rst]
      coarse_results = pd.concat(crs_frames)
      coarse_results = coarse_results.dropna()
      index_reduce = [norm_id(k) for k in coarse_results.index]
      coarse_results["index_reduce"] = index_reduce
      coarse_results.set_index('index_reduce', inplace=True)


      fine_frames = [pd.read_csv(k, index_col=0) for k in fine_db_rst]
      fine_results = pd.concat(fine_frames)
      fine_results = fine_results.dropna() 
      index_reduce = [norm_id(k) for k in fine_results.index]
      fine_results["index_reduce"] = index_reduce
      fine_results.set_index('index_reduce', inplace=True)
      std_results.to_csv(std_file)
      coarse_results.to_csv(coarse_file)
      fine_results.to_csv(fine_file)

    # print ("coarse_results: ", len(coarse_results))
    # print ("fine_results: ", len(fine_results))
    # print ("std_results: ", len(std_results))

    return std_results, coarse_results, fine_results
if __name__ == "__main__":
    FLAGS(sys.argv)
    # X_trval_csv, y_trval_csv, index_trval_csv, X_test_csv, y_test_csv, test_idx_csv = get_data_from_flags()
    # read_deformation(index_trval_csv)

    unlbl_job = "mix" # mix, "mix_2-24"
    ith_trial = "000"
    result_dir = get_savedir()
    filename = get_savefile()
    result_file = result_dir + "/" + filename + "_" + ith_trial +".pkl"
    unlbl_file, data, unlbl_X, unlbl_y, unlbl_index, unlbl_dir = load_unlbl_data(unlbl_job, result_file)

    # # get vasp calc data
    std_results, coarse_results, fine_results = query_db()

    # # map index to database
    data_map = map(functools.partial(id_qr_to_database, std_results=std_results,
      coarse_results=coarse_results, fine_results=fine_results), unlbl_index) 

    # # get
    y_map = np.array(list(data_map))
    print(y_map[:, 2])









